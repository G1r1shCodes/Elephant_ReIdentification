"""
Training Script for Dual-Branch Feature Extractor

Implements metric learning for elephant re-identification using:
- Triplet loss with hard negative mining
- Data augmentation
- Learning rate scheduling
- Model checkpointing
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from dual_branch_extractor import DualBranchFeatureExtractor


# ==================== Configuration ====================

class Config:
    """Training configuration."""
    
    # Paths (resolve to absolute paths)
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "processed_megadetector"
    CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "models"
    LOG_DIR = PROJECT_ROOT / "outputs" / "results"

    
    # Model (Updated to match methodology)
    INPUT_CHANNELS = 3
    TEXTURE_DIM = 256
    SEMANTIC_DIM = 256
    EMBEDDING_DIM = 128  # 128-dim as per methodology
    USE_BAM = True       # Use Biological Attention Map
    
    # Training
    BATCH_SIZE = 16  # Reduced for small dataset (23 identities)
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Triplet loss
    MARGIN = 0.3
    MINING_STRATEGY = "hard"  # 'hard', 'semi-hard', 'all'
    
    # Data
    IMAGE_SIZE = (224, 224)
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Optimization
    LR_SCHEDULER = "cosine"  # 'cosine', 'step', 'plateau'
    WARMUP_EPOCHS = 5
    
    # Checkpointing
    SAVE_FREQ = 5  # Save every N epochs
    EARLY_STOP_PATIENCE = 15
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4


# ==================== Dataset ====================

class ElephantDataset(Dataset):
    """
    Elephant re-identification dataset.
    
    Expects directory structure:
    data/processed/
        Makhna/
            Individual_1/
                image1.jpg
                image2.jpg
            Individual_2/
                ...
        Herd/
            Individual_1/
                ...
    """
    
    def __init__(self, root_dir: Path, transform=None, split='train'):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory of processed data
            transform: Image transformations
            split: 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Collect all images and their identity labels
        self.samples = []
        self.identity_to_idx = {}
        self.idx_to_identity = {}
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load all images and create identity mappings with train/val/test split."""
        import random
        
        # First pass: collect all identities and their images
        identity_images = defaultdict(list)
        identity_categories = {}
        
        # Process both Makhna and Herd directories
        for category in ['Makhna', 'Herd']:
            category_dir = self.root_dir / category
            if not category_dir.exists():
                continue
            
            # Each subdirectory is an individual elephant
            for individual_dir in category_dir.iterdir():
                if not individual_dir.is_dir():
                    continue
                
                identity_name = f"{category}_{individual_dir.name}"
                identity_categories[identity_name] = category
                
                # Collect all images for this individual
                for img_path in individual_dir.rglob('*.jpg'):
                    identity_images[identity_name].append(img_path)
        
        # Split identities into train/val/test (70/15/15)
        # Important: Split at IDENTITY level, not image level
        all_identities = list(identity_images.keys())
        random.seed(42)  # For reproducibility
        random.shuffle(all_identities)
        
        n_identities = len(all_identities)
        n_train = int(0.7 * n_identities)
        n_val = int(0.15 * n_identities)
        
        train_identities = all_identities[:n_train]
        val_identities = all_identities[n_train:n_train + n_val]
        test_identities = all_identities[n_train + n_val:]
        
        # Select identities based on split
        if self.split == 'train':
            selected_identities = train_identities
        elif self.split == 'val':
            selected_identities = val_identities
        elif self.split == 'test':
            selected_identities = test_identities
        else:
            selected_identities = all_identities  # Use all if split not specified
        
        # Create identity mappings and collect samples
        identity_idx = 0
        for identity_name in selected_identities:
            self.identity_to_idx[identity_name] = identity_idx
            self.idx_to_identity[identity_idx] = identity_name
            
            # Add all images for this identity
            for img_path in identity_images[identity_name]:
                self.samples.append({
                    'path': img_path,
                    'identity': identity_idx,
                    'category': identity_categories[identity_name]
                })
            
            identity_idx += 1
        
        print(f"[{self.split.upper()}] Loaded {len(self.samples)} images from {len(self.identity_to_idx)} individuals")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item by index.
        
        Returns:
            image: Preprocessed image tensor
            identity: Identity label
            category: 'Makhna' or 'Herd'
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, sample['identity'], sample['category']


# ==================== Data Augmentation ====================

def get_train_transforms(image_size):
    """Get training data augmentation pipeline with Random Erasing."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        # Random Erasing to prevent arrow bias (as per methodology)
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])

def get_val_transforms(image_size):
    """Get validation/test data transforms."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


# ==================== Triplet Loss ====================

class TripletLoss(nn.Module):
    """
    Triplet loss with hard negative mining.
    
    For each anchor, selects:
    - Hardest positive (same identity, farthest distance)
    - Hardest negative (different identity, closest distance)
    """
    
    def __init__(self, margin=0.3, mining='hard'):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            mining: Mining strategy ('hard', 'semi-hard', 'all')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(self, embeddings, labels):
        """
        Compute triplet loss.
        
        Args:
            embeddings: Feature embeddings [batch_size, feature_dim]
            labels: Identity labels [batch_size]
            
        Returns:
            loss: Triplet loss value
        """
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)
        
        # Mine triplets
        if self.mining == 'hard':
            loss = self._hard_triplet_loss(distances, labels)
        else:
            raise NotImplementedError(f"Mining strategy '{self.mining}' not implemented")
        
        return loss
    
    def _pairwise_distances(self, embeddings):
        """Compute pairwise Euclidean distances."""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        
        # Fix numerical errors
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
        
        return distances
    
    def _hard_triplet_loss(self, distances, labels):
        """Hard negative mining triplet loss."""
        batch_size = labels.size(0)
        
        # Get hardest positive and negative for each anchor
        loss = 0.0
        num_valid_triplets = 0
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Positive mask (same identity, excluding anchor itself)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size, device=labels.device) != i)
            
            # Negative mask (different identity)
            negative_mask = labels != anchor_label
            
            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue
            
            # Hardest positive (farthest same-identity sample)
            hardest_positive_dist = distances[i][positive_mask].max()
            
            # Hardest negative (closest different-identity sample)
            hardest_negative_dist = distances[i][negative_mask].min()
            
            # Triplet loss
            triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
            
            loss += triplet_loss
            num_valid_triplets += 1
        
        if num_valid_triplets > 0:
            loss = loss / num_valid_triplets
        
        return loss


# ==================== Training ====================

class Trainer:
    """Training manager for dual-branch feature extractor."""
    
    def __init__(self, config: Config):
        """Initialize trainer."""
        self.config = config
        
        # Create directories
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize model (updated for methodology)
        self.model = DualBranchFeatureExtractor(
            input_channels=config.INPUT_CHANNELS,
            texture_dim=config.TEXTURE_DIM,
            semantic_dim=config.SEMANTIC_DIM,
            embedding_dim=config.EMBEDDING_DIM,
            use_bam=config.USE_BAM
        ).to(config.DEVICE)
        
        # Loss and optimizer
        self.criterion = TripletLoss(margin=config.MARGIN, mining=config.MINING_STRATEGY)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        if config.LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for images, labels, _ in pbar:
            images = images.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(images)
            
            # Compute loss
            loss = self.criterion(embeddings, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)
                
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)
                
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        checkpoint_path = self.config.CHECKPOINT_DIR / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"Starting training on {self.config.DEVICE}")
        print(f"Total epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print("=" * 80)
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.SAVE_FREQ == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            print("=" * 80)
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


# ==================== Main ====================

def main():
    """Main training function."""
    # Configuration
    config = Config()
    
    print("Elephant Re-Identification - Training")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Data root: {config.DATA_ROOT}")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset = ElephantDataset(
        root_dir=config.DATA_ROOT,
        transform=get_train_transforms(config.IMAGE_SIZE),
        split='train'
    )
    
    val_dataset = ElephantDataset(
        root_dir=config.DATA_ROOT,
        transform=get_val_transforms(config.IMAGE_SIZE),
        split='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Identities: {len(train_dataset.identity_to_idx)}")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
