"""
Evaluation Script for Kaggle-trained Model (Epoch 74)

This script evaluates the Kaggle-trained model which uses a different
initialization signature than the main train.py model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from collections import defaultdict
import random
import json
import matplotlib.pyplot as plt


# ============================================================================
# Kaggle Model Architecture
# ============================================================================

class BiologicalAttentionMap(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ch_weights = self.channel_attention(x)
        x_ch = x * ch_weights
        sp_weights = self.spatial_attention(x_ch)
        x_attended = x_ch * sp_weights
        return x_attended, sp_weights


class TextureBranch(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True)
        )
    
    def forward(self, x, return_spatial=False):
        x = self.conv1(x)
        x = self.conv2(x)
        spatial = self.conv3(x)
        features = self.projection(spatial)
        return (features, spatial) if return_spatial else features


class SemanticBranch(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True)
        )
    
    def forward(self, x, return_spatial=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        spatial = self.conv4(x)
        features = self.projection(spatial)
        return (features, spatial) if return_spatial else features


class DualBranchFeatureExtractor(nn.Module):
    """Kaggle version - different initialization signature"""
    def __init__(self, embedding_dim=128, use_bam=True):
        super().__init__()
        self.texture_branch = TextureBranch(3, 256)
        self.semantic_branch = SemanticBranch(3, 256)
        self.use_bam = use_bam
        
        if use_bam:
            self.texture_bam = BiologicalAttentionMap(256, 16)
            self.semantic_bam = BiologicalAttentionMap(512, 16)
            combined_dim = 768
        else:
            combined_dim = 512
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        if self.use_bam:
            _, tex_spatial = self.texture_branch(x, True)
            _, sem_spatial = self.semantic_branch(x, True)
            tex_att, _ = self.texture_bam(tex_spatial)
            sem_att, _ = self.semantic_bam(sem_spatial)
            tex_pooled = F.adaptive_avg_pool2d(tex_att, (1, 1)).flatten(1)
            sem_pooled = F.adaptive_avg_pool2d(sem_att, (1, 1)).flatten(1)
            combined = torch.cat([tex_pooled, sem_pooled], dim=1)
        else:
            tex_feat = self.texture_branch(x)
            sem_feat = self.semantic_branch(x)
            combined = torch.cat([tex_feat, sem_feat], dim=1)
        
        embedding = self.fusion(combined)
        return F.normalize(embedding, p=2, dim=1)


# ============================================================================
# Dataset
# ============================================================================

class ElephantDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.samples = []
        self.identity_to_idx = {}
        self._load_dataset()
    
    def _load_dataset(self):
        identity_images = defaultdict(list)
        for category in ['Makhna', 'Herd']:
            category_dir = self.root_dir / category
            if not category_dir.exists():
                continue
            for individual_dir in category_dir.iterdir():
                if not individual_dir.is_dir():
                    continue
                identity_name = f'{category}_{individual_dir.name}'
                for img_path in individual_dir.rglob('*.jpg'):
                    identity_images[identity_name].append(img_path)
        
        all_ids = list(identity_images.keys())
        random.seed(42)
        random.shuffle(all_ids)
        n = len(all_ids)
        train_ids = all_ids[:int(0.7*n)]
        val_ids = all_ids[int(0.7*n):int(0.85*n)]
        test_ids = all_ids[int(0.85*n):]
        
        if self.split == 'train':
            selected_ids = train_ids
        elif self.split == 'val':
            selected_ids = val_ids
        else:
            selected_ids = test_ids
        
        for idx, identity_name in enumerate(selected_ids):
            self.identity_to_idx[identity_name] = idx
            for img_path in identity_images[identity_name]:
                self.samples.append({'path': img_path, 'identity': idx})
        print(f'[{self.split.upper()}] {len(self.samples)} images, {len(self.identity_to_idx)} identities')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample['path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, sample['identity']


# ============================================================================
# Evaluation
# ============================================================================

def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT = Path("data/processed_megadetector")
    model_path = Path("src/models/best_model.pth")
    results_dir = Path("outputs/results/kaggle_model_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Kaggle Model Evaluation (Epoch 74)")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Results: {results_dir}")
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(str(model_path), map_location=device)
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    model = DualBranchFeatureExtractor(embedding_dim=128, use_bam=True).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Load test dataset
    print("\nLoading test dataset...")
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = ElephantDataset(DATA_ROOT, val_transform, 'test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Extract features
    print("\nExtracting features...")
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            embeddings = model(images)
            features_list.append(embeddings.cpu())
            labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    print(f"Features shape: {features.shape}")
    
    # Split query/gallery
    num_samples = len(features)
    query_indices = list(range(0, num_samples, 2))
    gallery_indices = list(range(1, num_samples, 2))
    
    query_features = features[query_indices]
    query_labels = labels[query_indices]
    gallery_features = features[gallery_indices]
    gallery_labels = labels[gallery_indices]
    
    print(f"Query: {len(query_features)} samples")
    print(f"Gallery: {len(gallery_features)} samples")
    
    # Compute distances
    print("\nComputing similarity matrix...")
    dist_matrix = torch.cdist(query_features, gallery_features, p=2).numpy()
    
    # Evaluate
    print("Evaluating...")
    num_queries = len(query_labels)
    rank1, rank5, rank10 = 0, 0, 0
    average_precisions = []
    
    for i in range(num_queries):
        query_label = query_labels[i].item()
        sorted_indices = np.argsort(dist_matrix[i])
        sorted_gallery_labels = gallery_labels[sorted_indices].numpy()
        matches = sorted_gallery_labels == query_label
        
        if matches[0]:
            rank1 += 1
        if matches[:5].any():
            rank5 += 1
        if matches[:10].any():
            rank10 += 1
        
        if matches.sum() > 0:
            relevant_positions = np.where(matches)[0]
            precisions = [(k+1)/(pos+1) for k, pos in enumerate(relevant_positions)]
            average_precisions.append(np.mean(precisions))
    
    # Results
    print("\n" + "="*80)
    print("EVALUATION RESULTS - KAGGLE MODEL (EPOCH 74)")
    print("="*80)
    print(f"Rank-1 Accuracy:  {rank1/num_queries*100:.2f}%")
    print(f"Rank-5 Accuracy:  {rank5/num_queries*100:.2f}%")
    print(f"Rank-10 Accuracy: {rank10/num_queries*100:.2f}%")
    print(f"mAP:              {np.mean(average_precisions)*100:.2f}%")
    print(f"Num queries:      {num_queries}")
    print("="*80)
    
    # Save results
    metrics = {
        'Rank-1': rank1/num_queries,
        'Rank-5': rank5/num_queries,
        'Rank-10': rank10/num_queries,
        'mAP': float(np.mean(average_precisions)),
        'num_queries': num_queries,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'model_type': 'Kaggle trained (768-dim fusion)'
    }
    
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Results saved to {results_dir}/metrics.json")


if __name__ == "__main__":
    main()
