"""
Evaluation Script for Dual-Branch Feature Extractor

Evaluates the trained model on test set with metrics:
- Rank-1, Rank-5, Rank-10 accuracy
- Mean Average Precision (mAP)
- Confusion matrix
- Feature visualization
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

from dual_branch_extractor import DualBranchFeatureExtractor
from train import Config, ElephantDataset, get_val_transforms


class Evaluator:
    """Model evaluator for re-identification."""
    
    def __init__(self, model_path, config: Config):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object
        """
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = DualBranchFeatureExtractor(
            input_channels=config.INPUT_CHANNELS,
            texture_dim=config.TEXTURE_DIM,
            semantic_dim=config.SEMANTIC_DIM,
            embedding_dim=config.EMBEDDING_DIM,
            use_bam=config.USE_BAM
        ).to(self.device)
        
        # Handle both checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            # Assume checkpoint is the state dict directly
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        if 'best_val_loss' in checkpoint:
            print(f"✓ Best validation loss: {checkpoint['best_val_loss']}")
        elif 'losses' in checkpoint:
            print(f"✓ Losses: {checkpoint['losses']}")
    
    def extract_features(self, data_loader):
        """
        Extract features for all images in dataset.
        
        Returns:
            features: [N, embedding_dim] feature matrix
            labels: [N] identity labels
            paths: [N] image paths
        """
        features_list = []
        labels_list = []
        paths_list = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(data_loader, desc="Extracting features"):
                images = images.to(self.device)
                
                # Extract features
                embeddings = self.model(images)
                
                features_list.append(embeddings.cpu())
                labels_list.append(labels)
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return features, labels
    
    def compute_distance_matrix(self, query_features, gallery_features):
        """
        Compute pairwise Euclidean distance matrix.
        
        Args:
            query_features: [N_q, D] query features
            gallery_features: [N_g, D] gallery features
            
        Returns:
            distances: [N_q, N_g] distance matrix
        """
        # Compute Euclidean distances
        distances = torch.cdist(query_features, gallery_features, p=2)
        return distances.numpy()
    
    def evaluate_ranking(self, query_features, query_labels, gallery_features, gallery_labels):
        """
        Evaluate ranking metrics (Rank-1, Rank-5, Rank-10, mAP).
        
        Args:
            query_features: Query set features
            query_labels: Query set labels
            gallery_features: Gallery set features
            gallery_labels: Gallery set labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(query_features, gallery_features)
        
        num_queries = len(query_labels)
        
        # Rank accuracy
        rank1_correct = 0
        rank5_correct = 0
        rank10_correct = 0
        
        # mAP calculation
        average_precisions = []
        
        for i in range(num_queries):
            query_label = query_labels[i].item()
            
            # Get sorted indices (nearest to farthest)
            sorted_indices = np.argsort(dist_matrix[i])
            
            # Get gallery labels in sorted order
            sorted_gallery_labels = gallery_labels[sorted_indices].numpy()
            
            # Find matches
            matches = sorted_gallery_labels == query_label
            
            # Rank-k accuracy
            if matches[0]:
                rank1_correct += 1
            if matches[:5].any():
                rank5_correct += 1
            if matches[:10].any():
                rank10_correct += 1
            
            # Average Precision
            if matches.sum() > 0:
                # Precision at each relevant position
                relevant_positions = np.where(matches)[0]
                precisions = []
                for k, pos in enumerate(relevant_positions):
                    precision_at_k = (k + 1) / (pos + 1)
                    precisions.append(precision_at_k)
                average_precisions.append(np.mean(precisions))
        
        metrics = {
            'Rank-1': rank1_correct / num_queries,
            'Rank-5': rank5_correct / num_queries,
            'Rank-10': rank10_correct / num_queries,
            'mAP': np.mean(average_precisions) if average_precisions else 0.0,
            'num_queries': num_queries
        }
        
        return metrics
    
    def plot_confusion_matrix(self, query_labels, predictions, identity_names, save_path):
        """
        Plot confusion matrix.
        
        Args:
            query_labels: True labels
            predictions: Predicted labels (Rank-1)
            identity_names: List of identity names
            save_path: Where to save plot
        """
        cm = confusion_matrix(query_labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=identity_names, yticklabels=identity_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Rank-1)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"✓ Saved confusion matrix to {save_path}")
    
    def visualize_retrieval(self, query_idx, query_features, query_labels, 
                          gallery_features, gallery_labels, dataset, save_path, top_k=5):
        """
        Visualize top-k retrieval results for a query.
        
        Args:
            query_idx: Index of query sample
            query_features: All query features
            query_labels: All query labels
            gallery_features: All gallery features
            gallery_labels: All gallery labels
            dataset: Dataset object (to access images)
            save_path: Where to save visualization
            top_k: Number of top results to show
        """
        # Compute distances for this query
        query_feature = query_features[query_idx:query_idx+1]
        distances = torch.cdist(query_feature, gallery_features, p=2)[0].numpy()
        
        # Get top-k nearest neighbors
        sorted_indices = np.argsort(distances)[:top_k]
        
        # Create visualization
        fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 3))
        
        # Plot query image
        query_img = dataset.samples[query_idx]['path']
        query_label = query_labels[query_idx].item()
        axes[0].imshow(plt.imread(query_img))
        axes[0].set_title(f"Query\nID: {query_label}", color='blue', fontweight='bold')
        axes[0].axis('off')
        
        # Plot top-k results
        for i, idx in enumerate(sorted_indices):
            gallery_img = dataset.samples[idx]['path']
            gallery_label = gallery_labels[idx].item()
            
            # Green for correct match, red for incorrect
            is_correct = gallery_label == query_label
            color = 'green' if is_correct else 'red'
            
            axes[i+1].imshow(plt.imread(gallery_img))
            axes[i+1].set_title(f"Rank-{i+1}\nID: {gallery_label}\nDist: {distances[idx]:.2f}", 
                              color=color, fontweight='bold')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved retrieval visualization to {save_path}")


def main():
    """Main evaluation function."""
    
    # Configuration
    config = Config()
    
    # Paths
    model_path = config.CHECKPOINT_DIR / "best_model.pth"
    results_dir = config.LOG_DIR / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Elephant Re-Identification - Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Results: {results_dir}")
    print(f"Device: {config.DEVICE}")
    print("=" * 80)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ElephantDataset(
        root_dir=config.DATA_ROOT,
        transform=get_val_transforms(config.IMAGE_SIZE),
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test identities: {len(test_dataset.identity_to_idx)}")
    
    # Initialize evaluator
    evaluator = Evaluator(model_path, config)
    
    # Extract features
    print("\nExtracting features...")
    features, labels = evaluator.extract_features(test_loader)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # For re-identification, split into query and gallery
    # Use half as queries, half as gallery
    num_samples = len(features)
    query_indices = list(range(0, num_samples, 2))
    gallery_indices = list(range(1, num_samples, 2))
    
    query_features = features[query_indices]
    query_labels = labels[query_indices]
    gallery_features = features[gallery_indices]
    gallery_labels = labels[gallery_indices]
    
    print(f"\nQuery set: {len(query_features)} samples")
    print(f"Gallery set: {len(gallery_features)} samples")
    
    # Evaluate ranking
    print("\nEvaluating ranking metrics...")
    metrics = evaluator.evaluate_ranking(
        query_features, query_labels,
        gallery_features, gallery_labels
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Rank-1 Accuracy:  {metrics['Rank-1']*100:.2f}%")
    print(f"Rank-5 Accuracy:  {metrics['Rank-5']*100:.2f}%")
    print(f"Rank-10 Accuracy: {metrics['Rank-10']*100:.2f}%")
    print(f"mAP:              {metrics['mAP']*100:.2f}%")
    print("=" * 80)
    
    # Save metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Visualize few retrieval examples
    print("\nGenerating visualizations...")
    for i in range(min(5, len(query_features))):
        save_path = results_dir / f"retrieval_query_{i}.png"
        evaluator.visualize_retrieval(
            query_indices[i], features, labels,
            features, labels, test_dataset, save_path
        )
    
    print("\n✅ Evaluation completed!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
