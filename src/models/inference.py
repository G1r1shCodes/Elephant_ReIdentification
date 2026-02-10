"""
Inference Script for Elephant Re-Identification

Simple script to:
1. Load a trained model
2. Extract features from images
3. Find similar elephants in a gallery
"""

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

from dual_branch_extractor import DualBranchFeatureExtractor
from train import Config


class ElephantReidentifier:
    """Simple re-identification interface."""
    
    def __init__(self, model_path: str, device=None):
        """
        Initialize re-identifier.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize config
        config = Config()
        
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_feature(self, image_path: str) -> torch.Tensor:
        """
        Extract feature from a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            feature: [embedding_dim] feature vector
        """
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract feature
        with torch.no_grad():
            feature = self.model(image_tensor)
        
        return feature.cpu().squeeze(0)
    
    def compute_similarity(self, feature1: torch.Tensor, feature2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two features.
        
        Args:
            feature1: First feature vector
            feature2: Second feature vector
            
        Returns:
            similarity: Similarity score (higher = more similar)
        """
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            feature1.unsqueeze(0), 
            feature2.unsqueeze(0)
        )
        return similarity.item()
    
    def find_matches(self, query_image: str, gallery_images: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find top-k most similar images from gallery.
        
        Args:
            query_image: Path to query image
            gallery_images: List of gallery image paths
            top_k: Number of top matches to return
            
        Returns:
            matches: List of (image_path, similarity_score) tuples
        """
        # Extract query feature
        print("Extracting query feature...")
        query_feature = self.extract_feature(query_image)
        
        # Extract gallery features and compute similarities
        print("Searching gallery...")
        similarities = []
        for gallery_path in gallery_images:
            gallery_feature = self.extract_feature(gallery_path)
            similarity = self.compute_similarity(query_feature, gallery_feature)
            similarities.append((gallery_path, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def visualize_matches(self, query_image: str, matches: List[Tuple[str, float]], 
                         save_path: str = None):
        """
        Visualize query and top matches.
        
        Args:
            query_image: Path to query image
            matches: List of (image_path, similarity) from find_matches
            save_path: Optional path to save figure
        """
        num_matches = len(matches)
        fig, axes = plt.subplots(1, num_matches + 1, figsize=(15, 3))
        
        # Plot query
        query_img = cv2.imread(query_image)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(query_img)
        axes[0].set_title("Query", fontweight='bold', color='blue')
        axes[0].axis('off')
        
        # Plot matches
        for i, (match_path, similarity) in enumerate(matches):
            match_img = cv2.imread(match_path)
            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            axes[i+1].imshow(match_img)
            axes[i+1].set_title(f"Match {i+1}\nSim: {similarity:.3f}", fontweight='bold')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def demo():
    """Demo script showing how to use the re-identifier."""
    
    # Paths
    config = Config()
    model_path = config.CHECKPOINT_DIR / "best_model.pth"
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Initialize re-identifier
    reidentifier = ElephantReidentifier(str(model_path))
    
    print("\n" + "=" * 80)
    print("Elephant Re-Identification - Demo")
    print("=" * 80)
    
    # Example: Find matches in a directory
    data_dir = config.DATA_ROOT
    
    # Get some sample images
    all_images = list(data_dir.rglob("*.jpg"))[:50]  # Limit to 50 for demo
    
    if len(all_images) < 2:
        print("❌ Not enough images found in data directory")
        print(f"Please ensure images exist in {data_dir}")
        return
    
    # Use first image as query, rest as gallery
    query_image = str(all_images[0])
    gallery_images = [str(img) for img in all_images[1:]]
    
    print(f"\nQuery image: {Path(query_image).name}")
    print(f"Gallery size: {len(gallery_images)} images")
    
    # Find matches
    print("\nSearching for matches...")
    matches = reidentifier.find_matches(query_image, gallery_images, top_k=5)
    
    # Print results
    print("\n" + "=" * 80)
    print("TOP 5 MATCHES")
    print("=" * 80)
    for i, (match_path, similarity) in enumerate(matches):
        print(f"{i+1}. {Path(match_path).name} (Similarity: {similarity:.4f})")
    print("=" * 80)
    
    # Visualize
    output_dir = config.LOG_DIR / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "demo_matches.png"
    
    reidentifier.visualize_matches(query_image, matches, str(save_path))
    
    print(f"\n✅ Demo completed!")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    demo()
