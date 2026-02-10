"""
Visualize Biological Attention Maps (BAM)

This script demonstrates how BAM learns to focus on biologically relevant regions.
It creates a simple visualization showing:
1. Input image (simulated)
2. Texture attention map
3. Semantic attention map
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.models.dual_branch_extractor import DualBranchFeatureExtractor


def visualize_bam_attention():
    """Visualize BAM attention maps on a sample input."""
    print("\n" + "="*80)
    print("BIOLOGICAL ATTENTION MAP (BAM) VISUALIZATION")
    print("="*80)
    
    # Create model with BAM
    model = DualBranchFeatureExtractor(
        input_channels=3,
        texture_dim=256,
        semantic_dim=256,
        embedding_dim=128,
        use_bam=True
    )
    model.eval()
    
    # Create a simulated input (random for demonstration)
    # In practice, this would be a real elephant image
    test_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass with attention maps
    with torch.no_grad():
        embedding, texture_att, semantic_att = model(test_input, return_attention_maps=True)
    
    # Convert to numpy for visualization
    input_img = test_input[0].permute(1, 2, 0).numpy()
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    
    texture_att_np = texture_att[0, 0].numpy()
    semantic_att_np = semantic_att[0, 0].numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image\n(Simulated)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Texture attention
    im1 = axes[1].imshow(texture_att_np, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f'Texture Branch Attention\n(High Resolution: {texture_att_np.shape})', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Semantic attention
    im2 = axes[2].imshow(semantic_att_np, cmap='hot', interpolation='bilinear')
    axes[2].set_title(f'Semantic Branch Attention\n(Low Resolution: {semantic_att_np.shape})', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'bam_attention_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("ATTENTION STATISTICS")
    print("="*80)
    print(f"\nTexture Attention:")
    print(f"  Shape: {texture_att_np.shape}")
    print(f"  Range: [{texture_att_np.min():.3f}, {texture_att_np.max():.3f}]")
    print(f"  Mean: {texture_att_np.mean():.3f}")
    print(f"  Std: {texture_att_np.std():.6f}")
    
    print(f"\nSemantic Attention:")
    print(f"  Shape: {semantic_att_np.shape}")
    print(f"  Range: [{semantic_att_np.min():.3f}, {semantic_att_np.max():.3f}]")
    print(f"  Mean: {semantic_att_np.mean():.3f}")
    print(f"  Std: {semantic_att_np.std():.6f}")
    
    print("\n" + "="*80)
    print("EXPECTED BEHAVIOR AFTER TRAINING")
    print("="*80)
    print("\n📋 Texture Branch (High Resolution):")
    print("  Expected to focus on:")
    print("    • Ear depigmentation (pink spots)")
    print("    • Ear tears and notches")
    print("    • Skin texture patterns")
    print("    • Fine-grained local details")
    
    print("\n📋 Semantic Branch (Low Resolution):")
    print("  Expected to focus on:")
    print("    • Body bulk (Makhnas)")
    print("    • Head dome shape (Calves)")
    print("    • Ear curvature")
    print("    • Overall proportions")
    
    print("\n💡 Note:")
    print("  With random initialization, attention is relatively uniform.")
    print("  During training with metric learning, BAM will learn to focus on")
    print("  biologically discriminative regions specific to each elephant.")
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    
    return output_path


if __name__ == "__main__":
    try:
        output_path = visualize_bam_attention()
        print(f"\n🎯 Open the visualization at: {output_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: This script requires matplotlib. Install with:")
        print("  pip install matplotlib")
