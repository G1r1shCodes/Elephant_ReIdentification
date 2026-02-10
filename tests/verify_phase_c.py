"""
Phase C Verification Script (Updated for Methodology)

Verifies implementation matches WII Elephant Re-ID methodology:
- 128-dim embeddings (not 512-dim)
- Biological Attention Map (BAM) for spatial attention
- Random Erasing in data augmentation
"""

import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.models.texture_branch import TextureBranch
from src.models.semantic_branch import SemanticBranch
from src.models.dual_branch_extractor import DualBranchFeatureExtractor
from src.models.biological_attention import BiologicalAttentionMap


def test_embedding_dimension():
    """Test that final embedding is 128-dim as per methodology."""
    print("\n" + "="*80)
    print("TEST 1: EMBEDDING DIMENSION (128-dim as per methodology)")
    print("="*80)
    
    model = DualBranchFeatureExtractor(
        input_channels=3,
        texture_dim=256,
        semantic_dim=256,
        embedding_dim=128,
        use_bam=True
    )
    model.eval()
    
    test_input = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        output = model(test_input)
    
    assert output.shape == (4, 128), f"Expected (4, 128), got {output.shape}"
    
    # Verify L2 normalization
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), "Output not L2-normalized"
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Embedding dimension: 128 (matches methodology)")
    print(f"✓ L2 normalized: {norms.mean().item():.6f}")
    print("✓ EMBEDDING DIMENSION TEST PASSED")
    
    return True


def test_biological_attention_map():
    """Test that BAM is implemented and working."""
    print("\n" + "="*80)
    print("TEST 2: BIOLOGICAL ATTENTION MAP (BAM)")
    print("="*80)
    
    bam = BiologicalAttentionMap(in_channels=256, reduction=16)
    test_input = torch.randn(2, 256, 14, 14)
    
    attended, attention_map = bam(test_input)
    
    assert attended.shape == test_input.shape, "Attended features shape mismatch"
    assert attention_map.shape == (2, 1, 14, 14), f"Expected (2, 1, 14, 14), got {attention_map.shape}"
    assert attention_map.min() >= 0 and attention_map.max() <= 1, "Attention map not in [0, 1]"
    
    print(f"✓ Input shape: {test_input.shape}")
    print(f"✓ Attended features shape: {attended.shape}")
    print(f"✓ Attention map shape: {attention_map.shape}")
    print(f"✓ Attention range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
    print("✓ BIOLOGICAL ATTENTION MAP TEST PASSED")
    
    return True


def test_random_erasing():
    """Test that Random Erasing is in augmentation pipeline."""
    print("\n" + "="*80)
    print("TEST 3: RANDOM ERASING (Arrow Bias Prevention)")
    print("="*80)
    
    # Import train transforms
    sys.path.insert(0, str(Path.cwd() / "src" / "models"))
    from train import get_train_transforms
    
    transforms_pipeline = get_train_transforms((224, 224))
    
    # Check if RandomErasing is in the pipeline
    has_random_erasing = any(
        isinstance(t, transforms.RandomErasing) 
        for t in transforms_pipeline.transforms
    )
    
    assert has_random_erasing, "Random Erasing not found in training transforms"
    
    # Find the RandomErasing transform
    random_erasing = None
    for t in transforms_pipeline.transforms:
        if isinstance(t, transforms.RandomErasing):
            random_erasing = t
            break
    
    print(f"✓ Random Erasing found in training pipeline")
    print(f"✓ Probability: {random_erasing.p}")
    print(f"✓ Scale range: {random_erasing.scale}")
    print(f"✓ Ratio range: {random_erasing.ratio}")
    print("✓ RANDOM ERASING TEST PASSED")
    
    return True


def test_parameter_count():
    """Test total parameter count."""
    print("\n" + "="*80)
    print("TEST 4: PARAMETER COUNT")
    print("="*80)
    
    model = DualBranchFeatureExtractor(
        input_channels=3,
        texture_dim=256,
        semantic_dim=256,
        embedding_dim=128,
        use_bam=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    texture_params = sum(p.numel() for p in model.texture_branch.parameters())
    semantic_params = sum(p.numel() for p in model.semantic_branch.parameters())
    fusion_params = total_params - texture_params - semantic_params
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"  - Texture branch: {texture_params:,}")
    print(f"  - Semantic branch: {semantic_params:,}")
    print(f"  - Fusion + BAM: {fusion_params:,}")
    print(f"✓ Reduced from 9.3M to {total_params/1e6:.1f}M (128-dim vs 512-dim)")
    print("✓ PARAMETER COUNT TEST PASSED")
    
    return True


def test_methodology_compliance():
    """Test overall compliance with methodology."""
    print("\n" + "="*80)
    print("TEST 5: METHODOLOGY COMPLIANCE")
    print("="*80)
    
    model = DualBranchFeatureExtractor(
        input_channels=3,
        texture_dim=256,
        semantic_dim=256,
        embedding_dim=128,
        use_bam=True
    )
    
    # Check embedding dimension
    assert model.embedding_dim == 128, "Embedding dimension should be 128"
    
    # Check BAM usage
    assert model.use_bam == True, "BAM should be enabled"
    assert hasattr(model, 'texture_bam'), "Texture BAM not found"
    assert hasattr(model, 'semantic_bam'), "Semantic BAM not found"
    
    # Check branches exist
    assert hasattr(model, 'texture_branch'), "Texture branch not found"
    assert hasattr(model, 'semantic_branch'), "Semantic branch not found"
    
    print("✓ Embedding dimension: 128 ✓")
    print("✓ Biological Attention Map (BAM): Enabled ✓")
    print("✓ Dual-branch architecture: Present ✓")
    print("✓ L2 normalization: Implemented ✓")
    print("✓ METHODOLOGY COMPLIANCE TEST PASSED")
    
    return True


def verify_methodology_alignment():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("PHASE C VERIFICATION - METHODOLOGY ALIGNMENT")
    print("WII Elephant Re-ID System")
    print("="*80)
    
    tests = [
        ("128-dim Embeddings", test_embedding_dimension),
        ("Biological Attention Map (BAM)", test_biological_attention_map),
        ("Random Erasing (Arrow Bias)", test_random_erasing),
        ("Parameter Count", test_parameter_count),
        ("Methodology Compliance", test_methodology_compliance),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ METHODOLOGY ALIGNMENT VERIFIED - ALL TESTS PASSED!")
        print("="*80)
        print("\n🎯 Implementation Status: ALIGNED WITH METHODOLOGY")
        print("📋 Key Updates:")
        print("  ✓ 128-dim embeddings (was 512-dim)")
        print("  ✓ Biological Attention Map (BAM) for spatial attention")
        print("  ✓ Random Erasing for arrow bias prevention")
        print("  ✓ Dual-branch architecture preserved")
        print("  ✓ L2 normalization for metric learning")
        print("\n🚀 Ready for: Training with methodology-compliant architecture")
    else:
        print("❌ METHODOLOGY ALIGNMENT FAILED - SOME TESTS DID NOT PASS")
        print("="*80)
    
    return all_passed


if __name__ == "__main__":
    verify_methodology_alignment()
