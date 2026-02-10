"""
Dual-Branch Feature Extractor

Combines texture and semantic branches to handle biological heterogeneity
in elephant re-identification.

Architecture (Updated to match methodology):
- Texture Branch: Fine-grained local details (ears, skin)
- Semantic Branch: Global shape and structure (body, proportions)
- Biological Attention Map (BAM): Spatial attention on biometric regions
- Fusion: Concatenation + projection to 128-dim embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .texture_branch import TextureBranch
    from .semantic_branch import SemanticBranch
    from .biological_attention import BiologicalAttentionMap
except ImportError:
    from texture_branch import TextureBranch
    from semantic_branch import SemanticBranch
    from biological_attention import BiologicalAttentionMap



class DualBranchFeatureExtractor(nn.Module):
    """
    Dual-branch architecture for biologically-aware feature extraction.
    
    Updated to match WII methodology:
    - 128-dim final embedding (not 512-dim)
    - Spatial attention via Biological Attention Map (BAM)
    - Handles heterogeneity across sex and age groups
    """
    
    def __init__(self, 
                 input_channels=3,
                 texture_dim=256,
                 semantic_dim=256,
                 embedding_dim=128,  # Changed from fusion_dim=512
                 use_bam=True):      # Use Biological Attention Map
        """
        Initialize dual-branch feature extractor.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            texture_dim: Dimension of texture branch output
            semantic_dim: Dimension of semantic branch output
            embedding_dim: Final embedding dimension (128 as per methodology)
            use_bam: Whether to use Biological Attention Map
        """
        super(DualBranchFeatureExtractor, self).__init__()

        
        # Two parallel branches
        self.texture_branch = TextureBranch(
            input_channels=input_channels,
            feature_dim=texture_dim
        )
        
        self.semantic_branch = SemanticBranch(
            input_channels=input_channels,
            feature_dim=semantic_dim
        )
        
        self.use_bam = use_bam
        
        # Biological Attention Maps for each branch
        if use_bam:
            # Apply BAM before pooling in each branch
            # We'll apply it to intermediate features
            self.texture_bam = BiologicalAttentionMap(in_channels=256, reduction=16)
            self.semantic_bam = BiologicalAttentionMap(in_channels=512, reduction=16)
            # When using BAM, we concatenate attended spatial features: 256 + 512 = 768
            combined_dim = 256 + 512
        else:
            # When not using BAM, we concatenate branch output features
            combined_dim = texture_dim + semantic_dim
        
        # Project to final embedding dimension (128-dim as per methodology)
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim


        
    def forward(self, x, return_branch_features=False, return_attention_maps=False):
        """
        Forward pass through dual-branch extractor.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            return_branch_features: If True, return individual branch features
            return_attention_maps: If True, return BAM attention maps
            
        Returns:
            If return_branch_features=False and return_attention_maps=False:
                embedding: Final embedding [batch_size, embedding_dim]
            If return_branch_features=True:
                (embedding, texture_features, semantic_features)
            If return_attention_maps=True:
                (embedding, texture_attention_map, semantic_attention_map)
        """
        # Initialize attention maps
        texture_attention_map = None
        semantic_attention_map = None
        
        if self.use_bam:
            # Extract spatial features from both branches
            texture_features, texture_spatial = self.texture_branch(x, return_spatial_features=True)
            semantic_features, semantic_spatial = self.semantic_branch(x, return_spatial_features=True)
            
            # Apply Biological Attention Map to spatial features
            texture_attended, texture_attention_map = self.texture_bam(texture_spatial)
            semantic_attended, semantic_attention_map = self.semantic_bam(semantic_spatial)
            
            # Pool the attended spatial features
            texture_attended_pooled = F.adaptive_avg_pool2d(texture_attended, (1, 1))
            semantic_attended_pooled = F.adaptive_avg_pool2d(semantic_attended, (1, 1))
            
            # Flatten
            texture_attended_flat = texture_attended_pooled.view(texture_attended_pooled.size(0), -1)
            semantic_attended_flat = semantic_attended_pooled.view(semantic_attended_pooled.size(0), -1)
            
            # Project to branch feature dimensions
            # Note: We already have texture_features and semantic_features from branches
            # But we want to use the BAM-attended features instead
            # So we'll concatenate the attended features
            combined = torch.cat([texture_attended_flat, semantic_attended_flat], dim=1)
        else:
            # Extract features from both branches without BAM
            texture_features = self.texture_branch(x)
            semantic_features = self.semantic_branch(x)
            
            # Concatenate branch features
            combined = torch.cat([texture_features, semantic_features], dim=1)
        
        # Fuse features and project to embedding dimension
        embedding = self.fusion(combined)
        
        # L2 normalization for metric learning
        embedding = F.normalize(embedding, p=2, dim=1)
        
        if return_branch_features and return_attention_maps:
            return embedding, texture_features, semantic_features, texture_attention_map, semantic_attention_map
        elif return_branch_features:
            return embedding, texture_features, semantic_features
        elif return_attention_maps:
            return embedding, texture_attention_map, semantic_attention_map
        else:
            return embedding




if __name__ == "__main__":
    # Test the dual-branch extractor
    print("Testing Dual-Branch Feature Extractor (Updated for Methodology)")
    print("=" * 80)
    
    # Create model
    model = DualBranchFeatureExtractor(
        input_channels=3,
        texture_dim=256,
        semantic_dim=256,
        embedding_dim=128,  # 128-dim as per methodology
        use_bam=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    texture_params = sum(p.numel() for p in model.texture_branch.parameters())
    semantic_params = sum(p.numel() for p in model.semantic_branch.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"  Texture branch: {texture_params:,}")
    print(f"  Semantic branch: {semantic_params:,}")
    print(f"  Fusion + BAM: {total_params - texture_params - semantic_params:,}")
    print()
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    # Basic forward pass
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output L2 norm: {torch.norm(output, p=2, dim=1).mean().item():.4f}")
    print()
    
    # Forward pass with branch features
    embedding, texture, semantic = model(test_input, return_branch_features=True)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Texture features shape: {texture.shape}")
    print(f"Semantic features shape: {semantic.shape}")
    print()
    
    print("=" * 80)
    print("✓ Dual-branch extractor test passed!")
    print(f"✓ Final embedding dimension: {embedding.shape[1]} (expected: 128)")

