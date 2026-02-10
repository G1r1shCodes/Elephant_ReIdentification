"""
Biological Attention Map (BAM)

Spatial attention mechanism that learns WHERE to look based on
biologically meaningful regions (ears, temporal gland, head shape).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiologicalAttentionMap(nn.Module):
    """
    Spatial attention mechanism for elephant biometric features.
    
    Learns to focus on biologically relevant regions:
    - Makhnas: Temporal gland, cheek region, body bulk
    - Adult Females: Ear pinna, ear tears, facial texture
    - Calves/Juveniles: Head shape, ear curvature, proportions
    
    The attention is learned implicitly through metric learning,
    without explicit sex/age labels.
    """
    
    def __init__(self, in_channels, reduction=16):
        """
        Initialize Biological Attention Map.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for efficiency
        """
        super(BiologicalAttentionMap, self).__init__()
        
        # Channel attention (what features)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention (where to look)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Apply biological attention.
        
        Args:
            x: Input feature map [batch, channels, height, width]
            
        Returns:
            attended_features: Attention-weighted features
            attention_map: Spatial attention map for visualization
        """
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial attention
        attention_map = self.spatial_attention(x_channel)
        attended_features = x_channel * attention_map
        
        return attended_features, attention_map


if __name__ == "__main__":
    # Test BAM
    print("Testing Biological Attention Map...")
    
    bam = BiologicalAttentionMap(in_channels=256, reduction=16)
    test_input = torch.randn(2, 256, 14, 14)
    
    attended, attention_map = bam(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Attended features shape: {attended.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Attention map range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
    print("✓ BAM test passed!")
