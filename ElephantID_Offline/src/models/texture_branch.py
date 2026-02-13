"""
Texture Branch - Fine-grained Local Detail Extraction

Captures:
- Ear depigmentation (pink spots)
- Ear tears and notches
- Skin and trunk texture

Characteristics:
- Shallow architecture
- High spatial resolution
- Small receptive field

Dominant for: Adult females, some adult males
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureBranch(nn.Module):
    """
    Shallow CNN branch for fine-grained texture features.
    
    Architecture:
    - 3 convolutional layers
    - Maintains high spatial resolution
    - Small receptive field (focuses on local details)
    - Outputs 256-dimensional feature vector
    """
    
    def __init__(self, input_channels=3, feature_dim=256):
        """
        Initialize texture branch.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            feature_dim: Dimension of output feature vector
        """
        super(TextureBranch, self).__init__()
        
        # Shallow convolutional layers with small kernels
        # Layer 1: Focus on very fine details
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 2: Combine local patterns
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 3: Higher-level texture patterns
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fully connected layer to feature vector
        self.fc = nn.Linear(256 * 8 * 8, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, return_spatial_features=False):
        """
        Forward pass through texture branch.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            return_spatial_features: If True, return spatial features before pooling
            
        Returns:
            If return_spatial_features=False:
                Feature vector [batch_size, feature_dim]
            If return_spatial_features=True:
                (Feature vector, spatial_features) where spatial_features is [batch, 256, H, W]
        """
        # Conv block 1 - preserve spatial resolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Gentle downsampling
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Store spatial features before pooling (for BAM)
        spatial_features = x  # [batch, 256, H, W]
        
        # Adaptive pooling to fixed spatial size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer with dropout
        x = self.dropout(x)
        x = self.fc(x)
        
        # L2 normalization for metric learning
        x = F.normalize(x, p=2, dim=1)
        
        if return_spatial_features:
            return x, spatial_features
        return x
    
    def get_receptive_field(self):
        """
        Calculate theoretical receptive field size.
        
        Returns:
            Receptive field size in pixels
        """
        # 3x3 kernels with stride 1, pooling 2x2
        # RF = 1 + (3-1) + (3-1)*2 + (3-1)*4 = 1 + 2 + 4 + 8 = 15 pixels
        return 15


if __name__ == "__main__":
    # Test the texture branch
    print("Testing Texture Branch...")
    
    # Create model
    model = TextureBranch(input_channels=3, feature_dim=256)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Receptive field: {model.get_receptive_field()} pixels")
    print(f"Output L2 norm: {torch.norm(output, p=2, dim=1).mean().item():.4f}")
    print("✓ Texture branch test passed!")
