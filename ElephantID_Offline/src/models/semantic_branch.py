"""
Semantic Shape Branch - Global Geometric Structure Extraction

Captures:
- Body bulk (Makhnas)
- Head dome shape (Calves)
- Ear curvature
- Overall proportions

Characteristics:
- Deep architecture
- Low spatial resolution
- Large receptive field

Dominant for: Calves/Juveniles, Makhnas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticBranch(nn.Module):
    """
    Deep CNN branch for global shape and semantic features.
    
    Architecture:
    - 5 convolutional layers (deeper than texture branch)
    - Aggressive downsampling for global context
    - Large receptive field (captures overall shape)
    - Outputs 256-dimensional feature vector
    """
    
    def __init__(self, input_channels=3, feature_dim=256):
        """
        Initialize semantic branch.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            feature_dim: Dimension of output feature vector
        """
        super(SemanticBranch, self).__init__()
        
        # Deep convolutional layers with larger kernels
        # Layer 1: Initial feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 2: Build semantic understanding
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 3: Higher-level patterns
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Layer 4: Abstract shape features
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Layer 5: Global semantic features
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global average pooling (captures overall shape)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer to feature vector
        self.fc = nn.Linear(512, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x, return_spatial_features=False):
        """
        Forward pass through semantic branch.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            return_spatial_features: If True, return spatial features before pooling
            
        Returns:
            If return_spatial_features=False:
                Feature vector [batch_size, feature_dim]
            If return_spatial_features=True:
                (Feature vector, spatial_features) where spatial_features is [batch, 512, H, W]
        """
        # Conv block 1 - aggressive downsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Conv block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # Store spatial features before global pooling (for BAM)
        spatial_features = x  # [batch, 512, H, W]
        
        # Global average pooling (1x1 spatial output)
        x = self.global_pool(x)
        
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
        # Larger kernels and aggressive downsampling
        # Approximate RF: ~200+ pixels (covers most of the elephant)
        return 211


if __name__ == "__main__":
    # Test the semantic branch
    print("Testing Semantic Branch...")
    
    # Create model
    model = SemanticBranch(input_channels=3, feature_dim=256)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Receptive field: {model.get_receptive_field()} pixels")
    print(f"Output L2 norm: {torch.norm(output, p=2, dim=1).mean().item():.4f}")
    print("✓ Semantic branch test passed!")
