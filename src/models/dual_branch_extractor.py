"""
Standardized Dual-Branch Feature Extractor
Based on Kaggle Notebook Architecture for compatibility with makhna_model.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BAM(nn.Module):
    """Biological Attention Map (BAM) Module"""
    def __init__(self, in_channels, reduction_ratio=16, dilated=True):
        super(BAM, self).__init__()
        self.in_channels = in_channels
        
        # Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Spatial Attention
        if dilated:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, 3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, 3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, 1, 1, bias=False),
                nn.BatchNorm2d(1)
            )
        else:
             self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, 1, 1, bias=False),
                nn.BatchNorm2d(1)
            )

    def forward(self, x):
        # Channel attention
        att_c = self.channel_att(x)
        
        # Spatial attention
        att_s = self.spatial_att(x)
        
        # Fuse
        att = F.sigmoid(att_c + att_s)
        
        return x * att, att

class DualBranchFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=None, use_bam=False):
        super().__init__()
        self.use_bam = use_bam
        self.num_classes = num_classes
        
        # Handle torchvision version
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2
            base_model = models.resnet50(weights=weights)
        except (ImportError, AttributeError):
            base_model = models.resnet50(pretrained=True)
        
        # Split into texture (shallow) and semantic (deep)
        self.layer0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Texture Branch Components
        self.texture_reducer = nn.Conv2d(512, 1024, kernel_size=1)
        if self.use_bam:
             self.texture_bam = BAM(1024)
        
        # Semantic Branch Components
        if self.use_bam:
             self.semantic_bam = BAM(2048)
        
        # Embedding head
        self.fc = nn.Linear(2048 + 1024, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        
        # Classification Head (CRITICAL for stability)
        if self.num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
            
    def texture_branch(self, x, return_spatial=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # Fix: layer2 output is 512 channels. texture_reducer input is 512.
        feat = self.texture_reducer(x)
        
        if self.use_bam:
             feat_att, _ = self.texture_bam(feat)
             if return_spatial: return feat_att, feat 
             return feat_att
        
        if return_spatial: return feat, feat
        return feat

    def semantic_branch(self, x, return_spatial=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_bam:
             feat_att, _ = self.semantic_bam(x)
             if return_spatial: return feat_att, x
             return feat_att
        
        if return_spatial: return x, x
        return x

    def forward(self, x):
        # Texture Branch
        tex_feat_spatial = self.texture_branch(x)
        tex_feat = self.global_pool(tex_feat_spatial).flatten(1)
        
        # Semantic Branch
        sem_feat_spatial = self.semantic_branch(x)
        sem_feat = self.global_pool(sem_feat_spatial).flatten(1)
        
        # Fuse
        combined = torch.cat([tex_feat, sem_feat], dim=1)
        embedding_raw = self.fc(combined)
        embedding_raw = self.bn(embedding_raw)
        
        # Normalize for Triplet (metric learning)
        embedding = F.normalize(embedding_raw, p=2, dim=1)
        
        if self.training and self.num_classes:
            logits = self.classifier(embedding_raw)
            return embedding, logits
            
        return embedding
