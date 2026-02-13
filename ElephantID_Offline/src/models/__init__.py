"""
Elephant Re-Identification Models

Dual-branch feature extraction architecture for biologically-aware
elephant re-identification.
"""

from .dual_branch_extractor import DualBranchFeatureExtractor
from .texture_branch import TextureBranch
from .semantic_branch import SemanticBranch

__all__ = [
    'DualBranchFeatureExtractor',
    'TextureBranch',
    'SemanticBranch'
]
