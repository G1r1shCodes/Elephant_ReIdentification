"""
Verify TTA Variants on MAKHNA ONLY.
Variants:
1. Original Only (Control)
2. Original + Horizontal Flip
3. Full TTA (7 views)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.dual_branch_extractor import DualBranchFeatureExtractor

# Config
DATA_ROOT = Path("d:/Elephant_ReIdentification/data/restructured")
MODEL_PATH = Path("d:/Elephant_ReIdentification/makhna_model.pth")
IMAGE_SIZE = (256, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TTATransform:
    def __init__(self):
        self.base = transforms.Resize((256, 128))
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        
    def __call__(self, img):
        # 1. Original
        img_resized = self.base(img)
        v1 = self.norm(self.to_tensor(img_resized))
        # 2. Horizontal Flip
        v2 = self.norm(self.to_tensor(transforms.functional.hflip(img_resized)))
        # 3. Five Crop logic
        img_large = transforms.Resize((288, 144))(img)
        crops = transforms.FiveCrop((256, 128))(img_large)
        
        views = [v1, v2]
        for crop in crops:
            views.append(self.norm(self.to_tensor(crop)))
        return torch.stack(views) # [7, 3, H, W]

def evaluate_embeddings(embeddings, labels, name):
    similarity_matrix = np.dot(embeddings, embeddings.T)
    n = len(labels)
    ranks = []
    for i in range(n):
        scores = similarity_matrix[i].copy()
        scores[i] = -np.inf
        gt_mask = (labels == labels[i])
        gt_mask[i] = False
        if gt_mask.sum() == 0: continue
        sorted_indices = np.argsort(scores)[::-1]
        correct_ranks = np.where(gt_mask[sorted_indices])[0]
        if len(correct_ranks) > 0:
            ranks.append(correct_ranks[0] + 1)
            
    if len(ranks) > 0:
        rank1 = np.mean(np.array(ranks) == 1) * 100
        rank5 = np.mean(np.array(ranks) <= 5) * 100
        print(f"   {name:<20} | Rank-1: {rank1:.2f}% | Rank-5: {rank5:.2f}%")
        return rank1
    return 0.0

def main():
    print("="*70)
    print("TTA VARIANTS RECHECK (MAKHNA ONLY)")
    print("="*70)
    
    # 1. Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    emb_dim = checkpoint.get('embedding_dim', 128)
    num_classes = checkpoint.get('num_classes', 19)
    model = DualBranchFeatureExtractor(embedding_dim=emb_dim, num_classes=num_classes, use_bam=True)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(DEVICE)
    model.eval()
    
    # 2. Dataset Loop (FILTERED)
    image_paths = sorted(list(DATA_ROOT.rglob("*.jpg")) + list(DATA_ROOT.rglob("*.png")))
    valid_paths = []
    labels = []
    label_to_idx = {}
    
    for p in image_paths:
        identity = p.parent.name
        if not identity.startswith("Makhna"): continue
        if identity not in label_to_idx:
            label_to_idx[identity] = len(label_to_idx)
        labels.append(label_to_idx[identity])
        valid_paths.append(p)
        
    print(f"Dataset: {len(valid_paths)} images")
    
    # 3. Extract All Views
    tta_transform = TTATransform()
    
    # Store individual view embeddings
    # We need to average them differently for each variant
    # Shape: [N, 7, Dim]
    all_view_embeddings = []
    
    with torch.no_grad():
        for path in tqdm(valid_paths):
            try:
                img = Image.open(path).convert("RGB")
                batch = tta_transform(img).to(DEVICE) # [7, 3, H, W]
                
                output = model(batch)
                emb_batch = output[0] if isinstance(output, tuple) else output # [7, 128]
                
                # Normalize each view independently first? Usually we average then normalize.
                # But here we want to combine them flexibly.
                # Let's keep them raw, then average, then normalize.
                
                all_view_embeddings.append(emb_batch.cpu().numpy())
            except Exception as e:
                print(f"Error {path}: {e}")
                
    all_view_embeddings = np.stack(all_view_embeddings) # [N, 7, 128]
    labels = np.array(labels)
    
    print("\n📊 RESULTS:")
    print("-" * 60)
    
    # Variant 1: Original Only (View 0)
    emb_v1 = all_view_embeddings[:, 0, :] # [N, 128]
    # Normalize
    norms = np.linalg.norm(emb_v1, axis=1, keepdims=True)
    emb_v1 = emb_v1 / (norms + 1e-8)
    evaluate_embeddings(emb_v1, labels, "Original Only")
    
    # Variant 2: Original + Flip (View 0 + 1)
    emb_v2 = np.mean(all_view_embeddings[:, 0:2, :], axis=1) # [N, 128]
    norms = np.linalg.norm(emb_v2, axis=1, keepdims=True)
    emb_v2 = emb_v2 / (norms + 1e-8)
    evaluate_embeddings(emb_v2, labels, "Orig + Flip")
    
    # Variant 3: Full TTA (All 7 views)
    emb_v7 = np.mean(all_view_embeddings, axis=1) # [N, 128]
    norms = np.linalg.norm(emb_v7, axis=1, keepdims=True)
    emb_v7 = emb_v7 / (norms + 1e-8)
    evaluate_embeddings(emb_v7, labels, "Full TTA (7 views)")

if __name__ == "__main__":
    main()
