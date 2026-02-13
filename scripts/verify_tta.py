"""
Script to verify model accuracy using Test-Time Augmentation (TTA).
TARGET: MAKHNA ONLY (19 Classes)
Strategy:
1. Load baseline model (makhna_model.pth).
2. Filter for directories starting with "Makhna".
3. For each image:
   - Generate 7 augmented views (Original + Flip + 5 Crops).
   - Average embeddings.
4. LOO evaluation.
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
        return torch.stack(views)

def main():
    print("="*70)
    print("EXPERIMENT 4: TTA (MAKHNA ONLY)")
    print("="*70)
    
    # 1. Load Model
    print(f"Loading model: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    emb_dim = checkpoint.get('embedding_dim', 128)
    num_classes = checkpoint.get('num_classes', 19)
    
    model = DualBranchFeatureExtractor(embedding_dim=emb_dim, num_classes=num_classes, use_bam=True)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Flexible Load
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model.to(DEVICE)
    model.eval()
    
    # 2. Dataset Loop (FILTERED)
    print(f"Scanning dataset: {DATA_ROOT} (Makhna Only)")
    image_paths = sorted(list(DATA_ROOT.rglob("*.jpg")) + list(DATA_ROOT.rglob("*.png")))
    
    valid_paths = []
    labels = []
    label_to_idx = {}
    
    count_skipped = 0
    for p in image_paths:
        identity = p.parent.name
        # FILTER: Only "Makhna"
        if not identity.startswith("Makhna"):
            count_skipped += 1
            continue
            
        if identity not in label_to_idx:
            label_to_idx[identity] = len(label_to_idx)
        labels.append(label_to_idx[identity])
        valid_paths.append(p)
        
    print(f"Selected {len(valid_paths)} images from {len(label_to_idx)} Makhna identities")
    print(f"Skipped {count_skipped} non-Makhna images")
    
    # 3. Extract Embeddings with TTA
    print("Extracting embeddings with TTA...")
    tta_transform = TTATransform()
    embeddings = []
    
    with torch.no_grad():
        for path in tqdm(valid_paths):
            try:
                img = Image.open(path).convert("RGB")
                batch = tta_transform(img).to(DEVICE)
                output = model(batch)
                emb_batch = output[0] if isinstance(output, tuple) else output
                emb_avg = torch.mean(emb_batch, dim=0, keepdim=True)
                emb_avg = F.normalize(emb_avg, p=2, dim=1)
                embeddings.append(emb_avg.cpu().numpy())
            except Exception as e:
                print(f"Error {path}: {e}")
            
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # 4. Evaluate LOO
    print("\n🔍 Computing metrics (Leave-One-Out)...")
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
        print(f"\n📊 TTA RESULTS (MAKHNA ONLY):")
        print(f"   Rank-1: {rank1:.2f}%")
        print(f"   Rank-5: {rank5:.2f}%")
    else:
        print("No valid rankings.")

if __name__ == "__main__":
    main()
