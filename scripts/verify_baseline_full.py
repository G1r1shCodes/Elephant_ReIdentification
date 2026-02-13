"""
Verify Baseline accuracy on MAKHNA ONLY subset.
Single view.
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

class StandardTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __call__(self, img):
        return self.transform(img).unsqueeze(0)

def main():
    print("="*70)
    print("BASELINE CHECK: MAKHNA ONLY (Single View)")
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
    
    # 2. Dataset (Makhna Only)
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
        
    print(f"Dataset: {len(valid_paths)} images, {len(label_to_idx)} identities")
    
    # 3. Extract
    transform = StandardTransform()
    embeddings = []
    
    with torch.no_grad():
        for path in tqdm(valid_paths):
            try:
                img = Image.open(path).convert("RGB")
                batch = transform(img).to(DEVICE)
                output = model(batch)
                emb = output[0] if isinstance(output, tuple) else output
                embeddings.append(emb.cpu().numpy())
            except Exception as e:
                print(f"Error {path}: {e}")
                
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # 4. Metrics
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
        print(f"\n📊 BASELINE RESULTS (MAKHNA ONLY):")
        print(f"   Rank-1: {rank1:.2f}%")
        print(f"   Rank-5: {rank5:.2f}%")
    else:
        print("No valid rankings.")

if __name__ == "__main__":
    main()
