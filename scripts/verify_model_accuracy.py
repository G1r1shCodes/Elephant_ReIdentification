"""
Verify model accuracy before deploying to app.py
Load model and gallery embeddings, run quick evaluation
"""
import torch
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import roc_curve, auc

print("="*70)
print("VERIFYING MODEL ACCURACY")
print("="*70)


# Paths
model_path = Path("D:/Elephant_ReIdentification/makhna_model (2).pth")
gallery_path = Path("D:/Elephant_ReIdentification/gallery_embeddings (2).pt")

# Check files exist
print("\n📁 Checking files...")
print(f"   Model: {model_path.exists()} - {model_path}")
print(f"   Gallery: {gallery_path.exists()} - {gallery_path}")

if not model_path.exists() or not gallery_path.exists():
    print("\n❌ Files not found! Make sure to download from Kaggle:")
    print("   - makhna_model.pth")
    print("   - gallery_embeddings.pt")
    exit(1)

# Ensure we can load the model structure (even if not used for inference here, good for verification)
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.models.dual_branch_extractor import DualBranchFeatureExtractor
    print("   ✓ DualBranchFeatureExtractor imported successfully")
except ImportError as e:
    print(f"   ⚠️ Could not import model class: {e}")

# Load gallery
print("\n📊 Loading gallery embeddings...")
try:
    gallery_data = torch.load(gallery_path)
    embeddings = gallery_data['embeddings'].numpy()
    labels = gallery_data['labels'].numpy()
    idx_to_identity = gallery_data['idx_to_identity']
    
    print(f"   ✓ Gallery loaded")
    print(f"   - {len(embeddings)} embeddings")
    print(f"   - {len(np.unique(labels))} unique identities")
    print(f"   - Embedding dim: {embeddings.shape[1]}")
except Exception as e:
    print(f"❌ Error loading gallery: {e}")
    exit(1)


# Verify metrics
print("\n🔍 Computing metrics...")

# Similarity matrix
similarity_matrix = np.dot(embeddings, embeddings.T)

# Rank-1 accuracy
def evaluate_ranking(similarity, labels):
    n = len(labels)
    ranks = []
    
    for i in range(n):
        scores = similarity[i].copy()
        scores[i] = -np.inf  # Exclude self
        
        gt_mask = (labels == labels[i])
        gt_mask[i] = False
        
        sorted_indices = np.argsort(scores)[::-1]
        correct_ranks = np.where(gt_mask[sorted_indices])[0]
        
        if len(correct_ranks) > 0:
            ranks.append(correct_ranks[0] + 1)
    
    rank1 = np.mean(np.array(ranks) == 1) * 100
    rank5 = np.mean(np.array(ranks) <= 5) * 100
    
    return rank1, rank5

rank1, rank5 = evaluate_ranking(similarity_matrix, labels)

# Intra/Inter similarity
intra_sim = []
inter_sim = []

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        sim = similarity_matrix[i, j]
        if labels[i] == labels[j]:
            intra_sim.append(sim)
        else:
            inter_sim.append(sim)

margin = np.mean(intra_sim) - np.mean(inter_sim)

# ROC AUC
y_true = []
y_scores = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        y_true.append(1 if labels[i] == labels[j] else 0)
        y_scores.append(similarity_matrix[i, j])

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Display results
print("\n" + "="*70)
print("VERIFICATION RESULTS")
print("="*70)
print(f"\n📊 Ranking Performance:")
print(f"   Rank-1:  {rank1:.2f}%")
print(f"   Rank-5:  {rank5:.2f}%")

print(f"\n📈 Verification:")
print(f"   ROC AUC: {roc_auc:.3f}")

print(f"\n📐 Embedding Geometry:")
print(f"   Intra:   {np.mean(intra_sim):.4f}")
print(f"   Inter:   {np.mean(inter_sim):.4f}")
print(f"   Margin:  {margin:.4f}")

# Check against claimed metrics
print("\n" + "="*70)
print("COMPARISON WITH CLAIMED METRICS")
print("="*70)

claimed = {
    'rank1': 90.91,
    'rank5': 95.45,
    'roc_auc': 0.975,
    'intra': 0.6843,
    'inter': 0.1954,
    'margin': 0.4889
}

actual = {
    'rank1': rank1,
    'rank5': rank5,
    'roc_auc': roc_auc,
    'intra': np.mean(intra_sim),
    'inter': np.mean(inter_sim),
    'margin': margin
}

print("\n| Metric | Claimed | Actual | Match |")
print("|--------|---------|--------|-------|")
for key in ['rank1', 'rank5', 'roc_auc', 'intra', 'inter', 'margin']:
    c = claimed[key]
    a = actual[key]
    diff = abs(c - a)
    match = "✓" if diff < 0.02 else "✗"
    print(f"| {key:8s} | {c:7.2f} | {a:7.2f} | {match:5s} |")

print("\n" + "="*70)

# Final verdict
if abs(actual['rank1'] - claimed['rank1']) < 2.0 and abs(actual['roc_auc'] - claimed['roc_auc']) < 0.01:
    print("✅ ACCURACY VERIFIED - Model matches claimed performance!")
    print("\n🚀 Ready to deploy to app.py")
else:
    print("⚠️  Mismatch detected - Investigate differences")
    
print("="*70)
