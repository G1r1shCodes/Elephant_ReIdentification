"""
Standalone script to test Makhna model classification
Loads:
- Model: D:/Elephant_ReIdentification/makhna_model.pth
- Gallery: D:/Elephant_ReIdentification/gallery_embeddings (1).pt
- Data: data/processed_megadetector/Makhna

Outputs:
- Accuracy, Matrix, Per-class stats
"""
import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))
try:
    from src.models.dual_branch_extractor import DualBranchFeatureExtractor
except ImportError:
    # If run from root
    sys.path.append(str(Path("src/models").parent.parent))
    from src.models.dual_branch_extractor import DualBranchFeatureExtractor

print("="*70)
print("VERIFY SYSTEM ACCURACY")
print("="*70)

# Paths
MODEL_PATH = Path("D:/Elephant_ReIdentification/makhna_model.pth")
GALLERY_PATH = Path("D:/Elephant_ReIdentification/gallery_embeddings.pt")
DATA_PATH = Path("data/processed_megadetector/Makhna")

# Check
for p in [MODEL_PATH, GALLERY_PATH, DATA_PATH]:
    if not p.exists():
        print(f"❌ Missing: {p}")
        sys.exit(1)

print(f"✅ Found all files")

# Load Model
print("\n📦 Loading Model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
embedding_dim = checkpoint.get('embedding_dim', 128)

model = DualBranchFeatureExtractor(embedding_dim=embedding_dim, use_bam=True)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
print(f"✅ Model loaded (dim={embedding_dim})")

# Load Gallery
print("\n📊 Loading Gallery...")
gallery_data = torch.load(GALLERY_PATH)
gallery_embeddings = gallery_data['embeddings'].numpy()
gallery_labels = gallery_data['labels'].numpy()
idx_to_identity = gallery_data['idx_to_identity']
print(f"✅ Gallery loaded: {len(gallery_embeddings)} embeddings, {len(idx_to_identity)} identities")

# Transform (ImageNet norm)
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and Test Data
print("\n📂 Processing Makhna Dataset...")
true_labels = []
pred_labels = []

# Get all images
image_files = []
for elephant_dir in sorted(DATA_PATH.iterdir()):
    if not elephant_dir.is_dir(): continue
    name = elephant_dir.name
    for img_path in elephant_dir.glob("*.jpg"):
        image_files.append((img_path, name))

print(f"   Found {len(image_files)} images")

# Run Inference
correct = 0
total = 0

with torch.no_grad():
    for img_path, true_label in tqdm(image_files, desc="Classifying"):
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0)
            
            # Extract
            emb = model(img_t).squeeze().numpy()
            
            # Match
            sims = np.dot(gallery_embeddings, emb)
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            
            pred_idx = gallery_labels[best_idx]
            if isinstance(pred_idx, np.ndarray): pred_idx = pred_idx.item()
            
            pred_label = idx_to_identity[pred_idx]
            
            # Debug first 10 mismatches
            if true_label != pred_label and total < 20: 
                 print(f"\\nMISMATCH {Path(img_path).name}:")
                 print(f"  True: {true_label}")
                 print(f"  Pred: {pred_label} (idx {pred_idx})")
                 print(f"  Score: {best_score:.4f}")
                 print(f"  Best Match Index: {best_idx}")
                 
                 # Check if the image itself is in gallery?
                 # It should be.
            
            # Record
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            
            if true_label == pred_label:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"Error {img_path}: {e}")

# Results
accuracy = (correct / total) * 100 if total > 0 else 0
print("\n" + "="*70)
print(f"RESULTS (Accuracy: {accuracy:.2f}%)")
print("="*70)

print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, zero_division=0))

print("\n❌ Misclassifications (First 10):")
count = 0
for t, p, (path, _) in zip(true_labels, pred_labels, image_files):
    if t != p:
        print(f"   {path.name}: True={t}, Pred={p}")
        count += 1
        if count >= 10: break

print("\n" + "="*70)
