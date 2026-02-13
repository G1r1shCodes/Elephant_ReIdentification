"""
Comprehensive evaluation of Makhna model on Makhna dataset
Shows predicted vs ground truth, confusion matrix, per-class accuracy
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.models.dual_branch_extractor import DualBranchFeatureExtractor

print("="*70)
print("MAKHNA MODEL CLASSIFICATION TEST")
print("="*70)

# Paths
MODEL_PATH = Path("D:/Elephant_ReIdentification/makhna_model.pth")
GALLERY_PATH = Path("D:/Elephant_ReIdentification/gallery_embeddings.pt")
DATA_PATH = Path("data/processed_megadetector/Makhna")

# Check paths
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)
if not GALLERY_PATH.exists():
    print(f"❌ Gallery not found: {GALLERY_PATH}")
    exit(1)
if not DATA_PATH.exists():
    print(f"❌ Makhna data not found: {DATA_PATH}")
    exit(1)

print(f"\n✅ All paths found")

# Load model
print("\n📦 Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
embedding_dim = checkpoint.get('embedding_dim', 128)

model = DualBranchFeatureExtractor(
    embedding_dim=embedding_dim,
    use_bam=True
)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
print(f"✅ Model loaded (embedding_dim={embedding_dim})")

# Load gallery
print("\n📊 Loading gallery...")
gallery_data = torch.load(GALLERY_PATH)
gallery_embeddings = gallery_data['embeddings'].numpy()
gallery_labels = gallery_data['labels'].numpy()
idx_to_identity = gallery_data['idx_to_identity']
print(f"✅ Gallery loaded: {len(np.unique(gallery_labels))} identities, {len(gallery_embeddings)} embeddings")

# Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Makhna dataset
print("\n📂 Loading Makhna dataset...")
image_paths = []
ground_truth_labels = []

for elephant_dir in sorted(DATA_PATH.iterdir()):
    if not elephant_dir.is_dir():
        continue
    
    elephant_name = elephant_dir.name
    
    for img_path in elephant_dir.glob("*.jpg"):
        image_paths.append(img_path)
        ground_truth_labels.append(elephant_name)

print(f"✅ Loaded {len(image_paths)} images from {len(set(ground_truth_labels))} Makhna elephants")

# Extract embeddings for all test images
print("\n🔍 Extracting embeddings...")
test_embeddings = []

with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            embedding = model(img_tensor)
            test_embeddings.append(embedding.squeeze().cpu().numpy())
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            test_embeddings.append(np.zeros(embedding_dim))

test_embeddings = np.array(test_embeddings)
print(f"✅ Extracted {len(test_embeddings)} embeddings")

# Classify using gallery (nearest neighbor)
print("\n🎯 Classifying...")
predictions = []

for test_emb in tqdm(test_embeddings, desc="Matching"):
    # Compute similarity with all gallery embeddings
    similarities = np.dot(gallery_embeddings, test_emb)
    
    # Find best match
    best_idx = np.argmax(similarities)
    predicted_label_idx = gallery_labels[best_idx]
    predicted_identity = idx_to_identity[predicted_label_idx]
    
    predictions.append(predicted_identity)

predictions = np.array(predictions)
ground_truth_labels = np.array(ground_truth_labels)

# Compute metrics
print("\n" + "="*70)
print("CLASSIFICATION RESULTS")
print("="*70)

# Overall accuracy
overall_acc = accuracy_score(ground_truth_labels, predictions) * 100
print(f"\n📊 Overall Accuracy: {overall_acc:.2f}%")

# Per-class accuracy
print("\n📋 Per-Class Accuracy:")
print("-" * 70)
print(f"{'Elephant ID':<20} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
print("-" * 70)

unique_labels = sorted(set(ground_truth_labels))
class_accuracies = {}

for label in unique_labels:
    mask = ground_truth_labels == label
    total = np.sum(mask)
    correct = np.sum(predictions[mask] == label)
    acc = (correct / total * 100) if total > 0 else 0
    class_accuracies[label] = acc
    print(f"{label:<20} {total:>8} {correct:>8} {acc:>9.1f}%")

print("-" * 70)

# Show misclassifications
print("\n❌ Misclassifications:")
print("-" * 70)
misclass_count = 0

for i, (gt, pred) in enumerate(zip(ground_truth_labels, predictions)):
    if gt != pred:
        print(f"   {image_paths[i].name:<30} | GT: {gt:<15} | Pred: {pred:<15}")
        misclass_count += 1
        if misclass_count >= 20:  # Limit output
            remaining = np.sum(ground_truth_labels != predictions) - 20
            if remaining > 0:
                print(f"   ... and {remaining} more misclassifications")
            break

if misclass_count == 0:
    print("   None! Perfect classification! 🎉")

# Confusion Matrix
print("\n📊 Generating confusion matrix...")
cm = confusion_matrix(ground_truth_labels, predictions, labels=unique_labels)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix\\nOverall Accuracy: {overall_acc:.2f}%', fontsize=14, fontweight='bold')
plt.ylabel('Ground Truth', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('makhna_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved to makhna_confusion_matrix.png")

# Classification Report
print("\n📄 Detailed Classification Report:")
print("-" * 70)
report = classification_report(ground_truth_labels, predictions, labels=unique_labels, zero_division=0)
print(report)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nDataset: {len(image_paths)} images from {len(unique_labels)} Makhna elephants")
print(f"Overall Accuracy: {overall_acc:.2f}%")
print(f"Correctly Classified: {np.sum(predictions == ground_truth_labels)}/{len(predictions)}")
print(f"Misclassified: {np.sum(predictions != ground_truth_labels)}/{len(predictions)}")
print(f"\nBest Performing: {max(class_accuracies, key=class_accuracies.get)} ({class_accuracies[max(class_accuracies, key=class_accuracies.get)]:.1f}%)")
print(f"Worst Performing: {min(class_accuracies, key=class_accuracies.get)} ({class_accuracies[min(class_accuracies, key=class_accuracies.get)]:.1f}%)")
print("\n" + "="*70)
