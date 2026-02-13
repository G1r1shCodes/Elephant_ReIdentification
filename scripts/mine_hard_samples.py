"""
Mine hard samples from the Baseline model.
Path: scripts/mine_hard_samples.py

This script runs a Leave-One-Out evaluation and logs strictly which identities
are failing most often. It outputs a JSON file with error counts per identity.
"""
import torch
import numpy as np
import sys
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.models.dual_branch_extractor import DualBranchFeatureExtractor

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    print("="*60)
    print("🐘 HARD SAMPLE MINER")
    print("="*60)

    # 1. Load Model
    model_path = Path("makhna_model.pth")
    if not model_path.exists():
        print("❌ Baseline model not found!")
        return

    print("Loading Baseline Model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    embedding_dim = checkpoint.get('embedding_dim', 128)
    num_classes = checkpoint.get('num_classes', 19) # Default/Guess
    
    model = DualBranchFeatureExtractor(embedding_dim=embedding_dim, num_classes=num_classes, use_bam=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 2. Load Gallery (to get file paths and labels)
    # We need the original images to do a fresh pass to be sure, 
    # OR we can just use the embeddings if we have the identity map.
    # Let's use the embeddings since they represent the "state" of the model.
    gallery_path = Path("gallery_embeddings.pt")
    if not gallery_path.exists():
        print("❌ Gallery not found. Please run scripts/evaluate_kaggle_model.py first to generate it.")
        return

    print("Loading Embeddings...")
    data = torch.load(gallery_path)
    embeddings = data['embeddings'] # Tensor
    labels = data['labels'].numpy()
    idx_to_identity = data['idx_to_identity']
    
    # Normalize if not already
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()
    
    print(f"Loaded {len(embeddings)} embeddings for {len(np.unique(labels))} identities.")

    # 3. Mine Errors (LOO)
    print("\nMining Errors...")
    
    identity_stats = {} # {id: {'total': 0, 'errors': 0, 'confused_with': []}}
    
    # Initialize stats
    for idx, name in idx_to_identity.items():
        identity_stats[name] = {'total': 0, 'errors': 0, 'confused_with': {}}

    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    total_samples = len(labels)
    
    for i in tqdm(range(total_samples)):
        true_label = labels[i]
        true_name = idx_to_identity[true_label]
        
        identity_stats[true_name]['total'] += 1
        
        # Exclude self from search
        scores = similarity_matrix[i].copy()
        scores[i] = -np.inf 
        
        # Find best match
        best_idx = np.argmax(scores)
        pred_label = labels[best_idx]
        pred_name = idx_to_identity[pred_label]
        
        if pred_label != true_label:
            # ERROR FOUND
            identity_stats[true_name]['errors'] += 1
            
            # Log confusion
            if pred_name not in identity_stats[true_name]['confused_with']:
                identity_stats[true_name]['confused_with'][pred_name] = 0
            identity_stats[true_name]['confused_with'][pred_name] += 1

    # 4. Analyze & Output
    print("\n" + "="*60)
    print("🚨 HARD SAMPLES REPORT 🚨")
    print("="*60)
    print(f"{'Identity':<20} | {'Samples':<8} | {'Errors':<8} | {'Error Rate':<10}")
    print("-" * 60)
    
    hard_samples = []
    
    sorted_stats = sorted(identity_stats.items(), key=lambda x: x[1]['errors'], reverse=True)
    
    for name, stats in sorted_stats:
        total = stats['total']
        errors = stats['errors']
        if total == 0: continue
        
        error_rate = (errors / total) * 100
        
        print(f"{name:<20} | {total:<8} | {errors:<8} | {error_rate:.1f}%")
        
        if error_rate > 0:
            # Find top confusion
            top_confusion = max(stats['confused_with'].items(), key=lambda x: x[1]) if stats['confused_with'] else ("None", 0)
            # print(f"   ↳ Confused most with: {top_confusion[0]} ({top_confusion[1]} times)")
            
            hard_samples.append({
                'identity': name,
                'error_rate': error_rate,
                'weight': 1.0 + (error_rate / 100.0) * 2.0  # Simple heuristic: Max 3.0x weight for 100% error
            })

    # Save to file
    output_file = Path("hard_samples_config.json")
    with open(output_file, 'w') as f:
        json.dump(hard_samples, f, indent=4)
        
    print(f"\n✅ Stats saved to {output_file}")
    print("Use this file to weight the sampler in the next training run.")

if __name__ == "__main__":
    main()
