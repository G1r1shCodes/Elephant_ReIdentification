# Elephant Re-Identification Models

This directory contains the dual-branch feature extraction architecture for biologically-aware elephant re-identification.

---

## 📁 Files

### Core Models
- **`texture_branch.py`** - Fine-grained local detail extraction (ears, skin)
- **`semantic_branch.py`** - Global geometric structure (body shape, proportions)
- **`dual_branch_extractor.py`** - Combined architecture with attention fusion

### Training
- **`train.py`** - Training script with triplet loss and hard negative mining

### Package
- **`__init__.py`** - Package initialization

---

## 🏗️ Architecture Overview

```
Input Image (224x224x3)
         |
         ├─────────────────┬─────────────────┐
         |                 |                 |
    Texture Branch    Semantic Branch       |
    (Shallow CNN)      (Deep CNN)           |
         |                 |                 |
    256-dim feat      256-dim feat          |
         |                 |                 |
         └─────────────────┴─────────────────┘
                           |
                   Attention Fusion
                           |
                      512-dim feat
                           |
                    L2 Normalization
                           |
                       Output
```

---

## 🚀 Quick Start

### 1. Test Models

```bash
# Test texture branch
python -m src.models.texture_branch

# Test semantic branch  
python -m src.models.semantic_branch

# Test dual-branch extractor
python -m src.models.dual_branch_extractor
```

### 2. Explore Features (Jupyter Notebook)

```bash
jupyter notebook notebooks/feature_extraction_exploration.ipynb
```

### 3. Train Model

```bash
cd src/models
python train.py
```

---

## 📊 Model Specifications

| Component | Parameters | Receptive Field | Output Dim |
|-----------|------------|-----------------|------------|
| Texture Branch | 4,566,272 | 15 pixels | 256 |
| Semantic Branch | 4,183,808 | 211 pixels | 256 |
| Fusion Layers | 592,258 | - | 512 |
| **Total** | **9,342,338** | - | **512** |

---

## 🎯 Design Principles

### Texture Branch (Shallow)
**Purpose:** Capture fine-grained local details

**Targets:**
- Ear depigmentation (pink spots)
- Ear tears and notches
- Skin and trunk texture

**Dominant for:** Adult females, some adult males

### Semantic Branch (Deep)
**Purpose:** Capture global geometric structure

**Targets:**
- Body bulk (Makhnas)
- Head dome shape (Calves)
- Ear curvature
- Overall proportions

**Dominant for:** Calves/Juveniles, Makhnas

### Attention Fusion
- Learns adaptive weighting between branches
- Different weights for different elephant types
- Enables biological heterogeneity handling

---

## 🔧 Training Configuration

### Loss Function
- **Triplet Loss** with hard negative mining
- Margin: 0.3
- Mining strategy: Hard (hardest positive + hardest negative)

### Optimization
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Weight Decay:** 1e-4
- **Scheduler:** Cosine annealing

### Data Augmentation
- Random horizontal flip (p=0.5)
- Color jitter (brightness, contrast, saturation)
- Random rotation (±10°)
- ImageNet normalization

### Training Parameters
- **Batch Size:** 32
- **Epochs:** 100
- **Early Stopping:** 15 epochs patience
- **Checkpoint Frequency:** Every 5 epochs

---

## 📂 Expected Data Structure

```
data/processed/
├── Makhna/
│   ├── Individual_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Individual_2/
│   │   └── ...
│   └── ...
└── Herd/
    ├── Individual_1/
    │   └── ...
    └── ...
```

**Note:** Each subdirectory represents a unique elephant individual.

---

## 💾 Model Outputs

### Checkpoints
Saved to: `outputs/models/`

- `latest_checkpoint.pth` - Latest model state
- `best_model.pth` - Best model (lowest validation loss)

### Checkpoint Contents
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'train_losses': list,
    'val_losses': list,
    'best_val_loss': float
}
```

### Loading a Checkpoint
```python
from src.models.dual_branch_extractor import DualBranchFeatureExtractor

# Initialize model
model = DualBranchFeatureExtractor()

# Load checkpoint
checkpoint = torch.load('outputs/models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 📈 Evaluation Metrics

### Planned Metrics
- **Rank-1 Accuracy** - Percentage of queries where correct match is rank 1
- **mAP** - Mean Average Precision across all queries
- **CMC Curve** - Cumulative Matching Characteristic

### Feature Quality
- **Intra-class Distance** - Distance between same individual
- **Inter-class Distance** - Distance between different individuals
- **Separation Ratio** - Inter-class / Intra-class distance

---

## 🔬 Research Notes

### Biological Heterogeneity
The dual-branch architecture addresses the fact that different elephant groups rely on different visual cues:

1. **Adult Females:** Primarily texture-based (ear patterns)
2. **Makhnas (Adult Males):** Mix of texture and shape (body bulk)
3. **Calves/Juveniles:** Primarily shape-based (head dome, proportions)

### Attention Mechanism
The attention weights adapt per image, automatically focusing on the most relevant branch for each input.

---

## 📚 References

- **Triplet Loss:** Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
- **Hard Negative Mining:** Hermans et al., "In Defense of the Triplet Loss for Person Re-Identification" (2017)
- **Metric Learning:** Musgrave et al., "A Metric Learning Reality Check" (2020)

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `Config` class
- Use gradient accumulation
- Enable mixed precision training

### Poor Convergence
- Adjust learning rate
- Increase margin in triplet loss
- Check data augmentation strength

### Imbalanced Classes
- Use weighted sampling
- Adjust triplet mining strategy
- Ensure sufficient samples per identity

---

**Status:** ✅ Phase C Complete  
**Next:** Phase D - Training & Evaluation
