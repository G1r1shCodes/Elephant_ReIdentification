# Kaggle Training Setup

## 📁 Files

- **elephant_reid_training.ipynb** - Main training notebook (all 7 fixes applied)
- **dataset-metadata.json** - Dataset metadata for Kaggle
- **elephant-reid-dataset.zip** - Training dataset archive
- **dataset/** - Extracted dataset directory

## 🚀 Quick Start

1. Upload `elephant_reid_training.ipynb` to Kaggle
2. Enable GPU (P100 or T4) and Internet
3. Set `NUM_EPOCHS = 5` for initial test
4. Run all cells

## ✅ Success Criteria (5 epochs)

Monitor "Embedding Health" output:
- Embedding std > 0.01
- Intra-class > Inter-class similarity
- No collapse warnings

If successful → Train for 80-100 epochs!

## 📝 Applied Fixes

All 7 critical fixes are pre-applied:
1. Learning rate reduced to 0.0001
2. M-per-class batch sampler (4 images/elephant)
3. DataLoader with batch_sampler (not shuffle)
4. Simplified optimizer
5. Collapse monitoring after each epoch
6. TripletLoss return value bug fixed
7. Attention regularization disabled
