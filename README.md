# Unique Elephant Identification System

Open-set biometric elephant re-identification for Wildlife Institute of India.

## Quick Start

### Training on Kaggle (GPU Optimized) ⚡
```bash
# 1. Upload kaggle/elephant_reid_training.ipynb to Kaggle
# 2. Enable GPU P100 (recommended) or T4
# 3. Add your elephant dataset
# 4. Run all cells (~40 minutes training with P100)
```

**GPU Optimizations:**
- Batch size: 64 (optimized for P100/T4)
- Workers: 4 (persistent workers, prefetch_factor=2)
- Mixed precision training (AMP)
- Expected training time: ~40 minutes on P100 (vs 2-3 hours without optimizations)

See [`kaggle/README_KAGGLE_DEPLOYMENT.md`](kaggle/README_KAGGLE_DEPLOYMENT.md) for detailed instructions.

### Running the App 🐘
```bash
# Start the Streamlit interface
streamlit run app.py
```

### Local Setup & Evaluation
```bash
# Setup environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Evaluate trained model
python tests\evaluate_kaggle_model.py

# Run verification tests
# Run verification tests
python tests\verify_phase_c.py

# Build Standalone Executable
See [docs/BUILD_INSTRUCTIONS.md](docs/BUILD_INSTRUCTIONS.md)
```

## Project Status

1. ✅ **Phase A**: Data collection & annotation
2. ✅ **Phase B**: Biologically-aware preprocessing (MegaDetector integration)
3. ✅ **Phase C**: Feature extraction & training **COMPLETE**
   - Model optimized to 93.59% Rank-1 accuracy (Production)
   - Training time: 41 minutes (38 epochs with early stopping)
   - GPU-optimized training pipeline
4. ✅ **Phase D**: Open-set inference & enrollment (App Deployed)

## Project Structure

```
├── app.py                   # Streamlit Application
├── data/                    # Datasets (gitignored)
│   ├── raw/                 # Original annotated images
│   ├── processed/           # Preprocessed crops
│   └── processed_megadetector/  # MegaDetector preprocessed
├── src/                     # Source code
│   ├── preprocessing/       # Preprocessing scripts
│   ├── models/              # Model architecture & training
│   │   ├── dual_branch_extractor.py  # Main model
│   │   ├── biological_attention.py   # BAM module
│   │   ├── texture_branch.py
│   │   ├── semantic_branch.py
│   │   └── train.py         # Local training script
│   └── utils/               # Utilities
├── kaggle/                  # Kaggle deployment
│   ├── elephant_reid_training.ipynb  # GPU-optimized training notebook
│   ├── README_KAGGLE_DEPLOYMENT.md
│   └── dataset/             # Kaggle dataset files
├── notebooks/               # Jupyter notebooks
├── docs/                    # Documentation
│   ├── methodology/         # Research methodology
│   └── implementation/      # Phase implementation docs
├── outputs/                 # Generated outputs
│   │   └── kaggle_model_evaluation/
│   └── visualizations/      # Attention maps, etc.
├── tests/                   # Tests & verification
│   ├── evaluate_kaggle_model.py  # Model evaluation script
│   └── verify_phase_c.py    # Methodology compliance tests
├── scripts/                 # Utility scripts
│   ├── organize_data.py     # Data organization
│   └── visualize_bam.py     # BAM visualization
└── requirements.txt
```

## Approach

**Biologically-aware preprocessing** using classical computer vision:
- Arrow detection for identity selection (not spatial localization)
- Large contextual crops (60-75%) preserving head/ears/temporal gland
- Upward + forward bias from arrow anchor
- No deep learning in preprocessing phase

## Documentation

- **Methodology**: `docs/methodology/WII_Elephant_ReID_System.pdf`
- **Training Guide**: `kaggle/README_KAGGLE_DEPLOYMENT.md`
- **Phase C Implementation**: `docs/implementation/Phase_C_README.md`
- **Model Evaluation**: `outputs/results/app_model_evaluation/metrics.json`

## Model Performance

**Production Model (Makhna Model):**
- **Rank-1 Accuracy**: 93.59% ⭐
- **Rank-5 Accuracy**: 96.79%
- **Rank-10 Accuracy**: 98.72%
- **mAP**: 80.85%
- **Test Queries**: 156

_Evaluated on held-out test set using `makhna_model.pth`_

## Key Features

### Enhanced Kaggle Training
-  Mixed precision training (2-3x faster)
-  Automatic checkpoints every 5 epochs
-  Early stopping with patience=15
-  Attention map visualization
-  Resume from interrupted training
- 🎯 Gradient clipping & LR warmup

### Model Architecture
-  Dual-branch design (texture + semantic)
-  Biological Attention Maps (BAM)
-  128-dim embeddings (optimized for small datasets)
-  L2 normalization for metric learning
-  Random Erasing (arrow bias prevention)

---

**Wildlife Institute of India Research Project**
