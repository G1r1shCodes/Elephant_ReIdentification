# Elephant Re-Identification System

Open-set biometric elephant re-identification for Wildlife Institute of India.

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run preprocessing
cd src/preprocessing
python preprocess.py
```

## Project Structure

```
├── data/                    # Datasets (gitignored)
│   ├── raw/                 # Original annotated images
│   ├── processed/           # Preprocessed crops
│   └── processed_megadetector/
├── src/                     # Source code
│   ├── preprocessing/       # Preprocessing scripts
│   ├── models/              # Models (future)
│   └── utils/               # Utilities
├── notebooks/               # Jupyter notebooks
├── docs/                    # Documentation
│   ├── methodology/         # Research methodology
│   ├── progress/            # Progress reports
│   └── design_notes/        # Design decisions
├── outputs/                 # Generated outputs
├── tests/                   # Unit tests
└── scripts/                 # Standalone scripts
```

## Approach

**Biologically-aware preprocessing** using classical computer vision:
- Arrow detection for identity selection (not spatial localization)
- Large contextual crops (60-75%) preserving head/ears/temporal gland
- Upward + forward bias from arrow anchor
- No deep learning in preprocessing phase

## Phases

1. ✅ **Phase A**: Data collection & annotation
2. 🔄 **Phase B**: Biologically-aware preprocessing (current)
3. 📋 **Phase C**: Feature extraction (planned)
4. 📋 **Phase D**: Matching & re-identification (planned)

## Documentation

- Methodology: `docs/methodology/WII_Elephant_ReID_System.pdf`
- Design notes: `docs/design_notes/`
- Progress: `docs/progress/`

---

**Wildlife Institute of India Research Project**
