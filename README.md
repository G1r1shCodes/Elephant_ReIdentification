# Elephant Re-Identification (WII)

This repository implements an open-set biometric elephant re-identification system for the Wildlife Institute of India.

## 📁 Project Structure

```
Elephant_ReIdentification/
│
├── 📂 data/                          # Dataset (excluded from git)
│   ├── raw/                          # Original images with annotations
│   ├── processed/                    # Preprocessed crops
│   └── processed_megadetector/       # MegaDetector outputs
│
├── 📂 src/                           # Source code
│   ├── preprocessing/                # Data preprocessing scripts
│   ├── models/                       # Model architectures (future)
│   └── utils/                        # Utility functions
│
├── 📂 notebooks/                     # Jupyter notebooks for exploration
│
├── 📂 docs/                          # Documentation
│   ├── methodology/                  # Research methodology
│   ├── progress/                     # Progress reports
│   └── design_notes/                 # Design decisions and changes
│
├── 📂 outputs/                       # Generated outputs
│   ├── visualizations/               # Plots and comparisons
│   ├── models/                       # Trained models
│   └── results/                      # Experiment results
│
├── 📂 tests/                         # Unit tests
│
├── 📂 scripts/                       # Standalone scripts
│
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing

```bash
# Navigate to preprocessing directory
cd src/preprocessing

# Run preprocessing script
python preprocess.py
```

## 📚 Documentation

- **Methodology**: See `docs/methodology/WII_Elephant_ReID_System.pdf`
- **Design Notes**: See `docs/design_notes/`
- **Progress Reports**: See `docs/progress/`

## 🔬 Research Approach

This project follows a biologically-aware approach to elephant re-identification:

1. **Phase A**: Data collection and annotation
2. **Phase B**: Biologically-aware preprocessing (current)
3. **Phase C**: Feature extraction (planned)
4. **Phase D**: Matching and re-identification (planned)

### Key Principles

- **Arrow as Identity Selector**: Arrows indicate which elephant, not where to crop
- **Biological Bias**: Crops prioritize head/ears/temporal gland regions
- **Large Contextual Crops**: Preserve identity-bearing anatomy
- **Deterministic Processing**: Classical CV, no deep learning in preprocessing

## 📝 Code Implementation

Code follows the approved algorithmic roadmap documented in `docs/methodology/`.

## 🧪 Testing

```bash
# Run tests (when available)
pytest tests/
```

## 📄 License

Wildlife Institute of India Research Project
