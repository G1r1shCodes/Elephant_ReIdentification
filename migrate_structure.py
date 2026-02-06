"""
Elephant Re-Identification Project - Directory Structure Migration Script

This script safely reorganizes the project to follow ML/CV best practices:
1. Creates backup of current state
2. Creates new directory structure
3. Moves files to appropriate locations
4. Updates import paths in Python files
5. Removes unwanted files/directories

Author: Automated Migration Script
Date: 2026-02-06
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# Root directory
ROOT = Path(__file__).parent

# Backup directory name
BACKUP_DIR = ROOT / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Files and directories to DELETE
TO_DELETE = [
    ".venv",
    "visualization_outputs",
    "Elephant_Methodology/.obsidian"
]

# New directory structure to create
NEW_DIRS = [
    "src",
    "src/preprocessing",
    "src/models",
    "src/utils",
    "docs",
    "docs/methodology",
    "docs/progress",
    "docs/design_notes",
    "outputs",
    "outputs/visualizations",
    "outputs/models",
    "outputs/results",
    "tests",
    "scripts",
    "data/raw"
]

# File migrations: (source, destination)
FILE_MOVES = [
    # Move preprocessing scripts to src/
    ("preprocessing/preprocess.py", "src/preprocessing/preprocess.py"),
    ("preprocessing/preprocess_megadetector.py", "src/preprocessing/preprocess_megadetector.py"),
    
    # Move documentation
    ("preprocessing/CHANGES.md", "docs/design_notes/CHANGES.md"),
    ("preprocessing/CRITICAL_CHANGES.md", "docs/design_notes/CRITICAL_CHANGES.md"),
    ("preprocessing/REDESIGN_ANALYSIS.md", "docs/design_notes/REDESIGN_ANALYSIS.md"),
    ("notebooks/UPDATE_INSTRUCTIONS.md", "docs/design_notes/UPDATE_INSTRUCTIONS.md"),
    ("Progress_Report.docx", "docs/progress/Progress_Report.docx"),
    
    # Move methodology docs
    ("Elephant_Methodology/🐘 WII Elephant Re-Identification System 1.md", 
     "docs/methodology/WII_Elephant_ReID_System.md"),
    ("Elephant_Methodology/🐘 WII Elephant Re-Identification System 1.pdf", 
     "docs/methodology/WII_Elephant_ReID_System.pdf"),
]

# Directory renames: (old, new)
DIR_RENAMES = [
    ("data/Herd_ID_Udalguri_24", "data/raw/Herd_ID_Udalguri_24"),
    ("data/Makhna_id_udalguri_24", "data/raw/Makhna_id_udalguri_24"),
]


def create_backup():
    """Create backup of current state."""
    print(f"\n{'='*80}")
    print("STEP 1: Creating Backup")
    print(f"{'='*80}")
    
    print(f"Backup location: {BACKUP_DIR}")
    
    # Items to backup (exclude .venv as it's large and can be recreated)
    backup_items = [
        "preprocessing",
        "notebooks",
        "Elephant_Methodology",
        "Progress_Report.docx",
        "README.md",
        "requirements.txt"
    ]
    
    BACKUP_DIR.mkdir(exist_ok=True)
    
    for item in backup_items:
        src = ROOT / item
        if src.exists():
            dst = BACKUP_DIR / item
            if src.is_dir():
                shutil.copytree(src, dst)
                print(f"  ✓ Backed up directory: {item}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  ✓ Backed up file: {item}")
    
    print(f"\n✅ Backup completed successfully!")


def create_new_structure():
    """Create new directory structure."""
    print(f"\n{'='*80}")
    print("STEP 2: Creating New Directory Structure")
    print(f"{'='*80}")
    
    for dir_path in NEW_DIRS:
        full_path = ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    print(f"\n✅ New structure created!")


def move_files():
    """Move files to new locations."""
    print(f"\n{'='*80}")
    print("STEP 3: Moving Files to New Locations")
    print(f"{'='*80}")
    
    for src_rel, dst_rel in FILE_MOVES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  ✓ Moved: {src_rel} → {dst_rel}")
        else:
            print(f"  ⚠ Skipped (not found): {src_rel}")
    
    print(f"\n✅ Files moved successfully!")


def rename_directories():
    """Rename directories."""
    print(f"\n{'='*80}")
    print("STEP 4: Reorganizing Data Directory")
    print(f"{'='*80}")
    
    for old_rel, new_rel in DIR_RENAMES:
        old = ROOT / old_rel
        new = ROOT / new_rel
        
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
            print(f"  ✓ Moved: {old_rel} → {new_rel}")
        else:
            print(f"  ⚠ Skipped (not found): {old_rel}")
    
    print(f"\n✅ Data directory reorganized!")


def create_init_files():
    """Create __init__.py files for Python packages."""
    print(f"\n{'='*80}")
    print("STEP 5: Creating Python Package Files")
    print(f"{'='*80}")
    
    init_locations = [
        "src/__init__.py",
        "src/preprocessing/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_locations:
        full_path = ROOT / init_file
        if not full_path.exists():
            full_path.write_text('"""Package initialization."""\n')
            print(f"  ✓ Created: {init_file}")
    
    print(f"\n✅ Package files created!")


def update_import_paths():
    """Update import paths in moved Python files."""
    print(f"\n{'='*80}")
    print("STEP 6: Updating Import Paths")
    print(f"{'='*80}")
    
    # Update preprocess.py to use new data paths
    preprocess_file = ROOT / "src/preprocessing/preprocess.py"
    
    if preprocess_file.exists():
        content = preprocess_file.read_text(encoding='utf-8')
        
        # Update data paths
        content = content.replace(
            'DATA_ROOT = "data"',
            'DATA_ROOT = "../../data"'
        )
        content = content.replace(
            'MAKHNA_RAW = os.path.join(DATA_ROOT, "Makhna_id_udalguri_24", "Makhna_id_udalguri_24")',
            'MAKHNA_RAW = os.path.join(DATA_ROOT, "raw", "Makhna_id_udalguri_24", "Makhna_id_udalguri_24")'
        )
        content = content.replace(
            'HERD_RAW = os.path.join(DATA_ROOT, "Herd_ID_Udalguri_24", "Herd_ID_Udalguri_24")',
            'HERD_RAW = os.path.join(DATA_ROOT, "raw", "Herd_ID_Udalguri_24", "Herd_ID_Udalguri_24")'
        )
        
        preprocess_file.write_text(content, encoding='utf-8')
        print(f"  ✓ Updated: src/preprocessing/preprocess.py")
    
    # Update preprocess_megadetector.py
    preprocess_md_file = ROOT / "src/preprocessing/preprocess_megadetector.py"
    
    if preprocess_md_file.exists():
        content = preprocess_md_file.read_text(encoding='utf-8')
        
        # Update data paths (similar pattern)
        content = content.replace(
            'DATA_ROOT = "data"',
            'DATA_ROOT = "../../data"'
        )
        
        preprocess_md_file.write_text(content, encoding='utf-8')
        print(f"  ✓ Updated: src/preprocessing/preprocess_megadetector.py")
    
    print(f"\n✅ Import paths updated!")


def create_gitignore():
    """Create .gitignore file."""
    print(f"\n{'='*80}")
    print("STEP 7: Creating .gitignore")
    print(f"{'='*80}")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data (exclude large datasets from version control)
data/raw/
data/processed/
data/processed_megadetector/

# Outputs
outputs/visualizations/*.png
outputs/visualizations/*.jpg
outputs/models/*.pth
outputs/models/*.h5
outputs/results/*.csv
outputs/results/*.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Obsidian
.obsidian/

# Backups
backup_*/

# OS
Thumbs.db
.DS_Store

# Logs
*.log

# Environment variables
.env
.env.local

# Documentation builds
docs/_build/
"""
    
    gitignore_file = ROOT / ".gitignore"
    gitignore_file.write_text(gitignore_content, encoding='utf-8')
    print(f"  ✓ Created: .gitignore")

    
    print(f"\n✅ .gitignore created!")


def delete_unwanted():
    """Delete unwanted files and directories."""
    print(f"\n{'='*80}")
    print("STEP 8: Removing Unwanted Files/Directories")
    print(f"{'='*80}")
    
    for item in TO_DELETE:
        path = ROOT / item
        if path.exists():
            try:
                if path.is_dir():
                    # Special handling for .venv - it might be in use
                    if item == ".venv":
                        print(f"  ⚠ Skipping .venv (may be in use - delete manually later)")
                        continue
                    shutil.rmtree(path)
                    print(f"  ✓ Deleted directory: {item}")
                else:
                    path.unlink()
                    print(f"  ✓ Deleted file: {item}")
            except (PermissionError, OSError) as e:
                print(f"  ⚠ Could not delete {item}: {e}")
                print(f"     You can delete it manually later")
        else:
            print(f"  ⚠ Already removed: {item}")
    
    # Remove now-empty preprocessing directory
    preprocessing_dir = ROOT / "preprocessing"
    try:
        if preprocessing_dir.exists() and not any(preprocessing_dir.iterdir()):
            preprocessing_dir.rmdir()
            print(f"  ✓ Removed empty directory: preprocessing")
    except Exception as e:
        print(f"  ⚠ Could not remove preprocessing directory: {e}")
    
    # Remove now-empty Elephant_Methodology directory
    methodology_dir = ROOT / "Elephant_Methodology"
    try:
        if methodology_dir.exists() and not any(methodology_dir.iterdir()):
            methodology_dir.rmdir()
            print(f"  ✓ Removed empty directory: Elephant_Methodology")
    except Exception as e:
        print(f"  ⚠ Could not remove Elephant_Methodology directory: {e}")
    
    print(f"\n✅ Cleanup completed (some items may need manual deletion)!")


def update_readme():
    """Update README with new structure."""
    print(f"\n{'='*80}")
    print("STEP 9: Updating README")
    print(f"{'='*80}")
    
    readme_content = """# Elephant Re-Identification (WII)

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
.venv\\Scripts\\activate
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
"""
    
    readme_file = ROOT / "README.md"
    readme_file.write_text(readme_content, encoding='utf-8')
    print(f"  ✓ Updated: README.md")

    
    print(f"\n✅ README updated!")


def create_migration_summary():
    """Create a summary of the migration."""
    print(f"\n{'='*80}")
    print("STEP 10: Creating Migration Summary")
    print(f"{'='*80}")
    
    summary = f"""# Migration Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Changes Made

### 1. Backup Created
- Location: {BACKUP_DIR.name}
- Contains: Original preprocessing/, notebooks/, Elephant_Methodology/, and key files

### 2. New Directory Structure
Created the following directories:
{chr(10).join(f'  - {d}' for d in NEW_DIRS)}

### 3. Files Moved
{chr(10).join(f'  - {src} → {dst}' for src, dst in FILE_MOVES)}

### 4. Directories Reorganized
{chr(10).join(f'  - {old} → {new}' for old, new in DIR_RENAMES)}

### 5. Files/Directories Deleted
{chr(10).join(f'  - {item}' for item in TO_DELETE)}

### 6. New Files Created
  - .gitignore
  - src/__init__.py
  - src/preprocessing/__init__.py
  - src/models/__init__.py
  - src/utils/__init__.py
  - tests/__init__.py
  - README.md (updated)

### 7. Import Paths Updated
  - src/preprocessing/preprocess.py (data paths updated)
  - src/preprocessing/preprocess_megadetector.py (data paths updated)

## Next Steps

1. Verify the new structure works correctly
2. Run preprocessing from new location: `cd src/preprocessing && python preprocess.py`
3. If everything works, you can delete the backup: `{BACKUP_DIR.name}`
4. Initialize git repository if not already done: `git init`
5. Add files to git: `git add .`

## Rollback Instructions

If you need to rollback:
1. Delete the new directories: src/, docs/, outputs/, tests/, scripts/
2. Restore from backup: Copy contents of {BACKUP_DIR.name} back to root
3. Restore data structure manually if needed

## Notes

- Virtual environment (.venv) was deleted - recreate with: `python -m venv .venv`
- Visualization outputs were deleted - can be regenerated
- .obsidian directory was removed (IDE metadata)
"""
    
    summary_file = ROOT / "MIGRATION_SUMMARY.md"
    summary_file.write_text(summary, encoding='utf-8')
    print(f"  ✓ Created: MIGRATION_SUMMARY.md")

    
    print(f"\n✅ Migration summary created!")


def main():
    """Main migration function."""
    print("\n" + "="*80)
    print("ELEPHANT RE-IDENTIFICATION PROJECT - STRUCTURE MIGRATION")
    print("="*80)
    print("\nThis script will:")
    print("  1. Create a backup of current state")
    print("  2. Create new directory structure")
    print("  3. Move files to new locations")
    print("  4. Update import paths")
    print("  5. Delete unwanted files")
    print("  6. Create .gitignore and package files")
    print("\n" + "="*80)
    
    try:
        create_backup()
        create_new_structure()
        move_files()
        rename_directories()
        create_init_files()
        update_import_paths()
        create_gitignore()
        delete_unwanted()
        update_readme()
        create_migration_summary()
        
        print("\n" + "="*80)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📦 Backup saved to: {BACKUP_DIR.name}")
        print("📄 See MIGRATION_SUMMARY.md for details")
        print("\n🔍 Next steps:")
        print("  1. Review the new structure")
        print("  2. Test preprocessing: cd src/preprocessing && python preprocess.py")
        print("  3. If everything works, delete the backup folder")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: Migration failed!")
        print(f"Error: {e}")
        print(f"\n⚠️  Backup is available at: {BACKUP_DIR.name}")
        print("You can manually restore from the backup if needed.")
        raise


if __name__ == "__main__":
    main()
