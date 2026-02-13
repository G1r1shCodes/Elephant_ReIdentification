
import json
import re

source_path = "kaggle/elephant_reid_training_arcface.ipynb"
dest_path = "kaggle/elephant_reid_training_hard_aware.ipynb"

print(f"Reading {source_path}...")
with open(source_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The code to inject
sampler_injection = [
    "# --- HARD SAMPLE AWARE TRAINING ---\n",
    "from torch.utils.data import WeightedRandomSampler\n",
    "\n",
    "# Weights derived from Mining Report (Step 1257)\n",
    "# Identities with high error rates get boosted visibility\n",
    "hard_weights_map = {\n",
    "    'Makhna_14': 3.0,  # 100% Error\n",
    "    'Makhna_16': 3.0,  # 100% Error\n",
    "    'Makhna_18': 3.0,  # 100% Error\n",
    "    'Makhna_19': 3.0,  # 100% Error (Single sample)\n",
    "    'Makhna_7':  3.0,  # 75% Error\n",
    "    'Makhna_15': 2.5,  # 55% Error\n",
    "    'Makhna_5':  2.0,  # 43% Error\n",
    "    'Makhna_3':  1.8,  # 40% Error\n",
    "    'Makhna_1':  1.5,  # 35% Error\n",
    "    'Makhna_17': 2.0,  # 50% Error\n",
    "}\n",
    "default_weight = 1.0\n",
    "\n",
    "print('\\n⚖️  Configuring Hard Sample Weights...')\n",
    "# 1. Calculate weights for every sample in the dataset\n",
    "sample_weights = []\n",
    "all_labels = dataset.targets if hasattr(dataset, 'targets') else [s[1] for s in dataset.samples]\n",
    "all_classes = dataset.classes\n",
    "\n",
    "for label_idx in all_labels:\n",
    "    class_name = all_classes[label_idx]\n",
    "    w = hard_weights_map.get(class_name, default_weight)\n",
    "    sample_weights.append(w)\n",
    "\n",
    "sample_weights = torch.DoubleTensor(sample_weights)\n",
    "\n",
    "# 2. Create Sampler\n",
    "# Replacement=True is crucial for oversampling\n",
    "sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)\n",
    "print(f'   ✅ WeightedSampler ready with {len(sample_weights)} samples')\n",
    "\n",
    "# 3. Create DataLoader with Sampler (Shuffle must be False)\n",
    "train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)\n",
    "print('   ✅ DataLoader updated to use Hard Sample-Aware Sampler')\n"
]

# Find the cell that creates the DataLoader and replace it
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_code = "".join(cell['source'])
        # Look for the original DataLoader call
        if "train_loader = DataLoader" in source_code and "shuffle=True" in source_code:
            print("Found DataLoader cell. Injecting sampler logic...")
            # We replace the entire content of this cell with our new logic
            # This assumes the dataset variable is available as 'dataset' or 'train_dataset'
            # In previous notebooks it was 'full_dataset' or 'train_dataset'
            # Let's check the context.
            # Usually: train_loader = DataLoader(full_dataset, ...)
            
            # To be safe, let's keep the imports from the original cell if any, 
            # or just append our logic *after* the dataset is defined but *before* the loader?
            # Actually, standardizing on replacing the DataLoader block is best.
            
            # Let's try to preserve the variable name 'dataset' or 'train_dataset'
            dataset_var = "train_dataset"
            if "full_dataset" in source_code:
                dataset_var = "full_dataset"
            
            # Fix variable name in my injection code
            new_code = []
            for line in sampler_injection:
                 new_code.append(line.replace("dataset", dataset_var))
            
            cell['source'] = new_code
            found = True
            break

if not found:
    print("Warning: Could not find exact DataLoader cell to patch. Searching for partial match...")
    # Fallback: Look for any DataLoader
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
             if "DataLoader(" in "".join(cell['source']):
                 print("Found a DataLoader cell (fallback). Injecting...")
                 # Append to the *end* of the cell to overwrite train_loader?
                 # No, that's risky.
                 # Let's just create a NEW cell after the dataset creation.
                 pass

if found:
    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"✅ Created {dest_path}")
else:
    print("❌ Failed to patch notebook.")

