import os
import random
import shutil
from pathlib import Path

def create_demo_dataset():
    source_root = Path("data/restructured")
    dest_root = Path("data/demo_elephants")
    
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)
    
    # Get all identity folders
    identities = [d for d in source_root.iterdir() if d.is_dir()]
    
    # Separate Makhnas and Herds
    makhnas = [d for d in identities if d.name.lower().startswith("makhna")]
    herds = [d for d in identities if not d.name.lower().startswith("makhna")]
    
    print(f"Found {len(makhnas)} Makhnas and {len(herds)} Herd members.")
    
    # Prioritize Makhnas: Take almost all of them (or random 10)
    # Plus a few random herd members for variety
    selected_makhnas = random.sample(makhnas, min(len(makhnas), 12))
    selected_herds = random.sample(herds, min(len(herds), 3)) 
    
    # Combine (mostly Makhnas)
    selected_identities = selected_makhnas + selected_herds
    random.shuffle(selected_identities)
    
    count = 0
    print(f"Creating demo dataset in {dest_root} (Makhna Focused)...")
    
    for identity_dir in selected_identities:
        # Get images
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        if not images:
            continue
            
        # Select 1 image per identity (to maximize diversity)
        # Or 2 if we need more volume
        img_path = random.choice(images)
        
        # Create a clear filename: IdentityName_OriginalName
        clean_id = identity_dir.name.replace(" ", "_")
        new_name = f"{clean_id}__{img_path.name}"
        dest_path = dest_root / new_name
        
        shutil.copy2(img_path, dest_path)
        print(f"  Copied: {new_name}")
        count += 1
        
    print(f"\n✅ Created {count} demo images in 'data/demo_elephants'")

if __name__ == "__main__":
    create_demo_dataset()
