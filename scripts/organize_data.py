"""
Organize Processed Data by Individual Identity

This script reorganizes preprocessed images into individual-based directories
for training the re-identification model.

Input structure:
    data/processed/Makhna/  (all images mixed)
    data/processed/Herd/    (all images mixed)

Output structure:
    data/processed/Makhna/Makhna_1/  (images of individual 1)
    data/processed/Makhna/Makhna_2/  (images of individual 2)
    data/processed/Herd/Herd_1/      (images of individual 1)
    ...
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import re


def extract_individual_from_filename(filename: str, category: str) -> str:
    """
    Extract individual ID from processed filename.
    
    Processed filenames follow pattern: OriginalName_crop.jpg
    We need to map back to the original directory structure.
    
    Args:
        filename: Processed image filename
        category: 'Makhna' or 'Herd'
        
    Returns:
        Individual ID (e.g., 'Makhna_1', 'Herd_2')
    """
    # Remove _crop suffix and extension
    base_name = filename.replace('_crop.jpg', '').replace('_crop.JPG', '')
    base_name = base_name.replace('.jpg', '').replace('.JPG', '')
    
    return base_name


def find_original_individual(processed_filename: str, raw_root: Path, category: str) -> str:
    """
    Find which individual directory the original image came from.
    
    Args:
        processed_filename: Name of processed file
        raw_root: Root directory of raw data
        category: 'Makhna' or 'Herd'
        
    Returns:
        Individual ID (e.g., 'Makhna_1')
    """
    # Extract original filename (remove _crop suffix)
    original_name = processed_filename.replace('_crop.jpg', '.jpg').replace('_crop.JPG', '.JPG')
    original_name = original_name.replace('_crop.jpeg', '.jpeg')
    
    # Search for this file in raw data structure
    if category == 'Makhna':
        search_root = raw_root / 'Makhna_id_udalguri_24' / 'Makhna_id_udalguri_24'
    else:
        search_root = raw_root / 'Herd_ID_Udalguri_24' / 'Herd_ID_Udalguri_24'
    
    # Find the file
    for individual_dir in search_root.iterdir():
        if not individual_dir.is_dir():
            continue
        
        # Check if original file exists in this individual's directory
        for img_file in individual_dir.rglob('*'):
            if img_file.name == original_name:
                return individual_dir.name
    
    # If not found, return 'Unknown'
    return 'Unknown'


def organize_processed_data(processed_root: Path, raw_root: Path, dry_run: bool = True):
    """
    Organize processed data by individual identity.
    
    Args:
        processed_root: Root of processed data (data/processed)
        raw_root: Root of raw data (data/raw)
        dry_run: If True, only print what would be done
    """
    print("=" * 80)
    print("ORGANIZING PROCESSED DATA BY INDIVIDUAL IDENTITY")
    print("=" * 80)
    print(f"Processed root: {processed_root}")
    print(f"Raw root: {raw_root}")
    print(f"Dry run: {dry_run}")
    print("=" * 80)
    
    # Process both categories
    for category in ['Makhna', 'Herd']:
        print(f"\n{'='*80}")
        print(f"Processing {category} dataset...")
        print(f"{'='*80}")
        
        category_dir = processed_root / category
        if not category_dir.exists():
            print(f"⚠️  {category} directory not found, skipping...")
            continue
        
        # Collect all images currently in the category directory
        images = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.JPG'))
        print(f"Found {len(images)} processed images")
        
        # Map each image to its individual
        individual_mapping = defaultdict(list)
        
        for img_path in images:
            # Skip if it's already in a subdirectory
            if img_path.parent != category_dir:
                continue
            
            # Find which individual this belongs to
            individual_id = find_original_individual(img_path.name, raw_root, category)
            individual_mapping[individual_id].append(img_path)
        
        # Print summary
        print(f"\nFound {len(individual_mapping)} individuals:")
        for individual_id, img_list in sorted(individual_mapping.items()):
            print(f"  {individual_id}: {len(img_list)} images")
        
        # Reorganize files
        print(f"\n{'Reorganizing files...' if not dry_run else 'DRY RUN - Would reorganize files:'}")
        
        for individual_id, img_list in sorted(individual_mapping.items()):
            # Create individual directory
            individual_dir = category_dir / individual_id
            
            if not dry_run:
                individual_dir.mkdir(exist_ok=True)
            
            # Move images
            for img_path in img_list:
                new_path = individual_dir / img_path.name
                
                if dry_run:
                    print(f"  Would move: {img_path.name} -> {individual_id}/")
                else:
                    shutil.move(str(img_path), str(new_path))
            
            if not dry_run:
                print(f"  ✓ Moved {len(img_list)} images to {individual_id}/")
    
    print(f"\n{'='*80}")
    if dry_run:
        print("DRY RUN COMPLETE - No files were moved")
        print("Run with dry_run=False to actually reorganize files")
    else:
        print("REORGANIZATION COMPLETE!")
    print(f"{'='*80}")


def verify_organization(processed_root: Path):
    """
    Verify the organization is correct.
    
    Args:
        processed_root: Root of processed data
    """
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    for category in ['Makhna', 'Herd']:
        category_dir = processed_root / category
        if not category_dir.exists():
            continue
        
        print(f"\n{category}:")
        
        # Count individuals
        individuals = [d for d in category_dir.iterdir() if d.is_dir()]
        print(f"  Total individuals: {len(individuals)}")
        
        # Count images per individual
        total_images = 0
        for individual_dir in individuals:
            num_images = len(list(individual_dir.glob('*.jpg')) + list(individual_dir.glob('*.JPG')))
            total_images += num_images
            print(f"    {individual_dir.name}: {num_images} images")
        
        print(f"  Total images: {total_images}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Paths - use current working directory
    PROJECT_ROOT = Path.cwd()
    PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
    RAW_ROOT = PROJECT_ROOT / "data" / "raw"
    
    print("Elephant Re-Identification - Data Organization")
    print("=" * 80)
    
    # First, do a dry run
    print("\n🔍 DRY RUN - Checking what would be done...\n")
    organize_processed_data(PROCESSED_ROOT, RAW_ROOT, dry_run=True)
    
    # Ask for confirmation
    print("\n" + "=" * 80)
    response = input("\nProceed with reorganization? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\n🚀 Starting reorganization...\n")
        organize_processed_data(PROCESSED_ROOT, RAW_ROOT, dry_run=False)
        
        # Verify
        verify_organization(PROCESSED_ROOT)
        
        print("\n✅ Data organization complete!")
        print("You can now train the model with: python src/models/train.py")
    else:
        print("\n❌ Reorganization cancelled")
