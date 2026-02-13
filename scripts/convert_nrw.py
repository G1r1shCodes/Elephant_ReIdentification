"""
Simple preprocessing script to process NRW files.
"""
import os
from pathlib import Path
import cv2
import rawpy

# Paths
RAW_DATA = Path("d:/Elephant_ReIdentification/data/raw")
PROCESSED_DATA = Path("d:/Elephant_ReIdentification/data/processed_megadetector")

def convert_nrw_to_jpg(nrw_path, output_path):
    """Convert NRW to JPG"""
    try:
        with rawpy.imread(str(nrw_path)) as raw:
            rgb = raw.postprocess()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JPG
            cv2.imwrite(str(output_path), bgr)
            return True
    except Exception as e:
        print(f"Error converting {nrw_path.name}: {e}")
        return False

def process_all_nrw():
    """Process all NRW files in raw data"""
    nrw_files = list(RAW_DATA.rglob("*.NRW"))
    print(f"Found {len(nrw_files)} NRW files")
    
    processed = 0
    for nrw_file in nrw_files:
        # Get relative path from raw data
        rel_path = nrw_file.relative_to(RAW_DATA)
        
        # Create output path (change extension to .jpg)
        output_path = PROCESSED_DATA / rel_path.parent / f"{rel_path.stem}.jpg"
        
        print(f"Processing: {nrw_file.name} -> {output_path}")
        
        if convert_nrw_to_jpg(nrw_file, output_path):
            processed += 1
            print(f"  ✓ Converted ({processed}/{len(nrw_files)})")
        else:
            print(f"  ✗ Failed")
    
    print(f"\nComplete! Processed {processed}/{len(nrw_files)} files")

if __name__ == "__main__":
    process_all_nrw()
