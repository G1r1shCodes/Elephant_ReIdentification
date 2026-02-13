"""
Debug script to check what the preprocessing script sees
"""
import os
from pathlib import Path

# Same paths as preprocess_megadetector.py
DATA_ROOT = "../../data"
MAKHNA_RAW = os.path.join(DATA_ROOT, "raw", "Makhna_id_udalguri_24", "Makhna_id_udalguri_24")
HERD_RAW = os.path.join(DATA_ROOT, "raw", "Herd_ID_Udalguri_24", "Herd_ID_Udalguri_24")

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.NRW', '.nrw'}

print(f"Checking MAKHNA_RAW: {MAKHNA_RAW}")
print(f"Absolute path: {os.path.abspath(MAKHNA_RAW)}")
print(f"Path exists: {os.path.exists(MAKHNA_RAW)}")

if os.path.exists(MAKHNA_RAW):
    all_files = []
    for root, dirs, files in os.walk(MAKHNA_RAW):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext in VALID_EXTENSIONS:
                all_files.append((filename, ext))
    
    print(f"\nFound {len(all_files)} valid files")
    print(f"Extensions found: {set(ext for _, ext in all_files)}")
    print(f"Sample files:")
    for fname, ext in all_files[:10]:
        print(f"  {fname} ({ext})")
else:
    print("Path does not exist!")

# Check HERD too
print(f"\n\nChecking HERD_RAW: {HERD_RAW}")
print(f"Absolute path: {os.path.abspath(HERD_RAW)}")
print(f"Path exists: {os.path.exists(HERD_RAW)}")
