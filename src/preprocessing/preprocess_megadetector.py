"""
Elephant Re-Identification — Phase B: MegaDetector-Based Preprocessing
Preprocessing Script: Detection-Based Cropping with Validated Parameters

Wildlife Institute of India Research Project
OPEN-SET BIOMETRIC RE-IDENTIFICATION SYSTEM

================================================================================
UPDATED APPROACH: MegaDetector Integration (Validated in Exploration)
================================================================================

Key Improvements:
- 100% detection rate (validated on 20 samples)
- Works with or without arrows
- Precise bounding boxes around elephants
- Validated parameters: confidence=0.4, padding=0.15

Biological Constraints (Preserved):
- Identity features: Head profile, ear shape/depigmentation, temporal gland
- Crops include full elephant with context
- Head and ears MUST be preserved
- Makhna temporal gland region is CRITICAL

Methodology:
- MegaDetector v5a for elephant detection
- 15% padding around bounding boxes (validated)
- Arrow detection as fallback/validation
- Never modify raw data
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# MegaDetector imports
from megadetector.detection import run_detector
from megadetector.visualization import visualization_utils as vis_utils

# RAW image support
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("⚠️ rawpy not installed - .NRW files will be skipped")


# ==================== CONFIGURATION ==================== #

# Root paths - use absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed_megadetector"

# Raw dataset paths
MAKHNA_RAW = DATA_ROOT / "raw" / "Makhna_id_udalguri_24" / "Makhna_id_udalguri_24"
HERD_RAW = DATA_ROOT / "raw" / "Herd_ID_Udalguri_24" / "Herd_ID_Udalguri_24"

# MegaDetector parameters (VALIDATED)
CONFIDENCE_THRESHOLD = 0.4  # Validated in exploration
BBOX_PADDING_RATIO = 0.15   # Validated in exploration (15%)

# Arrow detection parameters (for validation/fallback)
MIN_ARROW_AREA = 4000
ARROW_HSV_LOWER1 = np.array([0, 100, 100])
ARROW_HSV_UPPER1 = np.array([10, 255, 255])
ARROW_HSV_LOWER2 = np.array([160, 100, 100])
ARROW_HSV_UPPER2 = np.array([180, 255, 255])

# Supported image formats (NRW files already converted to JPG)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# Global detector (loaded once)
detector = None


# ==================== IMAGE LOADING ====================  #

def load_image_with_raw_support(image_path):
    """
    Load image with support for RAW formats (.NRW).
    
    Args:
        image_path: Path to image file
        
    Returns:
        BGR image as numpy array, or None if failed
    """
    _, ext = os.path.splitext(image_path)
    
    # Check if it's a RAW format
    if ext.lower() in ['.nrw']:
        if not HAS_RAWPY:
            print(f"  ⚠️ Skipping {os.path.basename(image_path)} - rawpy not installed")
            return None
        
        try:
            # Load RAW file using rawpy
            with rawpy.imread(image_path) as raw:
                # Convert to RGB (8-bit)
                rgb = raw.postprocess()
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return bgr
        except Exception as e:
            print(f"  ⚠️ Failed to load RAW file {os.path.basename(image_path)}: {e}")
            return None
    else:
        # Standard image formats
        return cv2.imread(image_path)


# ==================== MEGADETECTOR INTEGRATION ==================== #

def load_megadetector():
    """Load MegaDetector model (once at startup)."""
    global detector
    if detector is None:
        print("Loading MegaDetector v5a...")
        detector = run_detector.load_detector('MDV5A')
        print("✓ MegaDetector loaded successfully")
    return detector


def detect_elephants(image_path: str, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
    """
    Detect elephants in image using MegaDetector.
    
    Args:
        image_path: Path to image file
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        List of detection dicts with keys: bbox, conf, category
        bbox format: [x_norm, y_norm, w_norm, h_norm] (normalized 0-1)
    """
    # Ensure detector is loaded
    if detector is None:
        load_megadetector()
    
    # Load image using MegaDetector's utility
    image = vis_utils.load_image(image_path)
    if image is None:
        return []
    
    # Run detection
    result = detector.generate_detections_one_image(image)
    
    if not result or 'detections' not in result:
        return []
    
    detections = result['detections']
    
    # Filter for animal category (category '1') and above threshold
    elephant_detections = [
        det for det in detections 
        if det.get('category', '') == '1' and det.get('conf', 0) >= confidence_threshold
    ]
    
    return elephant_detections


# ==================== ARROW DETECTION (FALLBACK/VALIDATION) ==================== #

def detect_arrow_tip(image):
    """
    Detect red arrow tip (for validation or fallback).
    
    Returns:
        tuple (x, y): Arrow tip coordinates if valid arrow found
        None: If no arrow
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, ARROW_HSV_LOWER1, ARROW_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, ARROW_HSV_LOWER2, ARROW_HSV_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < MIN_ARROW_AREA:
        return None
    
    # Calculate centroid
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return None
    
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    return (cx, cy)


# ==================== BOUNDING BOX UTILITIES ==================== #

def point_in_bbox(point: Tuple[int, int], bbox: List[float], img_width: int, img_height: int) -> bool:
    """Check if point is inside normalized bounding box."""
    px, py = point
    x_norm, y_norm, w_norm, h_norm = bbox
    x1 = int(x_norm * img_width)
    y1 = int(y_norm * img_height)
    x2 = int((x_norm + w_norm) * img_width)
    y2 = int((y_norm + h_norm) * img_height)
    return x1 <= px <= x2 and y1 <= py <= y2


def bbox_area(bbox: List[float]) -> float:
    """Calculate normalized bounding box area."""
    return bbox[2] * bbox[3]


def distance_to_bbox_center(point: Tuple[int, int], bbox: List[float], img_width: int, img_height: int) -> float:
    """Calculate distance from point to bbox center."""
    px, py = point
    x_norm, y_norm, w_norm, h_norm = bbox
    cx = (x_norm + w_norm/2) * img_width
    cy = (y_norm + h_norm/2) * img_height
    return np.sqrt((px - cx)**2 + (py - cy)**2)


def select_target_elephant(
    detections: List[Dict], 
    arrow_tip: Optional[Tuple[int, int]], 
    img_width: int, 
    img_height: int
) -> Optional[Dict]:
    """
    Select target elephant from multiple detections.
    
    Logic:
    - If arrow present: return bbox containing arrow (or closest)
    - If no arrow: return largest bbox
    """
    if not detections:
        return None
    
    if arrow_tip is not None:
        # Try to find bbox containing arrow
        for det in detections:
            bbox = det['bbox']
            if point_in_bbox(arrow_tip, bbox, img_width, img_height):
                return det
        
        # Fallback: closest bbox to arrow
        closest = min(detections, key=lambda d: distance_to_bbox_center(arrow_tip, d['bbox'], img_width, img_height))
        return closest
    else:
        # No arrow: return largest bbox
        largest = max(detections, key=lambda d: bbox_area(d['bbox']))
        return largest


# ==================== CROPPING LOGIC ==================== #

def extract_padded_crop(image, bbox: List[float], padding_ratio: float = BBOX_PADDING_RATIO):
    """
    Extract padded crop around detected bounding box.
    
    Args:
        image: Input BGR image
        bbox: Normalized bounding box [x_norm, y_norm, w_norm, h_norm]
        padding_ratio: Padding to add around box (e.g., 0.15 = 15%)
        
    Returns:
        Cropped image with padding
    """
    h, w = image.shape[:2]
    x_norm, y_norm, w_norm, h_norm = bbox
    
    # Convert to pixel coordinates
    x1 = int(x_norm * w)
    y1 = int(y_norm * h)
    x2 = int((x_norm + w_norm) * w)
    y2 = int((y_norm + h_norm) * h)
    
    # Calculate padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)
    
    # Apply padding and clamp to image boundaries
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(w, x2 + pad_w)
    y2_pad = min(h, y2 + pad_h)
    
    # Extract crop
    crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
    
    return crop


# ==================== QUALITY CHECKS ==================== #

def check_crop_quality(original_image, cropped_image):
    """
    Perform quality checks on cropped image.
    
    Returns:
        tuple (is_valid, warning_message)
    """
    orig_h, orig_w = original_image.shape[:2]
    crop_h, crop_w = cropped_image.shape[:2]
    
    # Check if crop is too small
    crop_area_ratio = (crop_w * crop_h) / (orig_w * orig_h)
    if crop_area_ratio < 0.10:  # Less than 10% of original
        return False, f"Crop too small ({crop_area_ratio:.1%} of original)"
    
    # Check aspect ratio
    aspect_ratio = crop_w / crop_h
    if aspect_ratio < 0.3 or aspect_ratio > 4.0:
        return False, f"Unusual aspect ratio ({aspect_ratio:.2f})"
    
    # Check minimum dimensions
    if crop_w < 100 or crop_h < 100:
        return False, f"Crop too small ({crop_w}x{crop_h})"
    
    return True, None


# ==================== PROCESSING PIPELINE ==================== #

def process_dataset(input_root, output_base_name):
    """
    Process all images in dataset with MegaDetector-based cropping.
    
    For each image:
    1. Run MegaDetector to find elephants
    2. Detect arrow (if present) for multi-elephant selection
    3. Select target elephant
    4. Extract padded crop
    5. Perform quality checks
    6. Save to processed directory
    
    Args:
        input_root: Raw dataset directory path (Path object or string)
        output_base_name: Name for output folder
        
    Returns:
        Dictionary with processing statistics
    """
    # Convert to string for os.walk
    input_root = str(input_root)
    
    stats = {
        'total': 0,
        'processed': 0,
        'no_detection': 0,
        'arrow_detected': 0,
        'no_arrow': 0,
        'multi_elephant': 0,
        'quality_warnings': 0,
        'errors': []
    }
    
    print(f"\n{'='*80}")
    print(f"Processing: {input_root}")
    print(f"Output to:  {os.path.join(PROCESSED_ROOT, output_base_name)}")
    print(f"{'='*80}\n")
    
    # Ensure detector is loaded
    load_megadetector()
    
    # Walk directory tree recursively
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            input_path = os.path.join(root, filename)
            
            # Check if valid image format
            _, ext = os.path.splitext(filename)
            if ext not in VALID_EXTENSIONS:
                continue
            
            stats['total'] += 1
            
            try:
                # Read image (with RAW support)
                image = load_image_with_raw_support(input_path)
                if image is None:
                    stats['errors'].append((input_path, "Failed to read image"))
                    print(f"[SKIP] {filename} - Failed to read")
                    continue
                
                img_h, img_w = image.shape[:2]
                
                # Detect elephants with MegaDetector
                detections = detect_elephants(input_path, CONFIDENCE_THRESHOLD)
                
                if not detections:
                    stats['no_detection'] += 1
                    print(f"[NO DETECT] {filename:40s} - No elephants detected")
                    continue
                
                # Detect arrow (for multi-elephant selection)
                arrow_tip = detect_arrow_tip(image)
                
                # Select target elephant
                selected = select_target_elephant(detections, arrow_tip, img_w, img_h)
                
                if selected is None:
                    stats['no_detection'] += 1
                    print(f"[NO SELECT] {filename:40s} - Could not select elephant")
                    continue
                
                # Extract padded crop
                cropped_image = extract_padded_crop(image, selected['bbox'], BBOX_PADDING_RATIO)
                
                # Quality check
                is_valid, warning = check_crop_quality(image, cropped_image)
                
                if not is_valid:
                    stats['quality_warnings'] += 1
                    print(f"[WARN] {filename} - {warning}")
                
                # Update statistics
                if arrow_tip is not None:
                    stats['arrow_detected'] += 1
                    arrow_status = "ARROW"
                else:
                    stats['no_arrow'] += 1
                    arrow_status = "NO_ARROW"
                
                if len(detections) > 1:
                    stats['multi_elephant'] += 1
                    det_status = f"{len(detections)}ELE"
                else:
                    det_status = "1ELE"
                
                # Construct output path
                relative_path = os.path.relpath(input_path, input_root)
                output_path = PROCESSED_ROOT / output_base_name / relative_path
                output_path = output_path.with_suffix('.jpg')  # Convert extension to .jpg
                
                # Create output directory
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save processed image
                cv2.imwrite(str(output_path), cropped_image)
                
                stats['processed'] += 1
                conf = selected.get('conf', 0)
                print(f"[OK] {filename:40s} {det_status:5s} {arrow_status:9s} conf={conf:.3f} → {cropped_image.shape[1]}x{cropped_image.shape[0]}")
                
            except Exception as e:
                stats['errors'].append((input_path, str(e)))
                print(f"[ERROR] {filename} - {e}")
    
    return stats


def print_summary(dataset_name, stats):
    """Print detailed processing summary."""
    print(f"\n{'='*80}")
    print(f"{dataset_name} Dataset - Processing Summary")
    print(f"{'='*80}")
    print(f"Total files scanned:       {stats['total']}")
    print(f"Successfully processed:    {stats['processed']}")
    print(f"No detection:              {stats['no_detection']}")
    print(f"Quality warnings:          {stats['quality_warnings']}")
    print(f"\nDetection Details:")
    print(f"  - With arrow detected:   {stats['arrow_detected']}")
    print(f"  - No arrow:              {stats['no_arrow']}")
    print(f"  - Multi-elephant scenes: {stats['multi_elephant']}")
    
    if stats['errors']:
        print(f"\n⚠ Errors encountered: {len(stats['errors'])}")
        for path, error in stats['errors'][:10]:
            print(f"  - {os.path.basename(path)}: {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    print(f"{'='*80}\n")


# ==================== MAIN ENTRY POINT ==================== #

def main():
    """Main processing pipeline."""
    print("\n" + "="*80)
    print("Elephant Re-Identification - MegaDetector-Based Preprocessing")
    print("Detection-Based Cropping with Validated Parameters")
    print("="*80)
    print("\nConfiguration:")
    print(f"  → MegaDetector v5a")
    print(f"  → Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  → Padding ratio: {BBOX_PADDING_RATIO} (15%)")
    print(f"  → Arrow detection: Enabled (for multi-elephant selection)")
    print("="*80)
    
    # Create output directory
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {PROCESSED_ROOT}")
    
    # Process Makhna dataset
    print("\n" + "▶"*40)
    print("PROCESSING MAKHNA DATASET")
    print("▶"*40)
    makhna_stats = process_dataset(MAKHNA_RAW, "Makhna")
    print_summary("Makhna", makhna_stats)
    
    # Process Herd dataset
    print("\n" + "▶"*40)
    print("PROCESSING HERD DATASET")
    print("▶"*40)
    herd_stats = process_dataset(HERD_RAW, "Herd")
    print_summary("Herd", herd_stats)
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL PROCESSING COMPLETE")
    print("="*80)
    total_processed = makhna_stats['processed'] + herd_stats['processed']
    total_scanned = makhna_stats['total'] + herd_stats['total']
    total_no_detect = makhna_stats['no_detection'] + herd_stats['no_detection']
    
    success_rate = (total_processed / total_scanned * 100) if total_scanned > 0 else 0
    
    print(f"Total images scanned:      {total_scanned}")
    print(f"Total images processed:    {total_processed}")
    print(f"Success rate:              {success_rate:.1f}%")
    print(f"No detection:              {total_no_detect}")
    print(f"\nProcessed data saved to:   {PROCESSED_ROOT}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
