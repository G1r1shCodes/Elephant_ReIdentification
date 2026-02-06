"""
Elephant Re-Identification — Phase B: Biologically-Aware Data Engineering
Preprocessing Script: LARGE CONTEXTUAL CROPS with Biological Anchoring

Wildlife Institute of India Research Project
OPEN-SET BIOMETRIC RE-IDENTIFICATION SYSTEM

================================================================================
CRITICAL DESIGN PRINCIPLE:
The arrow is an IDENTITY SELECTOR, NOT a spatial localization signal.
Arrow position does NOT correspond to biometric regions.
Cropping must be BIOLOGICALLY BIASED toward head/ears/temporal gland.
================================================================================

Biological Constraints:
- Identity features: Head profile, ear shape/depigmentation, temporal gland
- Flank-only or leg-only crops are biometrically USELESS
- Head and at least one ear MUST be preserved
- Makhna temporal gland region is CRITICAL

Methodology:
- NO object detection, NO bounding boxes, NO deep learning
- Deterministic classical CV only
- Large contextual crops (60-75% of image dimensions)
- Upward + Forward bias from arrow anchor
- Never modify raw data
"""

import cv2
import numpy as np
import os


# ==================== CONFIGURATION ==================== #

# Root paths
DATA_ROOT = "../../data"
PROCESSED_ROOT = os.path.join(DATA_ROOT, "processed")

# Raw dataset paths
MAKHNA_RAW = os.path.join(DATA_ROOT, "raw", "Makhna_id_udalguri_24", "Makhna_id_udalguri_24")
HERD_RAW = os.path.join(DATA_ROOT, "raw", "Herd_ID_Udalguri_24", "Herd_ID_Udalguri_24")

# Arrow detection parameters (for 4K images ~4608×3456)
MIN_ARROW_AREA = 4000  # Minimum pixels for valid arrow

# HSV strict red ranges (digital painted arrows)
# Range 1: Low hue reds (0-10°)
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
# Range 2: High hue reds (170-180°) - wrap-around
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

# ====================================================================================
# BIOLOGICAL CROP PARAMETERS
# ====================================================================================
# CRITICAL: These ratios are designed to preserve identity-bearing anatomy
#
# CASE A: Arrow Present (Multi-Elephant Scene)
# - Arrow indicates WHICH elephant, not WHERE to crop tightly
# - Must preserve head/ears/temporal gland regardless of arrow position
# - Use arrow as loose anchor, then apply BIOLOGICAL BIAS

ARROW_CROP_WIDTH_RATIO = 0.65   # 65% of image width (large contextual)
ARROW_CROP_HEIGHT_RATIO = 0.75  # 75% of image height (generous vertical)

# BIOLOGICAL OFFSET from arrow tip:
# - UPWARD BIAS: Elephants' heads are typically ABOVE or LEVEL with arrow
# - FORWARD BIAS: Prioritize front anatomy over rear
ARROW_UPWARD_BIAS = -0.15        # Move crop center 15% UPWARD from arrow tip
ARROW_FORWARD_BIAS_X = -0.10     # Move crop center 10% FORWARD (left if facing right)

# CASE B: No Arrow (Single Elephant)
# - Gentle trimming, preserve as much as possible
# - Avoid aggressive center cropping
NO_ARROW_CROP_WIDTH_RATIO = 0.85   # 85% of image width (minimal trim)
NO_ARROW_CROP_HEIGHT_RATIO = 0.85  # 85% of image height (minimal trim)
# ====================================================================================

# Supported image formats
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


# ==================== ARROW DETECTION ==================== #

def detect_arrow_tip(image):
    """
    Detect red arrow and return tip coordinates as IDENTITY ANCHOR.
    
    The arrow is NOT a bounding box. It indicates which elephant in
    multi-elephant scenes, NOT where to crop tightly.
    
    Algorithm:
    1. Convert to HSV color space
    2. Apply strict red masks (handles hue wrap-around)
    3. Find largest red contour (assume it's the arrow)
    4. Validate minimum area (rejects noise)
    5. Compute convex hull and centroid
    6. Find farthest point from centroid = arrow tip
    
    Args:
        image: BGR image (numpy array)
        
    Returns:
        tuple (x, y): Arrow tip coordinates if valid arrow found
        None: If no arrow or invalid arrow
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for both red ranges (handles hue wraparound at 180°)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)  # 0-10°
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)  # 170-180°
    
    # Combine masks (bitwise OR)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None  # No red regions found
    
    # Get largest contour (assume it's the arrow, not noise)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Validate minimum area (reject noise)
    area = cv2.contourArea(largest_contour)
    if area < MIN_ARROW_AREA:
        return None  # Too small, likely noise
    
    # Get convex hull (smooths contour)
    hull = cv2.convexHull(largest_contour)
    
    # Calculate centroid of hull
    moments = cv2.moments(hull)
    if moments["m00"] == 0:
        return None  # Degenerate contour
    
    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]
    centroid = np.array([centroid_x, centroid_y])
    
    # Find point farthest from centroid = arrow tip
    hull_points = hull.reshape(-1, 2)
    distances = np.linalg.norm(hull_points - centroid, axis=1)
    tip_idx = np.argmax(distances)
    tip_point = hull_points[tip_idx]
    
    return int(tip_point[0]), int(tip_point[1])


# ==================== BIOLOGICALLY-AWARE CROPPING ==================== #

def safe_crop_with_boundary_handling(image, center_x, center_y, crop_width, crop_height):
    """
    Extract crop with safe boundary clamping.
    
    If requested crop extends beyond image boundaries, clamps to valid region.
    Ensures no out-of-bounds access.
    
    Args:
        image: Input BGR image
        center_x, center_y: Desired crop center
        crop_width, crop_height: Desired crop dimensions
        
    Returns:
        Cropped image (may be smaller than requested if near boundaries)
    """
    img_height, img_width = image.shape[:2]
    
    # Calculate crop boundaries
    half_w = crop_width // 2
    half_h = crop_height // 2
    
    x1 = center_x - half_w
    y1 = center_y - half_h
    x2 = center_x + half_w
    y2 = center_y + half_h
    
    # Clamp to image boundaries
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(img_width, x2)
    y2_clamped = min(img_height, y2)
    
    # Extract crop
    cropped = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    
    return cropped


def extract_biologically_anchored_crop(image, arrow_tip=None):
    """
    Extract LARGE CONTEXTUAL CROP with BIOLOGICAL BIAS.
    
    ============================================================================
    CRITICAL LOGIC:
    ============================================================================
    
    CASE A: Arrow Present (Multi-Elephant)
    ---------
    - Arrow tip is used as LOOSE ANCHOR, NOT crop target
    - Apply UPWARD BIAS: Elephant head is typically above/level with arrow
    - Apply FORWARD BIAS: Prioritize front anatomy (head/ears/temporal gland)
    - Large crop (65% × 75%) preserves context
    - NEVER crop narrowly in arrow direction (causes head loss!)
    
    Biological Rationale:
    - Arrows often point to back/flank (easy to mark)
    - But identity features are in HEAD region (ears, temporal gland, profile)
    - Must bias crop UPWARD and FORWARD from arrow anchor
    - Large crop ensures head is included even if arrow is on rear
    
    CASE B: No Arrow (Single Elephant)
    ----------
    - Assume single elephant, likely well-centered
    - Gentle trimming (85% × 85%) to remove margins
    - Avoid aggressive center cropping
    - Preserve as much biometric context as possible
    
    ============================================================================
    
    Args:
        image: Input BGR image
        arrow_tip: tuple (x, y) or None
        
    Returns:
        Cropped image preserving identity-bearing anatomy
    """
    img_height, img_width = image.shape[:2]
    
    if arrow_tip is not None:
        # ====================================================================
        # CASE A: Arrow Present - Use BIOLOGICALLY-BIASED CONTEXTUAL CROP
        # ====================================================================
        
        crop_width = int(img_width * ARROW_CROP_WIDTH_RATIO)
        crop_height = int(img_height * ARROW_CROP_HEIGHT_RATIO)
        
        # CRITICAL: Apply BIOLOGICAL BIAS from arrow anchor
        # - UPWARD BIAS: Head is typically above arrow position
        # - FORWARD BIAS: Prioritize front anatomy
        
        # X-axis: Apply forward bias (shift left if facing right, assume left-facing)
        # In practice, this slight shift helps capture more of the front
        center_x = int(arrow_tip[0] + (ARROW_FORWARD_BIAS_X * img_width))
        
        # Y-axis: Apply UPWARD bias (negative = move up)
        # This is CRITICAL - elephants' heads are above/level with arrows
        center_y = int(arrow_tip[1] + (ARROW_UPWARD_BIAS * img_height))
        
        # Ensure crop center stays within reasonable bounds
        # (safe_crop_with_boundary_handling will clamp final boundaries)
        center_x = max(crop_width // 2, min(img_width - crop_width // 2, center_x))
        center_y = max(crop_height // 2, min(img_height - crop_height // 2, center_y))
        
    else:
        # ====================================================================
        # CASE B: No Arrow - Gentle Center Crop (Single Elephant)
        # ====================================================================
        
        crop_width = int(img_width * NO_ARROW_CROP_WIDTH_RATIO)
        crop_height = int(img_height * NO_ARROW_CROP_HEIGHT_RATIO)
        
        # Center crop (elephant likely well-centered in single-elephant images)
        center_x = img_width // 2
        center_y = img_height // 2
    
    return safe_crop_with_boundary_handling(image, center_x, center_y, crop_width, crop_height)


# ==================== QUALITY CHECKS ==================== #

def check_crop_quality(original_image, cropped_image, arrow_tip):
    """
    Perform basic quality checks on cropped image.
    
    Checks:
    - Crop is not too small (< 20% of original)
    - Crop has reasonable aspect ratio
    
    Args:
        original_image: Original input image
        cropped_image: Cropped output image
        arrow_tip: Arrow tip coordinates or None
        
    Returns:
        tuple (is_valid, warning_message)
    """
    orig_h, orig_w = original_image.shape[:2]
    crop_h, crop_w = cropped_image.shape[:2]
    
    # Check if crop is too small
    crop_area_ratio = (crop_w * crop_h) / (orig_w * orig_h)
    if crop_area_ratio < 0.20:  # Less than 20% of original
        return False, f"Crop too small ({crop_area_ratio:.1%} of original)"
    
    # Check aspect ratio (elephant images should be roughly landscape)
    aspect_ratio = crop_w / crop_h
    if aspect_ratio < 0.5 or aspect_ratio > 3.0:
        return False, f"Unusual aspect ratio ({aspect_ratio:.2f})"
    
    return True, None


# ==================== PROCESSING PIPELINE ==================== #

def process_dataset(input_root, output_base_name):
    """
    Process all images in dataset with biologically-aware cropping.
    
    For each valid image:
    1. Detect arrow (if present) - serves as identity anchor only
    2. Extract LARGE CONTEXTUAL CROP with biological bias
    3. Perform quality checks
    4. Save to processed directory maintaining structure
    
    Args:
        input_root: Raw dataset directory path
        output_base_name: Name for output folder ("Makhna" or "Herd")
        
    Returns:
        Dictionary with detailed processing statistics
    """
    stats = {
        'total': 0,
        'processed': 0,
        'skipped': 0,
        'arrow_detected': 0,
        'no_arrow': 0,
        'quality_warnings': 0,
        'errors': []
    }
    
    print(f"\n{'='*80}")
    print(f"Processing: {input_root}")
    print(f"Output to:  {os.path.join(PROCESSED_ROOT, output_base_name)}")
    print(f"{'='*80}\n")
    
    # Walk directory tree recursively
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            # Get full input path
            input_path = os.path.join(root, filename)
            
            # Check if valid image format
            _, ext = os.path.splitext(filename)
            if ext not in VALID_EXTENSIONS:
                continue  # Skip non-image files
            
            stats['total'] += 1
            
            try:
                # Read image
                image = cv2.imread(input_path)
                if image is None:
                    stats['skipped'] += 1
                    stats['errors'].append((input_path, "Failed to read image"))
                    print(f"[SKIP] {input_path} - Failed to read")
                    continue
                
                # Detect arrow (serves as identity anchor, NOT crop target)
                arrow_tip = detect_arrow_tip(image)
                
                # Extract biologically-anchored crop
                cropped_image = extract_biologically_anchored_crop(image, arrow_tip)
                
                # Quality check
                is_valid, warning = check_crop_quality(image, cropped_image, arrow_tip)
                
                if not is_valid:
                    stats['quality_warnings'] += 1
                    print(f"[WARN] {input_path} - {warning}")
                    # Still process, but flag for manual review
                
                # Update statistics
                if arrow_tip is not None:
                    stats['arrow_detected'] += 1
                    status = "ARROW"
                else:
                    stats['no_arrow'] += 1
                    status = "NO_ARROW"
                
                # Construct output path maintaining directory structure
                relative_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(PROCESSED_ROOT, output_base_name, relative_path)
                
                # Ensure output is .jpg
                output_path = os.path.splitext(output_path)[0] + ".jpg"
                
                # Create output directory
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save processed image
                cv2.imwrite(output_path, cropped_image)
                
                stats['processed'] += 1
                print(f"[OK - {status:9s}] {os.path.basename(input_path):40s} → {cropped_image.shape[1]}x{cropped_image.shape[0]}")
                
            except Exception as e:
                stats['skipped'] += 1
                stats['errors'].append((input_path, str(e)))
                print(f"[ERROR] {input_path} - {e}")
    
    return stats


def print_summary(dataset_name, stats):
    """Print detailed processing summary."""
    print(f"\n{'='*80}")
    print(f"{dataset_name} Dataset - Processing Summary")
    print(f"{'='*80}")
    print(f"Total files scanned:       {stats['total']}")
    print(f"Successfully processed:    {stats['processed']}")
    print(f"Skipped/Failed:            {stats['skipped']}")
    print(f"Quality warnings:          {stats['quality_warnings']}")
    print(f"\nCrop Strategy:")
    print(f"  - With arrow detected:   {stats['arrow_detected']} (biologically-biased)")
    print(f"  - No arrow (center):     {stats['no_arrow']} (gentle trim)")
    
    if stats['errors']:
        print(f"\n⚠ Errors encountered: {len(stats['errors'])}")
        for path, error in stats['errors'][:10]:  # Show first 10
            print(f"  - {os.path.basename(path)}: {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    print(f"{'='*80}\n")


# ==================== MAIN ENTRY POINT ==================== #

def main():
    """Main processing pipeline entry point."""
    print("\n" + "="*80)
    print("Elephant Re-Identification - Phase B: Biologically-Aware Data Engineering")
    print("LARGE CONTEXTUAL CROPS with Biological Anchoring")
    print("="*80)
    print("\nDesign Principle:")
    print("  → Arrow = Identity Selector (NOT spatial localization)")
    print("  → Biological Bias: Upward + Forward from arrow anchor")
    print("  → Preserve head/ears/temporal gland (identity features)")
    print("  → Large crops (60-75%) prevent identity loss")
    print("="*80)
    
    # Create processed data root directory
    os.makedirs(PROCESSED_ROOT, exist_ok=True)
    print(f"\nOutput directory: {PROCESSED_ROOT}")
    
    # Process Makhna dataset
    print("\n" + "▶"*40)
    print("PROCESSING MAKHNA DATASET (Adult males - tuskless)")
    print("▶"*40)
    makhna_stats = process_dataset(MAKHNA_RAW, "Makhna")
    print_summary("Makhna", makhna_stats)
    
    # Process Herd dataset
    print("\n" + "▶"*40)
    print("PROCESSING HERD DATASET (Females, juveniles, calves)")
    print("▶"*40)
    herd_stats = process_dataset(HERD_RAW, "Herd")
    print_summary("Herd", herd_stats)
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL PROCESSING COMPLETE")
    print("="*80)
    total_processed = makhna_stats['processed'] + herd_stats['processed']
    total_scanned = makhna_stats['total'] + herd_stats['total']
    total_arrow = makhna_stats['arrow_detected'] + herd_stats['arrow_detected']
    total_no_arrow = makhna_stats['no_arrow'] + herd_stats['no_arrow']
    total_warnings = makhna_stats['quality_warnings'] + herd_stats['quality_warnings']
    
    print(f"Total images scanned:      {total_scanned}")
    print(f"Total images processed:    {total_processed}")
    print(f"Total with arrows:         {total_arrow}")
    print(f"Total without arrows:      {total_no_arrow}")
    print(f"Quality warnings:          {total_warnings}")
    print(f"\nProcessed data saved to:   {PROCESSED_ROOT}")
    print("="*80 + "\n")
    
    # Research note
    print("📌 Research Note:")
    print("   - Review quality-warned images manually")
    print("   - Verify head/ear preservation in arrow-guided crops")
    print("   - Check that temporal gland regions are visible (Makhnas)")
    print()


if __name__ == "__main__":
    main()
