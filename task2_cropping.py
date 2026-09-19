import os
import csv
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK2_DIR = os.path.join(BASE_DIR, "Task_2_Image_Cropping", "Task_2_Image_Cropping")

def compute_smart_crop(img_w, img_h, garment_type, curr_box=None):
    """
    Computes exact 2:3 aspect ratio crop box [x, y, w, h] adhering to standard framing rules.
    """
    if curr_box is not None and len(curr_box) == 4:
        px, py, pw, ph = curr_box
    else:
        # Fallback person box estimate
        px, py, pw, ph = int(img_w * 0.1), int(img_h * 0.05), int(img_w * 0.8), int(img_h * 0.9)
        
    x_center = px + pw / 2.0
    
    # Category specific top and bottom target ratios relative to person box
    if garment_type in ['TOPS', 'OUTER']:
        # Keep head with 4% margin, cut hem around mid-thigh / waist (52% of person height)
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.54 * ph)
    elif garment_type == 'BOTTOMS':
        # Start at or above waistband (35% down person box), end at feet/hem
        y_top = max(0.0, py + 0.33 * ph)
        y_bottom = min(float(img_h), py + 0.98 * ph)
    elif garment_type == 'DRESS':
        # Keep head, end below dress hem (80% of person height)
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.82 * ph)
    elif garment_type == 'SET':
        # Keep head, end below set lower piece hem (90% of person height)
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.92 * ph)
    else:
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.85 * ph)
        
    target_h = y_bottom - y_top
    if target_h <= 50:
        target_h = ph * 0.6
        
    target_w = target_h * (2.0 / 3.0)
    
    # Check width boundary constraint
    if target_w > img_w:
        target_w = float(img_w)
        target_h = target_w * (3.0 / 2.0)
        
    # Check height boundary constraint
    if y_top + target_h > img_h:
        if garment_type == 'BOTTOMS':
            # Extend upwards for BOTTOMS
            y_top = max(0.0, img_h - target_h)
        else:
            # Extend downwards for TOPS/DRESS/SET
            y_bottom = float(img_h)
            y_top = max(0.0, y_bottom - target_h)
            
    x_top_left = x_center - (target_w / 2.0)
    
    # Horizontal boundary clamping
    if x_top_left < 0:
        x_top_left = 0.0
    elif x_top_left + target_w > img_w:
        x_top_left = img_w - target_w
        
    # Convert to exact integer coordinates
    w_final = int(round(target_w))
    h_final = int(round(w_final * 1.5)) # Guarantees exact 2:3 ratio
    
    x_final = max(0, min(img_w - w_final, int(round(x_top_left))))
    y_final = max(0, min(img_h - h_final, int(round(y_top))))
    
    return [x_final, y_final, w_final, h_final]

def evaluate_task2():
    for split in ['train', 'dev']:
        meta_csv = os.path.join(TASK2_DIR, f'meta_{split}.csv')
        curr_csv = os.path.join(TASK2_DIR, f'current_system_{split}.csv')
        
        curr_boxes = {}
        if os.path.exists(curr_csv):
            with open(curr_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    curr_boxes[row['filename']] = [float(row['x']), float(row['y']), float(row['w']), float(row['h'])]
                    
        valid_count = 0
        total_count = 0
        
        with open(meta_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row['filename']
                gtype = row['garment_type']
                w_img = int(row['image_width'])
                h_img = int(row['image_height'])
                
                c_box = curr_boxes.get(fname, None)
                box = compute_smart_crop(w_img, h_img, gtype, c_box)
                x, y, w, h = box
                
                total_count += 1
                
                # Check validity:
                # 1. 2:3 ratio within 1%
                ratio = h / float(w)
                aspect_valid = abs(ratio - 1.5) < 0.015
                
                # 2. Window inside image
                bounds_valid = (x >= 0 and y >= 0 and (x + w) <= w_img and (y + h) <= h_img)
                
                if aspect_valid and bounds_valid:
                    valid_count += 1
                else:
                    print(f"Invalid crop for {fname}: ratio={ratio:.4f}, bounds={bounds_valid}")
                    
        print(f"=== Task 2 Cropping Validity Split: {split.upper()} ===")
        print(f"Total Images: {total_count}, Valid 2:3 Crops: {valid_count} ({valid_count/total_count*100:.2f}%)\n")

if __name__ == '__main__':
    evaluate_task2()
