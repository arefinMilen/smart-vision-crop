import os
import cv2
import numpy as np
import csv
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(BASE_DIR, "Task_1_HeadCut_Detection", "Task_1_HeadCut_Detection")

def detect_head_cut_rule_pure(image_path):
    """
    Pure Rule-Based Head Cut Detector:
    Checks top boundary edge density, skin-tone ratio, hair-tone ratio,
    and foreground contour intersection with y=0.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Top row / top boundary analysis (top 2.5% of height)
    top_margin = max(4, int(h * 0.025))
    
    # Skin tone detection in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    top_skin = skin_mask[0:top_margin, :]
    top_skin_density = np.mean(top_skin > 0)
    
    # Edge detection at the top border
    edges = cv2.Canny(gray, 40, 120)
    top_edges = edges[0:top_margin, :]
    top_edge_density = np.mean(top_edges > 0)
    
    # Check dark hair / dark top boundary touching upper edge in center 60% of width
    center_start = int(w * 0.2)
    center_end = int(w * 0.8)
    top_center_gray = gray[0:top_margin, center_start:center_end]
    
    # Background color estimation (using top corners)
    top_left_bg = np.mean(gray[0:top_margin, 0:int(w*0.15)])
    top_right_bg = np.mean(gray[0:top_margin, int(w*0.85):w])
    bg_brightness = (top_left_bg + top_right_bg) / 2.0
    
    # Foreground diff at top center
    fg_diff = np.abs(top_center_gray.astype(float) - bg_brightness)
    top_fg_density = np.mean(fg_diff > 25)
    
    # Rule Heuristic:
    # If there is skin on the very top edge OR high foreground density touching y=0 in center
    if top_skin_density > 0.04:
        return 1
    if top_fg_density > 0.35 and top_edge_density > 0.08:
        return 1
    if top_skin_density > 0.015 and top_fg_density > 0.20:
        return 1
        
    return 0

def evaluate():
    for split in ['train', 'dev']:
        csv_path = os.path.join(TASK1_DIR, f'labels_{split}.csv')
        img_dir = os.path.join(TASK1_DIR, 'images', split)
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            correct = 0
            tp, fp, fn, tn = 0, 0, 0, 0
            total = 0
            
            for row in reader:
                fname = row['filename']
                target = int(row['head_cut'])
                img_path = os.path.join(img_dir, fname)
                
                pred = detect_head_cut_rule_pure(img_path)
                total += 1
                if pred == target:
                    correct += 1
                    
                if pred == 1 and target == 1:
                    tp += 1
                elif pred == 1 and target == 0:
                    fp += 1
                elif pred == 0 and target == 1:
                    fn += 1
                elif pred == 0 and target == 0:
                    tn += 1
                    
            acc = correct / total
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            print(f"=== Pure Rule-Based Split: {split.upper()} ===")
            print(f"Total: {total}, Correct: {correct}, Accuracy: {acc*100:.2f}%")
            print(f"TP: {tp}, FP (False Alarm): {fp}, FN (Missed): {fn}, TN: {tn}")
            print(f"Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%\n")

if __name__ == '__main__':
    evaluate()
