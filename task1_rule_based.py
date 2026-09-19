import os
import cv2
import numpy as np
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(BASE_DIR, "Task_1_HeadCut_Detection", "Task_1_HeadCut_Detection")

# Load OpenCV Cascade Classifier for Face Profile and Frontal Face
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

def detect_head_cut_rule(image_path):
    """
    Rule-based detection logic:
    1. Check top 5% edge boundary for skin-tone pixels and top-cluttered edges touching top boundary.
    2. Detect face bounding box location.
       - If a face is detected very close to the top border (y_min < 0.05 * height) and truncated, flag head_cut = 1.
    3. Analyze upper boundary skin/hair region continuity.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    all_faces = list(faces) + list(profiles)
    
    # Signal 1: Face touches or is extremely close to top edge
    face_cut_flag = False
    if len(all_faces) > 0:
        for (x, y, fw, fh) in all_faces:
            # If top of face box is in top 4% of image height
            if y <= int(0.05 * h):
                face_cut_flag = True
                break
                
    # Signal 2: Top edge color & edge profile analysis (Skin/Hair touching y=0)
    # Convert to HSV for skin color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define skin color ranges in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Top 3% band
    top_band_h = max(3, int(h * 0.03))
    top_skin = skin_mask[0:top_band_h, :]
    top_skin_ratio = np.sum(top_skin > 0) / (top_band_h * w)
    
    # Canny edge on top row
    edges = cv2.Canny(gray, 50, 150)
    top_edges = edges[0:top_band_h, :]
    top_edge_ratio = np.sum(top_edges > 0) / (top_band_h * w)
    
    # Decision heuristic combination
    if face_cut_flag:
        return 1
    if top_skin_ratio > 0.08:
        return 1
    if top_edge_ratio > 0.12 and top_skin_ratio > 0.03:
        return 1
        
    return 0

def evaluate_rule_based():
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
                
                pred = detect_head_cut_rule(img_path)
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
            
            print(f"=== Rule-Based Split: {split.upper()} ===")
            print(f"Total: {total}, Correct: {correct}, Accuracy: {acc*100:.2f}%")
            print(f"TP: {tp}, FP (False Alarm): {fp}, FN (Missed): {fn}, TN: {tn}")
            print(f"Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%\n")

if __name__ == '__main__':
    evaluate_rule_based()
