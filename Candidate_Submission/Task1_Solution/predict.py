import os
import sys
import argparse
import csv
import cv2
import numpy as np

face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml') if hasattr(cv2, 'data') else ""
face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None

def extract_deep_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(56)
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 30, 110)
    
    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    features = []
    
    percentages = [0.001, 0.003, 0.006, 0.010, 0.015, 0.022, 0.035, 0.05, 0.08, 0.12, 0.16]
    for p in percentages:
        slice_h = max(2, int(h * p))
        
        skin_density = np.mean(skin_mask[0:slice_h, :] > 0)
        features.append(skin_density)
        
        edge_density = np.mean(edges[0:slice_h, :] > 0)
        features.append(edge_density)
        
        c_start = int(w * 0.25)
        c_end = int(w * 0.75)
        c_slice = gray[0:slice_h, c_start:c_end]
        bg_left = np.mean(gray[0:slice_h, 0:int(w*0.15)])
        bg_right = np.mean(gray[0:slice_h, int(w*0.85):w])
        bg_val = (bg_left + bg_right) / 2.0
        fg_diff = np.mean(np.abs(c_slice.astype(float) - bg_val) > 18)
        features.append(fg_diff)
        
        features.append(np.mean(c_slice))
        
    row0 = gray[0:max(3, int(h*0.012)), :]
    bg_est = (np.mean(gray[0:10, 0:int(w*0.15)]) + np.mean(gray[0:10, int(w*0.85):w])) / 2.0
    bg_diff_row0 = np.mean(np.abs(row0.astype(float) - bg_est) < 22)
    features.append(bg_diff_row0)
    features.append(np.std(row0))
    
    top_center_row = gray[0:max(2, int(h*0.008)), int(w*0.3):int(w*0.7)]
    features.append(np.mean(top_center_row < 55))
    features.append(np.min(top_center_row))
    
    top_hair_region = gray[0:max(4, int(h*0.025)), int(w*0.25):int(w*0.75)]
    features.append(np.mean(top_hair_region < 40))
    features.append(np.mean((top_hair_region >= 40) & (top_hair_region < 110)))
    features.append(np.mean(top_hair_region >= 180))
    features.append(np.std(top_hair_region))
    
    for angle in [0, np.pi/4, np.pi/2]:
        g_kernel = cv2.getGaborKernel((9, 9), 3.0, angle, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray[0:max(10, int(h*0.1)), :], cv2.CV_8UC3, g_kernel)
        features.append(np.mean(filtered))
        features.append(np.std(filtered))
    
    if face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            top_y = min(y for (x, y, fw, fh) in faces)
            features.append(top_y / float(h))
            features.append(1.0 if top_y <= int(0.04 * h) else 0.0)
        else:
            features.append(1.0)
            features.append(0.0)
    else:
        features.append(1.0)
        features.append(0.0)
        
    features.append(h / w)
    
    return np.array(features, dtype=float)

def check_headroom_override(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    top_slice = gray[0:max(3, int(h * 0.012)), int(w * 0.25):int(w * 0.75)]
    bg_left = np.mean(gray[0:10, 0:int(w*0.15)])
    bg_right = np.mean(gray[0:10, int(w*0.85):w])
    bg_mean = (bg_left + bg_right) / 2.0
    
    diff = np.abs(top_slice.astype(float) - bg_mean)
    match_ratio = np.mean(diff < 22)
    if match_ratio > 0.35:
        return True
    return False

def predict_rule_based(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    top_margin = max(4, int(h * 0.025))
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    top_skin = skin_mask[0:top_margin, :]
    top_skin_density = np.mean(top_skin > 0)
    
    edges = cv2.Canny(gray, 40, 120)
    top_edges = edges[0:top_margin, :]
    top_edge_density = np.mean(top_edges > 0)
    
    center_start = int(w * 0.2)
    center_end = int(w * 0.8)
    top_center_gray = gray[0:top_margin, center_start:center_end]
    
    top_left_bg = np.mean(gray[0:top_margin, 0:int(w*0.15)])
    top_right_bg = np.mean(gray[0:top_margin, int(w*0.85):w])
    bg_brightness = (top_left_bg + top_right_bg) / 2.0
    
    fg_diff = np.abs(top_center_gray.astype(float) - bg_brightness)
    top_fg_density = np.mean(fg_diff > 25)
    
    if top_skin_density > 0.04:
        return 1
    if top_fg_density > 0.35 and top_edge_density > 0.08:
        return 1
    if top_skin_density > 0.015 and top_fg_density > 0.20:
        return 1
        
    return 0

def main():
    parser = argparse.ArgumentParser(description="Predict head-cut on product images.")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--mode", type=str, choices=["rule", "model"], default="model", help="Inference mode")
    args = parser.parse_args()

    image_dir = args.image_dir
    out_csv = args.out
    mode = args.mode

    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        sys.exit(1)

    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])

    model_data = None
    if mode == "model":
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_model.joblib")
        if os.path.exists(model_path):
            try:
                import joblib
                model_data = joblib.load(model_path)
            except Exception as e:
                print(f"Warning: Failed to load model ({e}). Falling back to rule-based.")

    results = []
    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        if mode == "model" and model_data is not None:
            feat = extract_deep_features(img_path).reshape(1, -1)
            model = model_data['model'] if isinstance(model_data, dict) else model_data
            scaler = model_data.get('scaler', None) if isinstance(model_data, dict) else None
            thresh = model_data.get('threshold', 0.83) if isinstance(model_data, dict) else 0.83
            
            if scaler is not None:
                feat = scaler.transform(feat)
                
            prob = model.predict_proba(feat)[0, 1]
            pred = 1 if prob >= thresh else 0
            
            if pred == 1 and prob < 0.88 and check_headroom_override(img_path):
                pred = 0
        else:
            pred = predict_rule_based(img_path)
            
        results.append((fname, pred))

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "head_cut"])
        for fname, pred in results:
            writer.writerow([fname, pred])

    print(f"Successfully processed {len(results)} images. Predictions saved to '{out_csv}'.")

if __name__ == "__main__":
    main()
