import os
import csv
import cv2
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(BASE_DIR, "Task_1_HeadCut_Detection", "Task_1_HeadCut_Detection")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "Candidate_Submission", "Task1_Solution")
MODEL_PATH = os.path.join(MODEL_OUT_DIR, "task1_model.joblib")

face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml') if hasattr(cv2, 'data') else ""
face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None

def extract_advanced_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(48)
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 30, 110)
    
    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    features = []
    
    percentages = [0.003, 0.008, 0.012, 0.018, 0.025, 0.04, 0.06, 0.10, 0.14]
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
        
    row0 = gray[0:max(3, int(h*0.015)), :]
    bg_est = (np.mean(gray[0:10, 0:int(w*0.15)]) + np.mean(gray[0:10, int(w*0.85):w])) / 2.0
    bg_diff_row0 = np.mean(np.abs(row0.astype(float) - bg_est) < 15)
    features.append(bg_diff_row0)
    
    top_hair_region = gray[0:max(4, int(h*0.025)), int(w*0.25):int(w*0.75)]
    features.append(np.mean(top_hair_region < 40))
    features.append(np.mean((top_hair_region >= 40) & (top_hair_region < 110)))
    features.append(np.mean(top_hair_region >= 180))
    features.append(np.std(top_hair_region))
    
    g_kernel = cv2.getGaborKernel((9, 9), 3.0, np.pi/4, 8.0, 0.5, 0, ktype=cv2.CV_32F)
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

def load_dataset(split):
    csv_path = os.path.join(TASK1_DIR, f'labels_{split}.csv')
    img_dir = os.path.join(TASK1_DIR, 'images', split)
    
    X, y, filenames = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']
            target = int(row['head_cut'])
            feat = extract_advanced_features(os.path.join(img_dir, fname))
            X.append(feat)
            y.append(target)
            filenames.append(fname)
            
    return np.array(X), np.array(y), filenames

def train_and_eval():
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    
    print("Extracting features for 90% accuracy model...")
    X_train, y_train, _ = load_dataset('train')
    X_dev, y_dev, _ = load_dataset('dev')
    
    clf1 = ExtraTreesClassifier(n_estimators=250, max_depth=9, random_state=42, class_weight='balanced')
    clf2 = GradientBoostingClassifier(n_estimators=140, learning_rate=0.06, max_depth=4, random_state=42)
    clf3 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
    
    ensemble = VotingClassifier(
        estimators=[('et', clf1), ('gb', clf2), ('rf', clf3)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    dev_probs = ensemble.predict_proba(X_dev)[:, 1]
    
    best_thresh = 0.50
    best_dev_acc = 0.0
    
    for thresh in np.arange(0.40, 0.75, 0.01):
        preds = (dev_probs >= thresh).astype(int)
        acc = accuracy_score(y_dev, preds)
        rec = recall_score(y_dev, preds, zero_division=0)
        if rec >= 0.95 and acc > best_dev_acc:
            best_dev_acc = acc
            best_thresh = thresh

    dev_preds = (dev_probs >= best_thresh).astype(int)
    dev_acc = accuracy_score(y_dev, dev_preds)
    dev_prec = precision_score(y_dev, dev_preds, zero_division=0)
    dev_rec = recall_score(y_dev, dev_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_dev, dev_preds).ravel()
    
    print(f"\n=== FINAL OPTIMIZED 90% DEV METRICS (Thresh={best_thresh:.2f}) ===")
    print(f"Accuracy: {dev_acc*100:.2f}%")
    print(f"Precision: {dev_prec*100:.2f}%, Recall: {dev_rec*100:.2f}%")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model_data = {
        'model': ensemble,
        'threshold': best_thresh
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved successfully to: {MODEL_PATH}")

if __name__ == '__main__':
    train_and_eval()
