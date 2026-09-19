import os
import sys
import argparse
import csv
import cv2
import numpy as np
from PIL import Image

face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml') if hasattr(cv2, 'data') else ""
face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None

def detect_dynamic_person_box(img_cv):
    """
    Detects dynamic person bounding box [px, py, pw, ph] using contour analysis and skin/face detection.
    """
    if img_cv is None:
        return None
        
    h, w, _ = img_cv.shape
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. Background subtraction via Otsu thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > (w * h * 0.05): # At least 5% of image area
            x, y, cw, ch = cv2.boundingRect(c)
            # Check horizontal centering
            cx = x + cw / 2.0
            if int(w * 0.15) <= cx <= int(w * 0.85):
                valid_boxes.append((x, y, cw, ch, area))
                
    if len(valid_boxes) > 0:
        # Pick largest central contour
        valid_boxes.sort(key=lambda b: b[4], reverse=True)
        bx, by, bw, bh, _ = valid_boxes[0]
        
        # Check face detection to refine top head boundary
        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces) > 0:
                face_y = min(fy for (fx, fy, fw, fh) in faces)
                by = min(by, face_y)
                
        return [bx, by, bw, bh]
        
    return None

def compute_smart_crop(img_w, img_h, garment_type, curr_box=None, img_cv=None):
    """
    Computes exact 2:3 aspect ratio crop box [x, y, w, h] adhering to standard framing rules.
    """
    if curr_box is not None and len(curr_box) == 4:
        px, py, pw, ph = curr_box
    else:
        dynamic_box = detect_dynamic_person_box(img_cv)
        if dynamic_box is not None:
            px, py, pw, ph = dynamic_box
        else:
            px, py, pw, ph = int(img_w * 0.1), int(img_h * 0.05), int(img_w * 0.8), int(img_h * 0.9)
        
    x_center = px + pw / 2.0
    
    if garment_type in ['TOPS', 'OUTER']:
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.54 * ph)
    elif garment_type == 'BOTTOMS':
        y_top = max(0.0, py + 0.33 * ph)
        y_bottom = min(float(img_h), py + 0.98 * ph)
    elif garment_type == 'DRESS':
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.82 * ph)
    elif garment_type == 'SET':
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.92 * ph)
    else:
        y_top = max(0.0, py - 0.04 * ph)
        y_bottom = min(float(img_h), py + 0.85 * ph)
        
    target_h = y_bottom - y_top
    if target_h <= 50:
        target_h = ph * 0.6
        
    target_w = target_h * (2.0 / 3.0)
    
    if target_w > img_w:
        target_w = float(img_w)
        target_h = target_w * (3.0 / 2.0)
        
    if y_top + target_h > img_h:
        if garment_type == 'BOTTOMS':
            y_top = max(0.0, img_h - target_h)
        else:
            y_bottom = float(img_h)
            y_top = max(0.0, y_bottom - target_h)
            
    x_top_left = x_center - (target_w / 2.0)
    
    if x_top_left < 0:
        x_top_left = 0.0
    elif x_top_left + target_w > img_w:
        x_top_left = img_w - target_w
        
    w_final = int(round(target_w))
    h_final = int(round(w_final * 1.5))
    
    x_final = max(0, min(img_w - w_final, int(round(x_top_left))))
    y_final = max(0, min(img_h - h_final, int(round(y_top))))
    
    return [x_final, y_final, w_final, h_final]

def main():
    parser = argparse.ArgumentParser(description="Crop product photos to 2:3 aspect ratio framing.")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("--meta", type=str, required=False, help="Path to meta CSV file (filename, garment_type, image_width, image_height)")
    parser.add_argument("--out", type=str, default="boxes.csv", help="Output bounding boxes CSV path")
    args = parser.parse_args()

    image_dir = args.image_dir
    meta_csv = args.meta
    out_csv = args.out

    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        sys.exit(1)

    meta_data = {}
    if meta_csv and os.path.exists(meta_csv):
        with open(meta_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_data[row['filename']] = {
                    'garment_type': row['garment_type'],
                    'width': int(row['image_width']),
                    'height': int(row['image_height'])
                }

    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])

    results = []
    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        img_cv = cv2.imread(img_path)
        
        if fname in meta_data:
            gtype = meta_data[fname]['garment_type']
            img_w = meta_data[fname]['width']
            img_h = meta_data[fname]['height']
        else:
            gtype = 'TOPS'
            if img_cv is not None:
                img_h, img_w, _ = img_cv.shape
            else:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size

        box = compute_smart_crop(img_w, img_h, gtype, img_cv=img_cv)
        results.append((fname, box[0], box[1], box[2], box[3]))

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x", "y", "w", "h"])
        for row in results:
            writer.writerow(row)

    print(f"Successfully cropped {len(results)} images. Boxes saved to '{out_csv}'.")

if __name__ == "__main__":
    main()
