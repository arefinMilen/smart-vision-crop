import os
import csv
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_previews():
    preview_dir = os.path.join(BASE_DIR, "preview_outputs")
    os.makedirs(preview_dir, exist_ok=True)
    
    # 1. Preview Task 1 Head-Cut Predictions
    t1_img_dir = os.path.join(BASE_DIR, "Task_1_HeadCut_Detection", "Task_1_HeadCut_Detection", "images", "dev")
    t1_csv = os.path.join(BASE_DIR, "Candidate_Submission", "Task1_Solution", "predictions.csv")
    
    if os.path.exists(t1_csv):
        with open(t1_csv, 'r') as f:
            reader = list(csv.DictReader(f))[:5] # First 5 samples
            for row in reader:
                fname = row['filename']
                pred = int(row['head_cut'])
                img_path = os.path.join(t1_img_dir, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    color = (0, 0, 255) if pred == 1 else (0, 255, 0)
                    text = "HEAD CUT DETECTED (1)" if pred == 1 else "HEAD OK (0)"
                    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
                    cv2.putText(img, f"{fname}: {text}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imwrite(os.path.join(preview_dir, f"t1_preview_{fname}"), img)
                    
    # 2. Preview Task 2 Crop Bounding Boxes
    t2_img_dir = os.path.join(BASE_DIR, "Task_2_Image_Cropping", "Task_2_Image_Cropping", "images", "dev")
    t2_csv = os.path.join(BASE_DIR, "Candidate_Submission", "Task2_Solution", "boxes.csv")
    
    if os.path.exists(t2_csv):
        with open(t2_csv, 'r') as f:
            reader = list(csv.DictReader(f))[:5] # First 5 samples
            for row in reader:
                fname = row['filename']
                x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
                img_path = os.path.join(t2_img_dir, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    # Draw Crop Window Box in Cyan
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 4)
                    # Label Aspect Ratio 2:3
                    cv2.putText(img, f"2:3 Crop ({w}x{h})", (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.imwrite(os.path.join(preview_dir, f"t2_crop_preview_{fname}"), img)
                    
                    # Also save actual cropped image
                    crop_img = img[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(preview_dir, f"t2_cropped_{fname}"), crop_img)

    print(f"Generated visual preview images inside folder: {preview_dir}")

if __name__ == '__main__':
    create_previews()
