import os
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def verify_task1():
    print("=== VERIFYING TASK 1 DATASET ===")
    task1_dir = os.path.join(BASE_DIR, "Task_1_HeadCut_Detection", "Task_1_HeadCut_Detection")
    train_csv = os.path.join(task1_dir, "labels_train.csv")
    dev_csv = os.path.join(task1_dir, "labels_dev.csv")
    
    df_train = pd.read_csv(train_csv)
    df_dev = pd.read_csv(dev_csv)
    
    print(f"Train labels count: {len(df_train)}")
    print(f"Train label distribution:\n{df_train['head_cut'].value_counts()}")
    print(f"Dev labels count: {len(df_dev)}")
    print(f"Dev label distribution:\n{df_dev['head_cut'].value_counts()}")
    
    # Check train images existence
    train_img_dir = os.path.join(task1_dir, "images", "train")
    missing_train = [f for f in df_train['filename'] if not os.path.exists(os.path.join(train_img_dir, f))]
    print(f"Missing train images: {len(missing_train)}")
    
    # Check dev images existence
    dev_img_dir = os.path.join(task1_dir, "images", "dev")
    missing_dev = [f for f in df_dev['filename'] if not os.path.exists(os.path.join(dev_img_dir, f))]
    print(f"Missing dev images: {len(missing_dev)}")
    
    # Sample image size check
    sample_img_path = os.path.join(train_img_dir, df_train.iloc[0]['filename'])
    with Image.open(sample_img_path) as img:
        print(f"Sample train image size: {img.size} (W, H), mode: {img.mode}")

def verify_task2():
    print("\n=== VERIFYING TASK 2 DATASET ===")
    task2_dir = os.path.join(BASE_DIR, "Task_2_Image_Cropping", "Task_2_Image_Cropping")
    meta_train = os.path.join(task2_dir, "meta_train.csv")
    meta_dev = os.path.join(task2_dir, "meta_dev.csv")
    
    df_train = pd.read_csv(meta_train)
    df_dev = pd.read_csv(meta_dev)
    
    print(f"Train meta count: {len(df_train)}")
    print(f"Train garment types:\n{df_train['garment_type'].value_counts()}")
    print(f"Dev meta count: {len(df_dev)}")
    print(f"Dev garment types:\n{df_dev['garment_type'].value_counts()}")
    
    train_img_dir = os.path.join(task2_dir, "images", "train")
    missing_train = [f for f in df_train['filename'] if not os.path.exists(os.path.join(train_img_dir, f))]
    print(f"Missing train images: {len(missing_train)}")
    
    dev_img_dir = os.path.join(task2_dir, "images", "dev")
    missing_dev = [f for f in df_dev['filename'] if not os.path.exists(os.path.join(dev_img_dir, f))]
    print(f"Missing dev images: {len(missing_dev)}")

if __name__ == "__main__":
    verify_task1()
    verify_task2()
