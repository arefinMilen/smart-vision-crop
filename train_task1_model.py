import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(BASE_DIR, "Task_1_HeadCut_Detection", "Task_1_HeadCut_Detection")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "Candidate_Submission", "Task1_Solution", "task1_resnet18.pth")

class HeadCutDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['filename'], int(row['head_cut'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, fname

def train_model():
    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = HeadCutDataset(
        os.path.join(TASK1_DIR, 'labels_train.csv'),
        os.path.join(TASK1_DIR, 'images', 'train'),
        transform=train_transform
    )

    dev_dataset = HeadCutDataset(
        os.path.join(TASK1_DIR, 'labels_dev.csv'),
        os.path.join(TASK1_DIR, 'images', 'dev'),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # Loss & Optimizer (Weighted loss if needed)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_dev_acc = 0.0
    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_train_acc = train_correct / len(train_dataset)

        # Validation on Dev
        model.eval()
        dev_correct = 0
        dev_tp, dev_fp, dev_fn, dev_tn = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels, _ in dev_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                dev_correct += torch.sum(preds == labels.data).item()

                for p, t in zip(preds, labels):
                    p_item, t_item = p.item(), t.item()
                    if p_item == 1 and t_item == 1: dev_tp += 1
                    elif p_item == 1 and t_item == 0: dev_fp += 1
                    elif p_item == 0 and t_item == 1: dev_fn += 1
                    elif p_item == 0 and t_item == 0: dev_tn += 1

        dev_acc = dev_correct / len(dev_dataset)
        scheduler.step(dev_acc)

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Dev Acc: {dev_acc*100:.2f}% (TP={dev_tp}, FP={dev_fp}, FN={dev_fn}, TN={dev_tn})")

        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nTraining Complete! Best Dev Accuracy: {best_dev_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
