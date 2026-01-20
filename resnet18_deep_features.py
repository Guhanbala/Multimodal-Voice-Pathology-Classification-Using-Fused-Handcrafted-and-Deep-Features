import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np

# ================= CONFIG =================
BASE_DIR = Path(__file__).parent.resolve()
SPEC_DIR = BASE_DIR / "Extracted_Features" / "Spectrograms"
OUTPUT_DIR = BASE_DIR / "Deep_Features"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4  # After grouping: 0,1,2,3
BATCH_SIZE = 16
EPOCHS = 7
LR = 1e-4

# =========================================

class MelDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform
        # Remap labels to 0-3
        label_map = {0: 0, 1: 1, 3: 1, 2: 2, 4: 3}
        self.df['label'] = self.df['label'].map(label_map)
        unique_labels = sorted(self.df['label'].unique())
        if max(unique_labels) >= NUM_CLASSES or min(unique_labels) < 0:
            print(f"⚠️ Invalid labels in dataset: {unique_labels}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['patient_id']
        label = int(row['label'])
        class_folder = f"C{row['original_label']}_" + row['disease'].replace(" ", "_")  # Use original label for folder
        img_path = self.root_dir / class_folder / f"{pid}.png"
        if not img_path.exists():
            print(f"⚠️ Missing image: {img_path} — skipping")
            return self.__getitem__((idx + 1) % len(self.df))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load metadata
metadata = pd.read_csv(BASE_DIR / "Final_5Class_Dataset" / "metadata.csv")
metadata['original_label'] = metadata['label']  # Save for folder names

# Group labels for training
label_map = {0: 0, 1: 1, 3: 1, 2: 2, 4: 3}
metadata['label'] = metadata['label'].map(label_map)
print("Labels after grouping:")
print(metadata['label'].value_counts())

# Split
train_df, test_df = train_test_split(metadata, test_size=0.2, stratify=metadata['label'], random_state=42)

# Speech
train_dataset_speech = MelDataset(train_df, SPEC_DIR / "speech", transform)
test_dataset_speech = MelDataset(test_df, SPEC_DIR / "speech", transform)
train_loader_speech = DataLoader(train_dataset_speech, batch_size=BATCH_SIZE, shuffle=True)
test_loader_speech = DataLoader(test_dataset_speech, batch_size=BATCH_SIZE, shuffle=False)

# EGG
train_dataset_egg = MelDataset(train_df, SPEC_DIR / "egg", transform)
test_dataset_egg = MelDataset(test_df, SPEC_DIR / "egg", transform)
train_loader_egg = DataLoader(train_dataset_egg, batch_size=BATCH_SIZE, shuffle=True)
test_loader_egg = DataLoader(test_dataset_egg, batch_size=BATCH_SIZE, shuffle=False)

# Model (shared for simplicity – train separately for speech/egg if needed)
def get_resnet():
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    # Fine-tune last block
    for param in resnet.parameters():
        param.requires_grad = False
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.fc.parameters():
        param.requires_grad = True
    return resnet.to(DEVICE)

criterion = nn.CrossEntropyLoss()
def train_model(loader, model):
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(loader):.4f}, Acc: {100.*correct/total:.2f}%")
    return model

# Train speech ResNet
model_speech = get_resnet()
model_speech = train_model(train_loader_speech, model_speech)
torch.save(model_speech.state_dict(), OUTPUT_DIR / "fine_tuned_resnet_speech.pth")

# Train EGG ResNet
model_egg = get_resnet()
model_egg = train_model(train_loader_egg, model_egg)
torch.save(model_egg.state_dict(), OUTPUT_DIR / "fine_tuned_resnet_egg.pth")

# Extract fine-tuned features
def extract_features(model, loader, output_csv, split_name=""):
    model.fc = nn.Identity()
    model.eval()
    rows = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Extracting {split_name}"):
            inputs = inputs.to(DEVICE)
            feats = model(inputs)
            for feat, label in zip(feats.cpu().numpy(), labels.numpy()):
                row = {'label': label}
                for i, v in enumerate(feat):
                    row[f"resnet_feat_{i+1}"] = v
                rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"✅ Fine-tuned features saved: {output_csv}")

extract_features(model_speech, train_loader_speech, OUTPUT_DIR / "resnet18_speech_train.csv", "train")
extract_features(model_speech, test_loader_speech, OUTPUT_DIR / "resnet18_speech_test.csv", "test")

extract_features(model_egg, train_loader_egg, OUTPUT_DIR / "resnet18_egg_train.csv", "train")
extract_features(model_egg, test_loader_egg, OUTPUT_DIR / "resnet18_egg_test.csv", "test")