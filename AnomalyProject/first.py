# import torch
# import torch.nn as nn
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader, WeightedRandomSampler
# import os
# import numpy as np
#
# # ==============================
# # 1. Configuration
# # ==============================
# data_dir = 'dataset'
# batch_size = 16
# num_epochs = 10
# num_classes = 3
# learning_rate = 1e-4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ==============================
# # 2. Transforms
# # ==============================
# train_transforms = transforms.Compose([
#     transforms.Resize((661, 661)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])
#
# val_transforms = transforms.Compose([
#     transforms.Resize((661, 661)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])
#
# # ==============================
# # 2.5 Optional: Auto-split data into train/val if val/ doesn't exist
# # ==============================
# from shutil import copy2
# import random
#
# split_ratio = 0.8  # 80% train, 20% val
# split_seed = 42
#
# train_path = os.path.join(data_dir, 'train')
# val_path = os.path.join(data_dir, 'val')
#
# if not os.path.exists(val_path) or not any(os.scandir(val_path)):
#     print("🔀 Splitting data into train/val folders...")
#     random.seed(split_seed)
#
#     for class_name in os.listdir(train_path):
#         class_dir = os.path.join(train_path, class_name)
#         if not os.path.isdir(class_dir):
#             continue
#
#         images = os.listdir(class_dir)
#         random.shuffle(images)
#
#         split_idx = int(len(images) * split_ratio)
#         train_imgs = images[:split_idx]
#         val_imgs = images[split_idx:]
#
#         val_class_dir = os.path.join(val_path, class_name)
#         os.makedirs(val_class_dir, exist_ok=True)
#
#         for img_name in val_imgs:
#             src = os.path.join(class_dir, img_name)
#             dst = os.path.join(val_class_dir, img_name)
#             copy2(src, dst)
#
#     print("✅ Split complete.")
#
# # ==============================
# # 3. Datasets and Dataloaders
# # ==============================
# train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
# val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)
#
# class_names = train_dataset.classes
# print("Classes:", class_names)
#
# # Calculate class counts for WeightedRandomSampler
# targets = train_dataset.targets  # list of class indices for each sample
# class_sample_counts = np.bincount(targets)  # count how many samples per class
#
# # Calculate weights for each class (inverse of frequency)
# class_weights = 1. / class_sample_counts
# # Weight for each sample in the dataset
# samples_weights = class_weights[targets]
#
# sampler = WeightedRandomSampler(weights=samples_weights,
#                                 num_samples=len(samples_weights),
#                                 replacement=True)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# # ==============================
# # 4. Model Setup (ResNet18)
# # ==============================
# model = models.resnet18(pretrained=True)
# model.fc = nn.Sequential(
#     nn.LayerNorm(model.fc.in_features),  # optional
#     nn.Dropout(0.5),
#     nn.Linear(model.fc.in_features, num_classes)
# )
#
# model = model.to(device)
#
# # Use class weights in loss function for imbalanced data
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#
# # ==============================
# # 5. Training Loop
# # ==============================
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     for imgs, labels in train_loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(imgs)
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * imgs.size(0)
#         _, preds = torch.max(outputs, 1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#
#     train_loss = running_loss / total
#     train_acc = correct / total
#
#     # Validation
#     model.eval()
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for imgs, labels in val_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs)
#             _, preds = torch.max(outputs, 1)
#
#             val_correct += (preds == labels).sum().item()
#             val_total += labels.size(0)
#
#     val_acc = val_correct / val_total
#
#     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
#
# # ==============================
# # 6. Save Model
# # ==============================
# torch.save(model, 'model.pth')
# print("✅ Model saved as model.pth")

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
from shutil import copy2
import random
from PIL import Image

# ==============================
# 1. Configuration
# ==============================
data_dir = 'dataset'
batch_size = 16
num_epochs = 10
num_classes = 3
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ==============================
# 2. Transforms
# ==============================
train_transforms = transforms.Compose([
    transforms.Resize((661, 661)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((661, 661)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 2.5 Auto-split if needed
# ==============================
split_ratio = 0.8
split_seed = 42

train_path = os.path.join(data_dir, 'train')
val_path = os.path.join(data_dir, 'val')

if not os.path.exists(val_path) or not any(os.scandir(val_path)):
    print("🔀 Splitting data into train/val folders...")
    random.seed(split_seed)

    for class_name in os.listdir(train_path):
        class_dir = os.path.join(train_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        print(f"🔍 Found {len(images)} images in {class_name}")
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        val_imgs = images[split_idx:]

        val_class_dir = os.path.join(val_path, class_name)
        os.makedirs(val_class_dir, exist_ok=True)

        for img_name in val_imgs:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(val_class_dir, img_name)
            copy2(src, dst)

    print("✅ Data split complete.")

# ==============================
# 3. Datasets and Dataloaders
# ==============================
print("📂 Loading datasets...")
train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_path, transform=val_transforms)

print("🔢 Classes found:", train_dataset.classes)
print("📊 Train dataset size:", len(train_dataset))
print("📊 Val dataset size:", len(val_dataset))

# Early exit if datasets are empty
if len(train_dataset) == 0:
    raise ValueError("❌ Train dataset is empty!")
if len(val_dataset) == 0:
    raise ValueError("❌ Val dataset is empty!")

# Sample check
print("🔍 Sample file:", train_dataset.samples[0])

# ==============================
# 3.5 Weighted Sampler
# ==============================
targets = train_dataset.targets
class_sample_counts = np.bincount(targets)
print("📊 Class sample counts:", class_sample_counts)

class_weights = 1. / class_sample_counts
samples_weights = class_weights[targets]

sampler = WeightedRandomSampler(weights=samples_weights,
                                num_samples=len(samples_weights),
                                replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Test batch loading
print("🧪 Testing train dataloader...")
for imgs, labels in train_loader:
    print(f"✅ Batch loaded. Image batch shape: {imgs.shape}, Label shape: {labels.shape}")
    break

# ==============================
# 4. Model Setup (ResNet18)
# ==============================
print("📦 Loading ResNet18 model...")
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.LayerNorm(model.fc.in_features),
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# ==============================
# 5. Training Loop
# ==============================
print("🚀 Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"[Epoch {epoch+1}/{num_epochs}] 🏋️ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# ==============================
# 6. Save Model
# ==============================
torch.save(model, 'model.pth')
print("✅ Model saved as model.pth")
