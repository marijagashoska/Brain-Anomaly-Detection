import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === 1. Match image and mask pairs ===
image_dir = 'sorted/2_trim'
mask_dir =  'Masks'


image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

mask_dict = {
    f.replace('_mask_0', '').replace('.png', ''): os.path.join(mask_dir, f)
    for f in mask_files
}

paired_paths = []
for img in image_files:
    base_name = img.replace('.png', '')
    if base_name in mask_dict:
        img_path = os.path.join(image_dir, img)
        mask_path = mask_dict[base_name]
        paired_paths.append((img_path, mask_path))

print(f"Found {len(paired_paths)} valid image-mask pairs.")

# === 2. Dataset ===
class UltrasoundSegmentationDataset(Dataset):
    def __init__(self, image_mask_pairs, transform=None):
        self.image_mask_pairs = image_mask_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)

# === 3. Albumentations Transform (no resizing) ===
transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# === 4. U-Net Model ===
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.dec2 = CBR(256 + 128, 128)
        self.dec1 = CBR(128 + 64, 64)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def center_crop(self, enc_feat, target_feat):
        _, _, h, w = target_feat.size()
        return torchvision.transforms.CenterCrop([h, w])(enc_feat)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up(e3)
        e2_crop = self.center_crop(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_crop], dim=1))

        d1 = self.up(d2)
        e1_crop = self.center_crop(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_crop], dim=1))

        return torch.sigmoid(self.final(d1))

# === 5. Dice Loss ===
def dice_loss(pred, target, smooth=1.):
    if pred.shape != target.shape:
        min_h = min(pred.shape[2], target.shape[2])
        min_w = min(pred.shape[3], target.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        target = target[:, :, :min_h, :min_w]

    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))



# === 6. Train/Val Split ===
train_paths, val_paths = train_test_split(paired_paths, test_size=0.2, random_state=42)
train_ds = UltrasoundSegmentationDataset(train_paths, transform)
val_ds = UltrasoundSegmentationDataset(val_paths, transform)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# === 7. Training ===
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
# for epoch in range(10):
#     model.train()
#     train_loss = 0
#     for img, mask in train_loader:
#         img, mask = img.to(device), mask.to(device)
#         pred = model(img)
#         loss = dice_loss(pred, mask)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for img, mask in val_loader:
#             img, mask = img.to(device), mask.to(device)
#             pred = model(img)
#             loss = dice_loss(pred, mask)
#             val_loss += loss.item()
#
#     print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
#
# === 7. Training ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    print(f"\n=== Epoch {epoch + 1} ===")

    # Training
    model.train()
    train_loss = 0
    print("Training...")
    for batch_idx, (img, mask) in enumerate(train_loader):
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = dice_loss(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0
    print("Validating...")
    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(val_loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = dice_loss(pred, mask)
            val_loss += loss.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  Val Batch {batch_idx + 1}/{len(val_loader)} - Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} Summary:")
    print(f"  Avg Train Loss: {train_loss / len(train_loader):.4f}")
    print(f"  Avg Val Loss:   {val_loss / len(val_loader):.4f}")

# === 8. Save Model ===
torch.save(model.state_dict(), 'unet_skull_segmentation.pth')

# === 9. Postprocessing: Fit Ellipse ===
def fit_ellipse_from_mask(mask_tensor):
    mask_np = mask_tensor.squeeze().cpu().numpy()
    mask_bin = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            hc = np.pi * (3*(axes[0]+axes[1])/2 - np.sqrt((3*axes[0]+axes[1])*(axes[0]+3*axes[1])))  # HC
            return ellipse, hc
    return None, None