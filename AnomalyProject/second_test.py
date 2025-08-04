# import math
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import torch
# import torchvision
# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import os
#
# # === Load model definition (same UNet as training) ===
# class UNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         def CBR(in_ch, out_ch):
#             return torch.nn.Sequential(
#                 torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
#                 torch.nn.BatchNorm2d(out_ch),
#                 torch.nn.ReLU(inplace=True),
#                 torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
#                 torch.nn.BatchNorm2d(out_ch),
#                 torch.nn.ReLU(inplace=True)
#             )
#         self.enc1 = CBR(1, 64)
#         self.enc2 = CBR(64, 128)
#         self.enc3 = CBR(128, 256)
#         self.pool = torch.nn.MaxPool2d(2)
#
#         self.dec2 = CBR(256 + 128, 128)
#         self.dec1 = CBR(128 + 64, 64)
#
#         self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.final = torch.nn.Conv2d(64, 1, kernel_size=1)
#
#     def center_crop(self, enc_feat, target_feat):
#         _, _, h, w = target_feat.size()
#         return torchvision.transforms.CenterCrop([h, w])(enc_feat)
#
#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#
#         d2 = self.up(e3)
#         e2_crop = self.center_crop(e2, d2)
#         d2 = self.dec2(torch.cat([d2, e2_crop], dim=1))
#
#         d1 = self.up(d2)
#         e1_crop = self.center_crop(e1, d1)
#         d1 = self.dec1(torch.cat([d1, e1_crop], dim=1))
#
#         return torch.sigmoid(self.final(d1))
#
#
# # === Preprocessing transform ===
# transform = A.Compose([
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])
#
# # === Load trained model ===
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet().to(device)
# model.load_state_dict(torch.load('unet_skull_segmentation.pth', map_location=device))
# model.eval()
#
# # === Load and preprocess test image ===
# image_path = 'Patient00305_Plane3_3_of_5.png'  # <<< CHANGE THIS
#
# image = np.array(Image.open(image_path).convert("L"))
# augmented = transform(image=image)
# input_tensor = augmented["image"].unsqueeze(0).to(device)
#
# # === Predict mask ===
# with torch.no_grad():
#     pred_mask = model(input_tensor).squeeze().cpu().numpy()
#     binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
#
# # === Optional: Fit ellipse ===
# def fit_ellipse_from_mask(mask_tensor):
#     if isinstance(mask_tensor, torch.Tensor):
#         mask_np = mask_tensor.squeeze().cpu().numpy()
#     else:
#         mask_np = np.squeeze(mask_tensor)
#
#     mask_bin = (mask_np > 0.5).astype(np.uint8) * 255
#     contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if contours:
#         cnt = max(contours, key=cv2.contourArea)
#         if len(cnt) >= 5:
#             ellipse = cv2.fitEllipse(cnt)
#             (center, axes, angle) = ellipse
#             a = max(axes) / 2  # semi-major axis
#             b = min(axes) / 2  # semi-minor axis
#
#             # Use numerically stable approximation for ellipse perimeter
#             h = ((a - b)**2) / ((a + b)**2)
#             hc = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
#             return ellipse, hc
#     return None, None
#
# ellipse, hc = fit_ellipse_from_mask(binary_mask)
#
# # === Show results ===
# result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# if ellipse:
#     cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)
#     print(f"Estimated HC: {hc:.2f} pixels")
# else:
#     print("No ellipse could be fitted.")
#
# # === Save or display ===
# cv2.imwrite("prediction_overlay.png", result_img)
# plt.figure(figsize=(12,4))
# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap='gray')
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(1, 3, 2)
# plt.imshow(pred_mask, cmap='gray')
# plt.title("Predicted Mask")
# plt.axis("off")
#
# plt.subplot(1, 3, 3)
# plt.imshow(result_img)
# plt.title("Overlay with Ellipse")
# plt.axis("off")
#
# plt.tight_layout()
# plt.show()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import math
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === Paths ===
input_dir = 'sorted/3_trim'
output_dir = 'segmentation_results'
os.makedirs(output_dir, exist_ok=True)

# === Load model definition ===
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_ch, out_ch):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.pool = torch.nn.MaxPool2d(2)
        self.dec2 = CBR(256 + 128, 128)
        self.dec1 = CBR(128 + 64, 64)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = torch.nn.Conv2d(64, 1, kernel_size=1)

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

# === Load trained model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('unet_skull_segmentation.pth', map_location=device))
model.eval()

# === Preprocessing ===
transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# === Ellipse fitting ===
def fit_ellipse_from_mask(mask_tensor):
    mask_np = mask_tensor.squeeze() if isinstance(mask_tensor, np.ndarray) else mask_tensor.squeeze().cpu().numpy()
    mask_bin = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            a = max(axes) / 2
            b = min(axes) / 2
            h = ((a - b)**2) / ((a + b)**2)
            hc = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
            return ellipse, hc
    return None, None

# === Process each image ===
for filename in os.listdir(input_dir):
    if not filename.endswith(".png"):
        continue

    print(f"\nProcessing: {filename}")
    image_path = os.path.join(input_dir, filename)
    image = np.array(Image.open(image_path).convert("L"))

    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(input_tensor).squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    ellipse, hc = fit_ellipse_from_mask(binary_mask)
    result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if ellipse:
        cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)
        print(f"  ➤ Estimated HC: {hc:.2f} pixels")
    else:
        print("  ➤ No ellipse could be fitted.")

    overlay_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
    mask_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.png")
    cv2.imwrite(overlay_path, result_img)
    cv2.imwrite(mask_path, binary_mask)
