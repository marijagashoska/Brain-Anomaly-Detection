# # test_extract_masks.py
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import math
# import cv2
# import numpy as np
# import torch
# import torchvision
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#
# # ========= PATHS (edit these) =========
# INPUT_DIR   = r"tester-segm"                   # folder with ultrasound images
# OUTPUT_DIR  = r"segmentation_results_masks"    # where to save masks/overlays
# MODEL_PATH  = r"segmentation_second.pth"       # your trained weights (or best .pth)
#
# # ========= KNOBS =========
# THRESH = 0.55               # threshold on sigmoid probabilities
# OPEN_K, CLOSE_K = 3, 5      # morphology (odd integers >= 3)
# KEEP_OVERLAY = True         # also save green overlay images
# SAVE_PROBS_NPY = False      # save raw probability arrays as .npy
# IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
#
# # ========= DEVICE =========
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ========= MODEL (same arch as your training code) =========
# USE_GROUPNORM = True
# def norm_layer(num_channels):
#     return torch.nn.GroupNorm(min(8, num_channels), num_channels) if USE_GROUPNORM else torch.nn.BatchNorm2d(num_channels)
#
# class DoubleConv(torch.nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_ch, out_ch, 3, padding=1), norm_layer(out_ch), torch.nn.ReLU(inplace=True),
#             torch.nn.Conv2d(out_ch, out_ch, 3, padding=1), norm_layer(out_ch), torch.nn.ReLU(inplace=True),
#         )
#     def forward(self, x): return self.conv(x)
#
# class UNet(torch.nn.Module):
#     def __init__(self, n_channels=1, n_classes=1, base=32):
#         super().__init__()
#         self.d1 = DoubleConv(n_channels, base)
#         self.d2 = DoubleConv(base, base*2)
#         self.d3 = DoubleConv(base*2, base*4)
#         self.d4 = DoubleConv(base*4, base*8)
#         self.maxp = torch.nn.MaxPool2d(2)
#         self.up   = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.u3 = DoubleConv(base*4 + base*8, base*4)
#         self.u2 = DoubleConv(base*2 + base*4, base*2)
#         self.u1 = DoubleConv(base   + base*2, base)
#         self.outc = torch.nn.Conv2d(base, n_classes, 1)
#
#     def _center_crop_to(self, enc_feat, tgt_feat):
#         _, _, h, w = tgt_feat.size()
#         return torchvision.transforms.CenterCrop([h, w])(enc_feat)
#
#     def forward(self, x):
#         c1 = self.d1(x); x = self.maxp(c1)
#         c2 = self.d2(x); x = self.maxp(c2)
#         c3 = self.d3(x); x = self.maxp(c3)
#         x  = self.d4(x)
#         x = self.up(x);  x = torch.cat([x, self._center_crop_to(c3, x)], 1); x = self.u3(x)
#         x = self.up(x);  x = torch.cat([x, self._center_crop_to(c2, x)], 1); x = self.u2(x)
#         x = self.up(x);  x = torch.cat([x, self._center_crop_to(c1, x)], 1); x = self.u1(x)
#         return self.outc(x)
#
# # ========= TRANSFORM =========
# transform = A.Compose([
#     A.Normalize(mean=(0.5,), std=(0.5,)),
#     ToTensorV2()
# ])
#
# # ========= HELPERS =========
# def largest_component(mask_bin):
#     """mask_bin uint8 {0,1} -> keep only largest connected component."""
#     m = (mask_bin > 0).astype(np.uint8)
#     if m.sum() == 0:
#         return m
#     num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
#     if num <= 1:
#         return np.zeros_like(m)
#     idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
#     return (labels == idx).astype(np.uint8)
#
# def clean_mask(mask_bin):
#     """Largest CC + fill holes + open/close."""
#     m = largest_component(mask_bin)
#
#     # fill holes via flood fill
#     h, w = m.shape
#     m255 = (m * 255).astype(np.uint8)
#     flood = m255.copy()
#     ffmask = np.zeros((h+2, w+2), np.uint8)
#     cv2.floodFill(flood, ffmask, (0,0), 255)
#     holes = cv2.bitwise_not(flood)
#     m255  = cv2.bitwise_or(m255, holes)
#
#     if OPEN_K >= 3:
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
#         m255 = cv2.morphologyEx(m255, cv2.MORPH_OPEN, k, iterations=1)
#     if CLOSE_K >= 3:
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
#         m255 = cv2.morphologyEx(m255, cv2.MORPH_CLOSE, k, iterations=1)
#
#     return (m255 > 0).astype(np.uint8)
#
# def make_overlay(gray, mask_bin, color=(0,255,0)):
#     bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#     out = bgr.copy()
#     out[mask_bin.astype(bool)] = color
#     return out
#
# # ========= LOAD MODEL =========
# model = UNet(n_channels=1, n_classes=1, base=32).to(DEVICE)
# try:
#     state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)  # PyTorch >= 2.4
# except TypeError:
#     state = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state)
# model.eval()
# print(f"Loaded model: {MODEL_PATH}")
#
# # ========= I/O =========
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMG_EXTS)])
# if not files:
#     raise RuntimeError(f"No images with {IMG_EXTS} in {INPUT_DIR}")
# print(f"Found {len(files)} images in {INPUT_DIR}")
# print(f"Saving to: {os.path.abspath(OUTPUT_DIR)}")
#
# # ========= INFER =========
# with torch.no_grad():
#     for fp in files:
#         base = os.path.splitext(os.path.basename(fp))[0]
#         gray = np.array(Image.open(fp).convert("L"))
#         H, W = gray.shape[:2]
#
#         x = transform(image=gray)["image"].unsqueeze(0).to(DEVICE)  # [1,1,H,W]
#         logits = model(x)
#         probs  = torch.sigmoid(logits)[0,0].cpu().numpy()
#         probs  = cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)
#
#         if SAVE_PROBS_NPY:
#             np.save(os.path.join(OUTPUT_DIR, f"{base}_probs.npy"), probs)
#
#         raw_mask = (probs >= THRESH).astype(np.uint8)
#         mask     = clean_mask(raw_mask)
#
#         # saves
#         mask_path = os.path.join(OUTPUT_DIR, f"{base}_mask.png")
#         cv2.imwrite(mask_path, mask*255)
#
#         if KEEP_OVERLAY:
#             overlay = make_overlay(gray, mask)
#             ov_path = os.path.join(OUTPUT_DIR, f"{base}_overlay.png")
#             cv2.imwrite(ov_path, overlay)
#
#         print(f"[OK] {base} -> saved mask{', overlay' if KEEP_OVERLAY else ''}")
#
# print("Done.")

# test_masks_ransac_ellipse.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math, random, cv2, numpy as np, torch, torchvision
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ========= PATHS (edit) =========
INPUT_DIR   = r"tester-segm"
OUTPUT_DIR  = r"segmentation_results_ransac"
MODEL_PATH  = r"segmentation_second.pth"   # your trained UNet weights

# ========= KNOBS =========
THRESH = 0.55                  # prob -> mask
OPEN_K, CLOSE_K = 3, 7         # clean the mask
MIN_MAIN_FRAC = 0.01           # drop masks if too small

# ROI dilation so ring edges near boundary are kept
ROI_DILATE = 9

# Edge extraction
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSS_K = 5
CANNY_LOW, CANNY_HIGH = 40, 120
EDGE_CLOSE_K = 5
INTENSITY_TOP_PCT = 0.70       # keep only bright edges (skull)

# RANSAC ellipse
RANSAC_ITERS = 500
RING_THICKNESS_FOR_INLIERS = 2
INLIER_BAND_DILATE = 2         # tolerance pixels around ring
MIN_AXIS_RATIO = 0.55          # b/a >= this
MIN_AREA_FRAC = 0.01           # ellipse area >= this * H*W

# Final ring drawing / safety
DRAW_RING_THICKNESS = 3
SHRINK_INSIDE_PX = 6           # shrink fitted ellipse to sit inside skull
CLIP_MASK_WITH_ELLIPSE = True  # removes left protrusion
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ========= DEVICE =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= MODEL (same as training) =========
USE_GROUPNORM = True
def norm_layer(c): return torch.nn.GroupNorm(min(8, c), c) if USE_GROUPNORM else torch.nn.BatchNorm2d(c)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1), norm_layer(out_ch), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1), norm_layer(out_ch), torch.nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(n_channels, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.maxp = torch.nn.MaxPool2d(2)
        self.up   = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.u3 = DoubleConv(base*4 + base*8, base*4)
        self.u2 = DoubleConv(base*2 + base*4, base*2)
        self.u1 = DoubleConv(base   + base*2, base)
        self.outc = torch.nn.Conv2d(base, n_classes, 1)
    def _center_crop_to(self, enc_feat, tgt_feat):
        _, _, h, w = tgt_feat.size()
        return torchvision.transforms.CenterCrop([h, w])(enc_feat)
    def forward(self, x):
        c1 = self.d1(x); x = self.maxp(c1)
        c2 = self.d2(x); x = self.maxp(c2)
        c3 = self.d3(x); x = self.maxp(c3)
        x  = self.d4(x)
        x = self.up(x);  x = torch.cat([x, self._center_crop_to(c3, x)], 1); x = self.u3(x)
        x = self.up(x);  x = torch.cat([x, self._center_crop_to(c2, x)], 1); x = self.u2(x)
        x = self.up(x);  x = torch.cat([x, self._center_crop_to(c1, x)], 1); x = self.u1(x)
        return self.outc(x)

# ========= TRANSFORM =========
transform = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])

# ========= HELPERS =========
def largest_component(m):
    m = (m>0).astype(np.uint8)
    if m.sum()==0: return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n<=1: return np.zeros_like(m)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab==idx).astype(np.uint8)

def clean_mask(raw, H, W):
    m = largest_component(raw)
    if m.sum() < MIN_MAIN_FRAC * H * W: return np.zeros((H,W), np.uint8)
    m255 = (m*255).astype(np.uint8)
    # fill holes
    flood = m255.copy()
    ffmask = np.zeros((H+2, W+2), np.uint8)
    cv2.floodFill(flood, ffmask, (0,0), 255)
    m255 = cv2.bitwise_or(m255, cv2.bitwise_not(flood))
    if OPEN_K >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
        m255 = cv2.morphologyEx(m255, cv2.MORPH_OPEN, k, 1)
    if CLOSE_K >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
        m255 = cv2.morphologyEx(m255, cv2.MORPH_CLOSE, k, 1)
    return (m255>0).astype(np.uint8)

def roi_from_mask(mask, dilate_px=ROI_DILATE):
    if dilate_px<=0: return (mask>0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    return cv2.dilate((mask>0).astype(np.uint8), k, 1)>0

def bright_edges(gray, roi_bool):
    H,W = gray.shape
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    eq = clahe.apply(gray)
    if GAUSS_K>1: eq = cv2.GaussianBlur(eq, (GAUSS_K,GAUSS_K), 0)
    edges = cv2.Canny(eq, CANNY_LOW, CANNY_HIGH)
    edges[~roi_bool] = 0
    # brightness gating
    vals = gray[roi_bool]
    thr = np.percentile(vals, 100*INTENSITY_TOP_PCT) if vals.size>0 else 255
    bright = (gray >= thr).astype(np.uint8)
    edges = cv2.bitwise_and(edges, edges, mask=bright)
    # close gaps
    if EDGE_CLOSE_K>1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(EDGE_CLOSE_K,EDGE_CLOSE_K))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, 1)
    return edges

def ellipse_area(ell):
    (_, axes, _) = ell
    return math.pi*(axes[0]/2.0)*(axes[1]/2.0)

def axis_ratio(ell):
    _, axes, _ = ell
    a,b = max(axes), min(axes)
    return (b/a) if a>0 else 0.0

def draw_ring(shape, ell, thickness):
    h,w = shape
    ring = np.zeros((h,w), np.uint8)
    if ell is not None:
        cv2.ellipse(ring, ell, 255, thickness, lineType=cv2.LINE_AA)
    return ring

def shrink_ellipse(ell, px):
    if ell is None or px<=0: return ell
    (c, axes, ang) = ell
    maj, minr = float(axes[0]), float(axes[1])
    maj2 = max(maj - 2*px, 2.0)
    min2 = max(minr - 2*px, 2.0)
    if maj2<=2.0 or min2<=2.0: return None
    return (c, (maj2, min2), ang)

def ransac_ellipse(edges, gray, min_area_frac, min_axis_ratio,
                   iters=RANSAC_ITERS, ring_th=RING_THICKNESS_FOR_INLIERS,
                   band_dilate=INLIER_BAND_DILATE):
    """RANSAC over edge points; score by inlier fraction & ring brightness."""
    ys, xs = np.where(edges>0)
    pts = np.stack([xs, ys], axis=1)
    if len(pts) < 50:
        return None
    H,W = edges.shape
    area_min = H*W*min_area_frac

    # Precompute structuring element for band dilation
    if band_dilate>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band_dilate+1, 2*band_dilate+1))
    best = None
    best_score = -1.0

    for _ in range(iters):
        # sample 5 points
        idx = np.random.choice(len(pts), size=5, replace=False)
        sample = pts[idx].reshape(-1,1,2).astype(np.int32)
        try:
            ell = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        # sanity checks
        if axis_ratio(ell) < min_axis_ratio:
            continue
        if ellipse_area(ell) < area_min:
            continue

        # make thin ring band mask
        ring = draw_ring(edges.shape, ell, ring_th)
        if band_dilate>0:
            ring = cv2.dilate(ring, k, 1)
        ring_bool = ring>0

        # inliers: edge pixels that fall on the ring band
        inliers = np.logical_and(edges>0, ring_bool)
        inlier_count = int(inliers.sum())
        inlier_frac = inlier_count / max(len(pts), 1)

        # brightness score on the band
        bright_score = float(gray[ring_bool].mean()) if ring_bool.any() else 0.0

        # combine scores (tune weights if needed)
        score = 0.7*inlier_frac + 0.3*(bright_score/255.0)

        if score > best_score:
            best_score = score
            best = ell

    # optional refinement: re-fit on inliers of best
    if best is not None:
        ring = draw_ring(edges.shape, best, ring_th)
        if band_dilate>0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band_dilate+1, 2*band_dilate+1))
            ring = cv2.dilate(ring, k, 1)
        yy, xx = np.where(np.logical_and(edges>0, ring>0))
        if len(xx) >= 5:
            pts2 = np.stack([xx, yy], axis=1).reshape(-1,1,2).astype(np.int32)
            try:
                best = cv2.fitEllipse(pts2)
            except cv2.error:
                pass
    return best

def approx_hc(ell):
    (_, axes, _) = ell
    a = max(axes)/2.0; b = min(axes)/2.0
    h = ((a-b)**2)/((a+b)**2 + 1e-8)
    return math.pi*(a+b)*(1 + (3*h)/(10 + math.sqrt(4 - 3*h)))

# ========= LOAD MODEL =========
model = UNet(n_channels=1, n_classes=1, base=32).to(DEVICE)
try:
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)  # PyTorch >=2.4
except TypeError:
    state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state); model.eval()
print(f"Loaded model: {MODEL_PATH}")

# ========= I/O =========
os.makedirs(OUTPUT_DIR, exist_ok=True)
files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMG_EXTS)])
if not files: raise RuntimeError(f"No images with {IMG_EXTS} in {INPUT_DIR}")
print(f"Found {len(files)} images in {INPUT_DIR}")
print(f"Saving to: {os.path.abspath(OUTPUT_DIR)}")

with torch.no_grad():
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        gray = np.array(Image.open(fp).convert("L"))
        H, W = gray.shape[:2]

        x = transform(image=gray)["image"].unsqueeze(0).to(DEVICE)
        probs = torch.sigmoid(model(x))[0,0].cpu().numpy()
        probs = cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)

        raw = (probs >= THRESH).astype(np.uint8)
        mask = clean_mask(raw, H, W)

        # ROI & edges
        roi = roi_from_mask(mask, ROI_DILATE)
        edges = bright_edges(gray, roi)

        # Robust ellipse
        ell = ransac_ellipse(edges, gray, MIN_AREA_FRAC, MIN_AXIS_RATIO)
        if ell is not None and SHRINK_INSIDE_PX>0:
            ell = shrink_ellipse(ell, SHRINK_INSIDE_PX)

        # Clip mask with filled ellipse to kill any protrusion
        if CLIP_MASK_WITH_ELLIPSE and ell is not None:
            ell_fill = np.zeros((H,W), np.uint8)
            cv2.ellipse(ell_fill, ell, 255, -1, lineType=cv2.LINE_AA)
            mask = cv2.bitwise_and(mask, (ell_fill>0).astype(np.uint8))

        # Build outputs
        ring = np.zeros((H,W), np.uint8)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if ell is not None:
            cv2.ellipse(ring, ell, 255, DRAW_RING_THICKNESS, lineType=cv2.LINE_AA)
            cv2.ellipse(overlay, ell, (0,255,0), 2)
            print(f"[OK] {base}: HC≈{approx_hc(ell):.1f}px  (inliers-based)")
        else:
            print(f"[WARN] {base}: ellipse not found; check thresholds.")

        # green fill for mask + ring on top
        overlay[mask.astype(bool)] = (0,255,0)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_mask.png"), mask*255)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_ring.png"), ring)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_overlay.png"), overlay)

print("Done.")
