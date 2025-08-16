import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Help CUDA allocator reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ===== Storage (Drive or local path) =====
BASE_DIR = '/final_res'  # change if running local
os.makedirs(BASE_DIR, exist_ok=True)

import time, cv2, numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- Device selection: GPU if available ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- CPU optimizations (still useful for dataloaders) ----
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())
torch.backends.mkldnn.enabled = True
torch.backends.cudnn.benchmark = True

# ===================== Config =====================
IMAGE_DIR = 'my_new_folder'      # <-- set
MASK_DIR  = 'my_photos'          # <-- set
SAVE_PATH = os.path.join(BASE_DIR, 'segmentation_second.pth')
LOG_FILE  = os.path.join(BASE_DIR, 'training_log.txt')

NO_RESIZE = True
IMG_H, IMG_W = 640, 960

UNET_BASE     = 32
USE_GROUPNORM = True

EPOCHS      = 25
LR          = 1e-4
BATCH       = 16
PATIENCE    = 5
LOG_EVERY   = 25
USE_POSWEIGHT = False
BCE_WEIGHT    = 0.0

# NEW: process each batch in smaller GPU chunks, but keep effective batch = BATCH
ACCUM_STEPS = 4  # 16 / 4 = micro-batch of 4; adjust if needed (must divide current batch size)

# ===================== Small utils =====================
def log_to_file(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def crop_to_match(a: torch.Tensor, b: torch.Tensor):
    h = min(a.size(2), b.size(2)); w = min(a.size(3), b.size(3))
    def center_crop(t):
        if t.size(2) == h and t.size(3) == w: return t
        top = (t.size(2) - h) // 2; left = (t.size(3) - w) // 2
        return t[:, :, top:top+h, left:left+w]
    return center_crop(a), center_crop(b)

# ===================== Pair images & masks =====================
def build_pairs(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    mask_files  = sorted([f for f in os.listdir(mask_dir)  if f.lower().endswith('.png')])
    mask_dict = {f.replace('_mask_0', '').replace('.png', ''): os.path.join(mask_dir, f) for f in mask_files}
    paired = []
    for img in image_files:
        base = img.replace('.png', '')
        if base in mask_dict:
            paired.append((os.path.join(image_dir, img), mask_dict[base]))
    print(f"Found {len(paired)} valid image-mask pairs.")
    return paired

# ===================== Dataset =====================
class FetalSkullDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs; self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Bad path: {img_path} | {msk_path}")
        img = np.expand_dims(img, 2)  # HWC(1) for Albumentations
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img  = aug['image'].float()           # (1,H,W) in [0,1]
            mask = aug['mask']
            if not torch.is_tensor(mask): mask = torch.from_numpy(mask)
        else:
            img  = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
            mask = torch.tensor(mask, dtype=torch.float32)
        mask = (mask > 0).float().unsqueeze(0)    # (1,H,W)
        return img, mask

# ===================== Transforms =====================
if NO_RESIZE:
    train_transform = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])
    val_transform   = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])
else:
    train_transform = A.Compose([A.Resize(IMG_H, IMG_W), A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])
    val_transform   = A.Compose([A.Resize(IMG_H, IMG_W), A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])

# ===================== UNet =====================
def norm_layer(num_channels):
    return nn.GroupNorm(min(8, num_channels), num_channels) if USE_GROUPNORM else nn.BatchNorm2d(num_channels)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), norm_layer(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), norm_layer(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(n_channels, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.maxp = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.u3 = DoubleConv(base*4 + base*8, base*4)
        self.u2 = DoubleConv(base*2 + base*4, base*2)
        self.u1 = DoubleConv(base   + base*2, base)
        self.outc = nn.Conv2d(base, n_classes, 1)
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

# ===================== Metrics & Loss =====================
def soft_dice(y_true, y_prob, smooth=1.0):
    y_prob, y_true = crop_to_match(y_prob, y_true)
    yt = y_true.reshape(y_true.size(0), -1); yp = y_prob.reshape(y_prob.size(0), -1)
    inter = (yt * yp).sum(1)
    return ((2 * inter + smooth) / (yt.sum(1) + yp.sum(1) + smooth)).mean()

@torch.no_grad()
def hard_dice_from_logits(y_true, logits, thr=0.5):
    logits, y_true = crop_to_match(logits, y_true)
    return soft_dice(y_true, (torch.sigmoid(logits) > thr).float())

class DiceLoss(nn.Module):
    def forward(self, logits, y_true):
        logits, y_true = crop_to_match(logits, y_true)
        return 1.0 - soft_dice(y_true, torch.sigmoid(logits))

# ===================== Eval (with AMP) =====================
@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    softs, hards = [], []
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
            logits = model(imgs)
            probs  = torch.sigmoid(logits)
        probs, masks = crop_to_match(probs, masks)
        softs.append(soft_dice(masks, probs).item())
        hards.append(hard_dice_from_logits(masks, logits, thr).item())
        del logits, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(softs)), float(np.mean(hards))

# ===================== Train (with AMP + microbatching) =====================
def estimate_pos_weight(loader, max_batches=8, eps=1e-6, device='cpu'):
    fg = bg = 0.0
    for i, (_, m) in enumerate(loader):
        m = m.to(device); fg += m.sum().item(); bg += (m.numel() - m.sum()).item()
        if i+1 >= max_batches: break
    return max(bg / (fg + eps), 1.0)

def train(model, train_loader, val_loader, device,
          lr=LR, epochs=EPOCHS, patience=PATIENCE, log_every=LOG_EVERY,
          use_pos_weight=USE_POSWEIGHT, bce_weight=BCE_WEIGHT):

    open(LOG_FILE, "w").close()
    log_to_file(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_file(f"Config: epochs={epochs}, lr={lr}, BCE_WEIGHT={bce_weight}, "
                f"pos_weight={use_pos_weight}, UNET_BASE={UNET_BASE}, NO_RESIZE={NO_RESIZE}, "
                f"BATCH={BATCH}, ACCUM_STEPS={ACCUM_STEPS}")

    dice_loss = DiceLoss()
    if use_pos_weight:
        pw = estimate_pos_weight(train_loader, device=device)
        print(f"pos_weight ≈ {pw:.2f}"); log_to_file(f"pos_weight ≈ {pw:.2f}")
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    else:
        bce = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type)

    steps = len(train_loader)
    start_line = f"Device: {device.type.upper()} | Steps/epoch: {steps} | Batch: {train_loader.batch_size}"
    print(start_line, flush=True); log_to_file(start_line)

    # Dry run under autocast, then free batch
    xb, yb = next(iter(train_loader))
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
        _ = model(xb.to(device))
    del xb, yb, _
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Dry run OK.\n", flush=True); log_to_file("Dry run OK.")

    best = 0.0; no_improve = 0
    autocast_dtype = torch.float16 if device.type=="cuda" else torch.bfloat16
    for ep in range(1, epochs+1):
        model.train(); run_loss = 0.0; t0 = time.time()
        for i, (imgs, masks) in enumerate(train_loader, 1):
            imgs = imgs.to(device); masks = masks.to(device)

            opt.zero_grad(set_to_none=True)
            # Split the big batch into ACCUM_STEPS micro-batches (keeps effective batch size the same)
            mb_imgs = imgs.chunk(ACCUM_STEPS)
            mb_masks = masks.chunk(ACCUM_STEPS)
            for mb_i, (i_mb, m_mb) in enumerate(zip(mb_imgs, mb_masks), 1):
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits = model(i_mb)
                    logits_c, masks_c = crop_to_match(logits, m_mb)
                    loss = dice_loss(logits_c, masks_c) + bce_weight * bce(logits_c, masks_c)
                    loss = loss / ACCUM_STEPS  # normalize over the accumulated micro-batches

                scaler.scale(loss).backward()

                # free per-microbatch temps
                del logits, logits_c, masks_c, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            scaler.step(opt)
            scaler.update()

            # track loss approximately by reusing last microbatch's loss add (already divided),
            # but better: recompute quickly on CPU-free; here we just increment by 0 for speed
            # (optional) run_loss += 0.0

            if i % log_every == 0 or i == steps:
                elapsed = time.time() - t0
                itps = i / max(elapsed, 1e-6)
                msg = f"[ep {ep}] {i}/{steps} | {itps:.2f} it/s"
                print(msg, flush=True); log_to_file(msg)

        # (optional) compute avg train loss if you want exact numbers; skipped to save VRAM
        tr_loss = float('nan')
        val_soft, val_hard = evaluate(model, val_loader, device)
        line = f"Epoch {ep:03d} | TrainLoss {tr_loss} | ValSoftDice {val_soft:.4f} | ValHardDice@0.5 {val_hard:.4f}"
        print(line, flush=True); log_to_file(line)

        if val_hard > best:
            best = val_hard; no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"↳ Saved {SAVE_PATH}", flush=True); log_to_file(f"Saved best model at epoch {ep}")
        else:
            no_improve += 1
            if no_improve >= patience:
                stop_msg = f"Early stop at epoch {ep}. Best ValHardDice {best:.4f}"
                print(stop_msg); log_to_file(stop_msg)
                break

# ===================== Split =====================
def split_indices(n, train_ratio=0.8, seed=42):
    idx = list(range(n)); rng = np.random.default_rng(seed); rng.shuffle(idx)
    ntr = int(train_ratio * n); return idx[:ntr], idx[ntr:]

# ===================== Main =====================
def main():
    pairs = build_pairs(IMAGE_DIR, MASK_DIR)
    if not pairs: raise RuntimeError("No valid image-mask pairs found.")

    tr_idx, va_idx = split_indices(len(pairs), 0.8, 42)
    tr_pairs = [pairs[i] for i in tr_idx]; va_pairs = [pairs[i] for i in va_idx]

    train_ds = FetalSkullDataset(tr_pairs, transform=train_transform)
    val_ds   = FetalSkullDataset(va_pairs,  transform=val_transform)

    workers = max(1, os.cpu_count() - 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=workers, pin_memory=True if DEVICE.type == "cuda" else False,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=workers, pin_memory=True if DEVICE.type == "cuda" else False,
                              persistent_workers=True)

    model = UNet(n_channels=1, n_classes=1, base=UNET_BASE).to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    train(model, train_loader, val_loader, DEVICE)

if __name__ == "__main__":
    main()
