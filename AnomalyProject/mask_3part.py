#!/usr/bin/env python3
import os, shutil
from pathlib import Path

# ========= HARD-CODE YOUR PATHS HERE =========
SORTED_DIR = Path(r"C:\Users\Lenovo\Downloads\Brain-Anomaly-Detection\AnomalyProject\sorted")
MASK_DIR   = Path(r"C:\Users\Lenovo\Downloads\Brain-Anomaly-Detection\AnomalyProject\Masks")
TRIMS      = ["1_trim","2+3_trim","2_trim", "3_trim"]   # change to ["3_trim"] if you only want third trimester
DRY_RUN    = False                   # set True to preview actions
# ============================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MASK_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# If your masks follow a specific suffix, add it here.
CANDIDATE_SUFFIXES = [
    "",                # exact stem match
    "_mask",
    "_mask_0",
    "_mask_1",
    "_label",
    "_labels",
    "_seg",
    "_segm",
    "_annotation",
    "_ann",
]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def find_mask(mask_root: Path, stem: str):
    """
    Try common naming variants for masks.
    Returns the first matching mask path or None.
    Preference: entries with '_mask' in the name if multiple exist.
    """
    candidates = []
    for suf in CANDIDATE_SUFFIXES:
        for ext in MASK_EXTS:
            p = mask_root / f"{stem}{suf}{ext}"
            if p.exists():
                candidates.append(p)

    if not candidates:
        # fall back: any file in mask_dir starting with <stem> and a mask-y extension
        globbed = list(mask_root.glob(f"{stem}*"))
        candidates = [p for p in globbed if p.is_file() and p.suffix.lower() in MASK_EXTS]

    if not candidates:
        return None

    # Prefer names containing "_mask", then shorter name
    candidates.sort(key=lambda x: (0 if "_mask" in x.stem.lower() else 1, len(x.name)))
    return candidates[0]

def copy_mask(mask_path: Path, dest_dir: Path, image_stem: str, dry=False):
    dst = dest_dir / f"{image_stem}{mask_path.suffix.lower()}"
    if dry:
        print(f"[DRY] copy: {mask_path} -> {dst}")
    else:
        shutil.copy2(mask_path, dst)

def fill_for_trim(trim_dir: Path, out_dir: Path, mask_root: Path, dry=False):
    ensure_dir(out_dir)
    imgs = list_images(trim_dir)
    matched, missing = 0, 0

    for img in imgs:
        stem = img.stem
        mask = find_mask(mask_root, stem)
        if mask is None:
            missing += 1
            print(f"[WARN] No mask found for '{stem}'")
            continue
        copy_mask(mask, out_dir, stem, dry=dry)
        matched += 1

    print(f"{trim_dir.name}: copied {matched} mask(s); missing {missing}")

def main():
    if not SORTED_DIR.exists():
        raise SystemExit(f"ERROR: SORTED_DIR not found: {SORTED_DIR}")
    if not MASK_DIR.exists():
        raise SystemExit(f"ERROR: MASK_DIR not found: {MASK_DIR}")

    print(f"sorted: {SORTED_DIR}")
    print(f"mask dir: {MASK_DIR}")
    print(f"processing trims: {TRIMS}")
    print(f"dry-run: {DRY_RUN}")

    for trim_name in TRIMS:
        trim_dir = SORTED_DIR / trim_name
        if not trim_dir.is_dir():
            print(f"[SKIP] {trim_dir} does not exist.")
            continue
        out_dir = SORTED_DIR / f"{trim_name}_mask"
        fill_for_trim(trim_dir, out_dir, MASK_DIR, dry=DRY_RUN)

if __name__ == "__main__":
    main()
