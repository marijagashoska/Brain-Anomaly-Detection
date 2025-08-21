# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import os
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
#
# import json, sys, csv
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# from datetime import datetime
#
# import torch
# import open_clip
# from PIL import Image
# import pandas as pd
#
# # =====================
# # 🔧 GLOBALS – EDIT THESE
# # =====================
# ROOT_DIR     = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\sorted"  # contains 2_trim, 3_trim, 2+3_trim
# CONFIG_PATH  = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\FetalCLIP_config.json"
# WEIGHTS_PATH = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\Weights\Weights\FetalCLIP_weights.pt"
#
# EXCEL_PATH   = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\tt_reports\tt_measurements.xlsx"
# EXCEL_SHEET  = "Summary"          # <-- only this sheet will be updated
# IMAGE_COL    = "image"            # column in Summary with image name/path
# GA_COL       = "ga"               # new column to write
# GA_INSERT_AFTER = "mask_path"     # try to insert after this column if present; else append
#
# _TS = datetime.now().strftime("%Y%m%d_%H%M%S")
# OUTPUT_EXCEL_PATH = str(Path(EXCEL_PATH).with_name(Path(EXCEL_PATH).stem + f"_with_ga_{_TS}" + Path(EXCEL_PATH).suffix))
#
# MODEL_NAME   = "FetalCLIP"
# DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# TOPK         = 1
# SAVE_CSV     = True
# CSV_PATH     = str(Path(ROOT_DIR) / f"ga_predictions_{_TS}.csv")
# # =====================
#
# ALLOWED_LABELS = {"2_trim":"2","3_trim":"3","2+3_trim":"2+3","1_trim":"1"}
#
# def trimester_weeks_from_key(key: str) -> List[int]:
#     if key == "2":   return list(range(14, 28))
#     if key == "3":   return list(range(28, 41))
#     if key == "2+3": return list(range(14, 41))
#     raise ValueError(f"Bad trimester key: {key}")
#
#
# def build_prompts(weeks: List[int], template: str = "{w} weeks gestation") -> List[str]:
#     return [template.format(w=w) for w in weeks]
#
# def register_model_from_config(model_name: str, config_json_path: str):
#     with open(config_json_path, "r", encoding="utf-8") as f:
#         cfg = json.load(f)
#     open_clip.factory._MODEL_CONFIGS[model_name] = cfg
#     return cfg
#
# def load_model_and_tools(model_name: str, weights_path: str, device: str):
#     model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=str(weights_path))
#     model.eval().to(device)
#     tokenizer = open_clip.get_tokenizer(model_name)
#     return model, preprocess, tokenizer
#
# def load_image(image_path: Path, preprocess, device: str):
#     img = Image.open(image_path).convert("RGB")
#     return preprocess(img).unsqueeze(0).to(device)
#
# @torch.no_grad()
# def encode_texts(model, tokenizer, prompts: List[str], device: str) -> torch.Tensor:
#     toks = tokenizer(prompts).to(device)
#     feats = model.encode_text(toks)
#     return feats / feats.norm(dim=-1, keepdim=True)
#
# @torch.no_grad()
# def encode_image(model, img_t: torch.Tensor) -> torch.Tensor:
#     feats = model.encode_image(img_t)
#     return feats / feats.norm(dim=-1, keepdim=True)
#
# def predict_week(img_feat: torch.Tensor, txt_feats: torch.Tensor, weeks: List[int], topk: int = 1):
#     sims = (img_feat @ txt_feats.T).squeeze(0)
#     best_idx = int(torch.argmax(sims).item())
#     best_week = weeks[best_idx]
#     top = None
#     if topk and topk > 1:
#         vals, idxs = torch.topk(sims, k=min(topk, len(weeks)))
#         top = [(weeks[int(i.item())], float(v.item())) for v, i in zip(vals, idxs)]
#     return best_week, sims.detach().cpu().numpy(), top
#
# def gather_images_only_trim(root: Path) -> Dict[str, List[Path]]:
#     valid_ext = {".png", ".jpg", ".jpeg"}
#     out = {}
#     for sub in root.iterdir():
#         if sub.is_dir() and sub.name.lower() in ALLOWED_LABELS:
#             imgs = [p for p in sub.rglob("*") if p.suffix.lower() in valid_ext]
#             if imgs:
#                 out[sub.name] = sorted(imgs)
#     return out
#
# # ---------- Excel helpers ----------
# def ensure_ga_column(df: pd.DataFrame, ga_col: str, insert_after: Optional[str] = None) -> pd.DataFrame:
#     if ga_col in df.columns:
#         return df
#     if insert_after and insert_after in df.columns:
#         cols = list(df.columns)
#         idx = cols.index(insert_after) + 1
#         left = df.iloc[:, :idx]
#         right = df.iloc[:, idx:]
#         ga_series = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), name=ga_col)
#         return pd.concat([left, ga_series, right], axis=1)
#     df[ga_col] = pd.Series(dtype="Int64")
#     return df
#
# def _basename_lower(x: str) -> Tuple[str, str]:
#     base = Path(str(x)).name
#     return base.lower(), Path(base).stem.lower()
#
# def build_excel_image_index(df: pd.DataFrame, image_col: str) -> Dict[str, List[int]]:
#     idx: Dict[str, List[int]] = {}
#     s = df[image_col].astype(str)
#     for i, val in enumerate(s):
#         with_ext, stem = _basename_lower(val)
#         idx.setdefault(with_ext, []).append(i)
#         idx.setdefault(stem, []).append(i)
#     return idx
#
# def save_excel_all_sheets(sheets: Dict[str, pd.DataFrame], out_path: str):
#     with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
#         for name, sdf in sheets.items():
#             sdf.to_excel(writer, sheet_name=name, index=False)
#
# # ---------- main ----------
# def main():
#     root = Path(ROOT_DIR)
#     sheets = pd.read_excel(EXCEL_PATH, sheet_name=None, engine="openpyxl")
#     if EXCEL_SHEET not in sheets:
#         raise ValueError(f"Sheet '{EXCEL_SHEET}' not found. Available: {list(sheets.keys())}")
#
#     df = sheets[EXCEL_SHEET]
#     if IMAGE_COL not in df.columns:
#         raise ValueError(f"'{EXCEL_SHEET}' missing column '{IMAGE_COL}'. Columns: {list(df.columns)}")
#
#     # ensure GA column on Summary only
#     df = ensure_ga_column(df, GA_COL, insert_after=GA_INSERT_AFTER)
#     sheets[EXCEL_SHEET] = df  # write back
#     excel_index = build_excel_image_index(df, IMAGE_COL)
#
#     # model
#     register_model_from_config(MODEL_NAME, CONFIG_PATH)
#     model, preprocess, tokenizer = load_model_and_tools(MODEL_NAME, WEIGHTS_PATH, DEVICE)
#
#     per_label_images = gather_images_only_trim(root)
#     if not per_label_images:
#         print(f"[INFO] No images found in {root}. Expected subfolders: {', '.join(ALLOWED_LABELS.keys())}")
#         return
#
#     # cache text feats per trimester bucket
#     text_cache: Dict[str, Tuple[List[int], torch.Tensor]] = {}
#
#     writer = None
#     csv_fh = None
#     if SAVE_CSV:
#         csv_fh = open(CSV_PATH, "w", newline="", encoding="utf-8")
#         writer = csv.writer(csv_fh)
#         writer.writerow(["folder", "image_filename", "excel_hits", "predicted_week", "status"])
#
#     total_images, updated_rows, missing = 0, 0, 0
#
#     try:
#         for folder_name, images in per_label_images.items():
#             tri_key = ALLOWED_LABELS[folder_name.lower()]
#             if tri_key not in text_cache:
#                 weeks = trimester_weeks_from_key(tri_key)
#                 prompts = build_prompts(weeks)
#                 txt_feats = encode_texts(model, tokenizer, prompts, DEVICE)
#                 text_cache[tri_key] = (weeks, txt_feats)
#             else:
#                 weeks, txt_feats = text_cache[tri_key]
#
#             print(f"\n=== {folder_name} | {len(images)} images ===")
#             for img_path in images:
#                 try:
#                     img_t = load_image(img_path, preprocess, DEVICE)
#                     img_feat = encode_image(model, img_t)
#                     best_week, _, _ = predict_week(img_feat, txt_feats, weeks, TOPK)
#
#                     fname_ext = img_path.name.lower()
#                     fname_stem = img_path.stem.lower()
#                     row_ids = excel_index.get(fname_ext, []) + excel_index.get(fname_stem, [])
#                     hits = len(row_ids)
#
#                     if hits > 0:
#                         df.loc[row_ids, GA_COL] = int(best_week)
#                         updated_rows += hits
#                         status = "UPDATED"
#                     else:
#                         missing += 1
#                         status = "NO_MATCH"
#
#                     print(f"[{status}] {img_path.name:40s} -> {best_week:2d} weeks | excel_hits={hits}")
#                     if writer:
#                         writer.writerow([folder_name, img_path.name, hits, int(best_week), status])
#
#                     total_images += 1
#
#                 except Exception as e:
#                     print(f"[FAIL] {img_path} | {e}")
#
#     except KeyboardInterrupt:
#         print("\n[INTERRUPTED] Saving progress...")
#
#     # write updated Summary back, preserve other sheets
#     sheets[EXCEL_SHEET] = df
#     print(f"\n[DEBUG] Saving to: {OUTPUT_EXCEL_PATH}")
#     save_excel_all_sheets(sheets, OUTPUT_EXCEL_PATH)
#
#     if csv_fh:
#         csv_fh.close()
#         print(f"[CSV] saved: {CSV_PATH}")
#
#     print(f"\nDone. Images processed: {total_images} | Summary rows updated: {updated_rows} | Images with no match: {missing}")
#     print(f"[EXCEL] original file unchanged: {EXCEL_PATH}")
#     print(f"[EXCEL] new file with GA column (Summary only): {OUTPUT_EXCEL_PATH}")
#
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"[ERROR] {e}", file=sys.stderr)
#         sys.exit(1)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json, sys, csv, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import open_clip
from PIL import Image
import pandas as pd

# =====================
# 🔧 GLOBALS – EDIT THESE
# =====================
ROOT_DIR     = r"C:\Users\Lenovo\Downloads\Brain-Anomaly-Detection\AnomalyProject\sorted"  # contains ..._mask folders
CONFIG_PATH  = r"C:\Users\Lenovo\Downloads\Brain-Anomaly-Detection\AnomalyProject\FetalCLIP_config.json"
WEIGHTS_PATH = r"C:\Users\Lenovo\Downloads\Brain-Anomaly-Detection\AnomalyProject\Weights\Weights\FetalCLIP_weights.pt"

EXCEL_PATH   = r"C:\Users\Lenovo\Downloads\Brain-Anomaly-Detection\AnomalyProject\tt_reports\tt_measurements.xlsx"
EXCEL_SHEET  = "Summary"          # only this sheet is edited in-place
IMAGE_COL    = "image"            # column in Summary with image name/path
GA_COL       = "ga"               # new column to write

# If you want a safety copy before overwriting, set to True
MAKE_BACKUP_COPY = False

MODEL_NAME   = "FetalCLIP"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TOPK         = 1
SAVE_CSV     = True
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH     = str(Path(ROOT_DIR) / f"ga_predictions_{_TS}.csv")
# =====================

# map visible folder name -> trimester bucket key
ALLOWED_MASK_FOLDERS = {
    "1_trim_mask": "1",
    "2_trim_mask": "2",
    "3_trim_mask": "3",
    "2+3_trim_mask": "2+3",
}

def trimester_weeks_from_key(key: str) -> List[int]:
    if key == "1":   return list(range(10, 14))  # optional; won’t be used unless you keep 1st-trim images
    if key == "2":   return list(range(14, 28))
    if key == "3":   return list(range(28, 41))
    if key == "2+3": return list(range(14, 41))
    raise ValueError(f"Bad trimester key: {key}")

def build_prompts(weeks: List[int], template: str = "{w} weeks gestation") -> List[str]:
    return [template.format(w=w) for w in weeks]

def register_model_from_config(model_name: str, config_json_path: str):
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    open_clip.factory._MODEL_CONFIGS[model_name] = cfg
    return cfg

def load_model_and_tools(model_name: str, weights_path: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=str(weights_path))
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

def load_image(image_path: Path, preprocess, device: str):
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)

@torch.no_grad()
def encode_texts(model, tokenizer, prompts: List[str], device: str) -> torch.Tensor:
    toks = tokenizer(prompts).to(device)
    feats = model.encode_text(toks)
    return feats / feats.norm(dim=-1, keepdim=True)

@torch.no_grad()
def encode_image(model, img_t: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(img_t)
    return feats / feats.norm(dim=-1, keepdim=True)

def predict_week(img_feat: torch.Tensor, txt_feats: torch.Tensor, weeks: List[int], topk: int = 1):
    sims = (img_feat @ txt_feats.T).squeeze(0)
    best_idx = int(torch.argmax(sims).item())
    best_week = weeks[best_idx]
    top = None
    if topk and topk > 1:
        vals, idxs = torch.topk(sims, k=min(topk, len(weeks)))
        top = [(weeks[int(i.item())], float(v.item())) for v, i in zip(vals, idxs)]
    return best_week, sims.detach().cpu().numpy(), top

# ---------- Mask discovery & mapping ----------
_VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_MASK_PAT = re.compile(r"(?:_mask(?:_\d+)?)$", re.IGNORECASE)  # strips _mask or _mask_0 etc from the STEM

def strip_mask_from_stem(stem: str) -> str:
    """Return original image stem by removing trailing mask suffixes."""
    return _MASK_PAT.sub("", stem)

def mask_folders(root: Path) -> Dict[str, List[Path]]:
    """Return {mask_folder: [mask_paths...]} for allowed mask folders only."""
    out: Dict[str, List[Path]] = {}
    for sub in root.iterdir():
        if sub.is_dir() and sub.name in ALLOWED_MASK_FOLDERS:
            files = [p for p in sub.rglob("*") if p.suffix.lower() in _VALID_EXT]
            if files:
                out[sub.name] = sorted(files)
    return out

def probable_original_image_path(mask_path: Path) -> Optional[Path]:
    """
    Try to find the original image:
    - switch folder 'X_trim_mask' -> 'X_trim'
    - replace stem by stripping '_mask' / '_mask_#'
    - keep same extension if possible; otherwise try common extensions
    """
    mfolder = mask_path.parent
    base = mask_path.stem
    orig_stem = strip_mask_from_stem(base)

    # switch folder
    nonmask_folder_name = mfolder.name.replace("_mask", "")
    orig_folder = mfolder.parent / nonmask_folder_name

    # candidates
    candidates = [orig_folder / f"{orig_stem}{mask_path.suffix}"]
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidates.append(orig_folder / f"{orig_stem}{ext}")

    for c in candidates:
        if c.exists():
            return c
    return None

# ---------- Excel helpers ----------
def ensure_ga_column(df: pd.DataFrame, ga_col: str) -> pd.DataFrame:
    if ga_col not in df.columns:
        df[ga_col] = pd.Series(dtype="Int64")  # nullable integer
    return df

def _basename_lower(x: str) -> Tuple[str, str]:
    base = Path(str(x)).name
    return base.lower(), Path(base).stem.lower()

def build_excel_image_index(df: pd.DataFrame, image_col: str) -> Dict[str, List[int]]:
    """
    Build an index so we can match both with extension and by stem only.
    """
    idx: Dict[str, List[int]] = {}
    s = df[image_col].astype(str)
    for i, val in enumerate(s):
        with_ext, stem = _basename_lower(val)
        idx.setdefault(with_ext, []).append(i)
        idx.setdefault(stem, []).append(i)
    return idx

# ---------- main ----------
def main():
    root = Path(ROOT_DIR)

    # load Excel & sheet
    all_sheets = pd.read_excel(EXCEL_PATH, sheet_name=None, engine="openpyxl")
    if EXCEL_SHEET not in all_sheets:
        raise ValueError(f"Sheet '{EXCEL_SHEET}' not found. Available: {list(all_sheets.keys())}")

    df = all_sheets[EXCEL_SHEET]
    if IMAGE_COL not in df.columns:
        raise ValueError(f"'{EXCEL_SHEET}' missing column '{IMAGE_COL}'. Columns: {list(df.columns)}")

    # add GA column if missing
    df = ensure_ga_column(df, GA_COL)
    all_sheets[EXCEL_SHEET] = df
    excel_index = build_excel_image_index(df, IMAGE_COL)

    # model
    register_model_from_config(MODEL_NAME, CONFIG_PATH)
    model, preprocess, tokenizer = load_model_and_tools(MODEL_NAME, WEIGHTS_PATH, DEVICE)

    per_mask_folder = mask_folders(root)
    if not per_mask_folder:
        print(f"[INFO] No mask images found in {root}. Expected any of: {', '.join(ALLOWED_MASK_FOLDERS.keys())}")
        return

    # cache text feats per trimester bucket
    text_cache: Dict[str, Tuple[List[int], torch.Tensor]] = {}

    writer = None
    csv_fh = None
    if SAVE_CSV:
        csv_fh = open(CSV_PATH, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_fh)
        writer.writerow(["mask_folder", "mask_filename", "orig_image_used", "excel_hits", "predicted_week", "status"])

    total_masks, updated_rows, missing = 0, 0, 0

    try:
        for folder_name, masks in per_mask_folder.items():
            tri_key = ALLOWED_MASK_FOLDERS[folder_name]
            if tri_key not in text_cache:
                weeks = trimester_weeks_from_key(tri_key)
                prompts = build_prompts(weeks)
                txt_feats = encode_texts(model, tokenizer, prompts, DEVICE)
                text_cache[tri_key] = (weeks, txt_feats)
            else:
                weeks, txt_feats = text_cache[tri_key]

            print(f"\n=== {folder_name} | {len(masks)} masks ===")
            for mask_path in masks:
                try:
                    # Find original image (preferred) or fall back to mask
                    orig_img = probable_original_image_path(mask_path)
                    img_for_model = orig_img if orig_img is not None else mask_path

                    img_t = load_image(img_for_model, preprocess, DEVICE)
                    img_feat = encode_image(model, img_t)
                    best_week, _, _ = predict_week(img_feat, txt_feats, weeks, TOPK)

                    # Build matching keys for Excel based on original image name (preferred)
                    preferred_name = (orig_img.name if orig_img else mask_path.name).lower()
                    preferred_stem = Path(preferred_name).stem.lower()

                    # also include the raw mask file name & stem just in case the xlsx used mask names
                    mask_name = mask_path.name.lower()
                    mask_stem = mask_path.stem.lower()

                    row_ids = []
                    for key in (preferred_name, preferred_stem, mask_name, mask_stem):
                        row_ids += excel_index.get(key, [])

                    hits = len(set(row_ids))
                    if hits > 0:
                        df.loc[sorted(set(row_ids)), GA_COL] = int(best_week)
                        updated_rows += hits
                        status = "UPDATED"
                    else:
                        missing += 1
                        status = "NO_MATCH"

                    print(f"[{status}] {mask_path.name:40s} -> {best_week:2d} w | excel_hits={hits} | used={'ORIG' if orig_img else 'MASK'}")
                    if writer:
                        writer.writerow([folder_name, mask_path.name, (orig_img.name if orig_img else mask_path.name), hits, int(best_week), status])

                    total_masks += 1

                except Exception as e:
                    print(f"[FAIL] {mask_path} | {e}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving progress...")

    # write updated Summary back (IN-PLACE in the SAME XLSX)
    all_sheets[EXCEL_SHEET] = df

    # optional safety copy
    if MAKE_BACKUP_COPY:
        backup = str(Path(EXCEL_PATH).with_name(Path(EXCEL_PATH).stem + f"_backup_{_TS}" + Path(EXCEL_PATH).suffix))
        with pd.ExcelWriter(backup, engine="openpyxl", mode="w") as writer:
            for name, sdf in all_sheets.items():
                sdf.to_excel(writer, sheet_name=name, index=False)
        print(f"[EXCEL] backup saved: {backup}")

    # overwrite the original file
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="w") as writer:
        for name, sdf in all_sheets.items():
            sdf.to_excel(writer, sheet_name=name, index=False)

    if csv_fh:
        csv_fh.close()
        print(f"[CSV] saved: {CSV_PATH}")

    print(f"\nDone. Masks processed: {total_masks} | Summary rows updated: {updated_rows} | Masks with no match: {missing}")
    print(f"[EXCEL] updated in place: {EXCEL_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
