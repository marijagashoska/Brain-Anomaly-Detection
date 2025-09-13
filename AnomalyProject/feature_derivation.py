import os, re, math, glob, textwrap
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

BASE_DIR    = r"C:\Users\Lenovo\Downloads\Brain-Anomaly-Detection\AnomalyProject"
INPUT_ROOT  = os.path.join(BASE_DIR, "sorted")
PIXEL_CSV   = os.path.join(BASE_DIR, "Trans-Thalamic-Pixel-Size.csv")

TRIM_FOLDERS = {
    "2_trim_mask": "2",
    "1_trim_mask": "1",
    "3_trim_mask": "3",
    "2+3_trim_mask": "2+3",
}

OUT_DIR   = os.path.join(BASE_DIR, "tt_reports")
OUT_CSV   = os.path.join(OUT_DIR, "tt_measurements_report.csv")
OUT_XLSX  = os.path.join(OUT_DIR, "tt_measurements.xlsx")
DEBUG_TXT = os.path.join(OUT_DIR, "debug_report.txt")

MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_SUFFIXES = [
    "_mask_0", "_mask_1", "_mask",
    "_label", "_labels", "_seg", "_segm",
    "_annotation", "_ann"
]

def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_all_images(root: str) -> List[str]:
    files = []
    for ext in MASK_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return [p for p in files if os.path.isfile(p)]

def looks_like_mask(p: str) -> bool:
    stem = os.path.splitext(os.path.basename(p))[0].lower()
    if any(stem.endswith(suf) for suf in MASK_SUFFIXES):
        return True
    return "_mask" in stem or "_seg" in stem or "_label" in stem

def list_masks(root: str) -> List[str]:
    files = list_all_images(root)
    masks = [p for p in files if looks_like_mask(p)]
    return masks if masks else files

def strip_mask_suffix(stem: str) -> str:
    for suf in MASK_SUFFIXES:
        if stem.lower().endswith(suf):
            return stem[: -len(suf)]
    m = re.match(r"^(.*)_(mask|seg|label)(\_\d+)?$", stem, re.IGNORECASE)
    return m.group(1) if m else stem

def load_pixel_sizes(csv_path: str) -> Tuple[Dict[str, float], Dict[str, float], List[str], str]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Label" not in df.columns:
        label_like = [c for c in df.columns if c.lower().strip() == "label"]
        if not label_like:
            raise ValueError("CSV must contain a 'Label' column.")
        df = df.rename(columns={label_like[0]: "Label"})
    pix_cols = [c for c in df.columns if "pixel" in c.lower()]
    if not pix_cols:
        raise ValueError("CSV must have a column containing 'pixel' with the mm/px values (e.g., 'Pixel  in mm').")
    mm_col = pix_cols[0]

    exact_map: Dict[str, float] = {}
    label_basenames: List[str] = []
    for lbl, mm in zip(df["Label"], df[mm_col]):
        base = os.path.basename(str(lbl)).strip()
        exact_map[base] = float(mm)
        label_basenames.append(base)

    pat_vals: Dict[str, List[float]] = {}
    for base, mm in exact_map.items():
        m = re.match(r"^(Patient\d+)", base, re.IGNORECASE)
        if not m:
            continue
        pat = m.group(1)
        pat_vals.setdefault(pat, []).append(mm)
    patient_avg_map = {pat: float(np.mean(vals)) for pat, vals in pat_vals.items()}

    return exact_map, patient_avg_map, label_basenames[:10], mm_col

def lookup_mm_per_px(mask_stem: str, exact_map: Dict[str, float], patient_avg_map: Dict[str, float]) -> Tuple[Optional[float], str, str]:
    image_stem = strip_mask_suffix(mask_stem)
    candidates = [image_stem + ext for ext in (".png", ".jpg", ".jpeg")]
    for c in candidates:
        if c in exact_map:
            return exact_map[c], c, "exact"

    startswith_hits = [k for k in exact_map.keys() if k.startswith(image_stem)]
    if startswith_hits:
        chosen = sorted(startswith_hits, key=len)[0]
        return exact_map[chosen], chosen, "startswith"

    m = re.match(r"^(Patient\d+)", image_stem, re.IGNORECASE)
    if m:
        pat = m.group(1)
        if pat in patient_avg_map:
            return patient_avg_map[pat], f"{pat} (avg)", "patient_avg"

    return None, image_stem + ".png", "missing"

def try_binarize(mask_gray: np.ndarray) -> np.ndarray:
    _t, bw = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(bw) < 0.05 * bw.size:
        _t, bw = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    return bw

def fit_head_ellipse_from_mask(mask_path: str):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    bw = try_binarize(m)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    areas = [cv2.contourArea(c) for c in cnts]
    max_idx = int(np.argmax(areas))
    if areas[max_idx] < 200:
        return None
    cnt = cnts[max_idx]
    if len(cnt) < 5:
        return None
    return cv2.fitEllipse(cnt)

def biometry_from_ellipse(ellipse, mm_per_px: float) -> Dict[str, float]:
    (_, _), (MA_px, ma_px), _ = ellipse
    major_px = max(MA_px, ma_px)
    minor_px = min(MA_px, ma_px)
    BPD_mm = minor_px * mm_per_px
    OFD_mm = major_px * mm_per_px
    a_px = major_px / 2.0
    b_px = minor_px / 2.0
    hc_px = math.pi * (3*(a_px + b_px) - math.sqrt((3*a_px + b_px) * (a_px + 3*b_px)))
    HC_mm = hc_px * mm_per_px
    a_mm = a_px * mm_per_px
    b_mm = b_px * mm_per_px
    area_mm2 = math.pi * a_mm * b_mm
    CI_pct = (BPD_mm / OFD_mm) * 100.0 if OFD_mm > 0 else float("nan")
    return {"BPD_mm": BPD_mm, "OFD_mm": OFD_mm, "HC_mm": HC_mm, "CI_pct": CI_pct, "SkullArea_mm2": area_mm2}

def extract_patient_id(s: str) -> str:
    m = re.search(r"(Patient\d+)", s, re.IGNORECASE)
    return m.group(1) if m else "Unknown"

def unique_sheetname(existing: set, base: str) -> str:
    name = base[:31] or "Sheet"
    if name not in existing:
        existing.add(name); return name
    i = 2
    while True:
        suffix = f"_{i}"
        trimmed = (base[:31-len(suffix)] + suffix) if len(base) + len(suffix) > 31 else base + suffix
        if trimmed not in existing:
            existing.add(trimmed); return trimmed
        i += 1

def main():
    ensure_exists(INPUT_ROOT)
    ensure_exists(PIXEL_CSV)
    ensure_dir(OUT_DIR)

    exact_map, patient_avg_map, sample_labels, mm_col = load_pixel_sizes(PIXEL_CSV)

    collected = []
    for sub, tri in TRIM_FOLDERS.items():
        subdir = os.path.join(INPUT_ROOT, sub)
        if not os.path.isdir(subdir):
            print(f"[WARN] Subfolder missing (skipped): {subdir}")
            continue
        masks = list_masks(subdir)
        for p in masks:
            collected.append((p, tri))

    print(f"[INFO] CSV pixel column: '{mm_col}'")
    print(f"[INFO] Loaded {len(exact_map)} label->mm entries. Sample: {sample_labels}")
    print(f"[INFO] Collected {len(collected)} mask files from {len(TRIM_FOLDERS)} subfolders.")

    rows = []
    reasons = {"pixel size missing":0, "ellipse fit failed":0, "ok":0}

    for mask_path, trimester in collected:
        mask_fname = os.path.basename(mask_path)
        mask_stem, _ = os.path.splitext(mask_fname)

        mm_per_px, used_label, source = lookup_mm_per_px(mask_stem, exact_map, patient_avg_map)
        pat_id = extract_patient_id(used_label) if source != "missing" else extract_patient_id(mask_fname)

        if mm_per_px is None:
            rows.append({
                "patient": pat_id,
                "image": used_label,
                "trimester": trimester,
                "pixel_mm": "",
                "BPD_mm": "", "OFD_mm": "", "HC_mm": "", "CI_pct": "", "SkullArea_mm2": "",
                "status": "SKIPPED: pixel size missing",
                "mm_source": source,
                "mask_path": mask_path
            })
            reasons["pixel size missing"] += 1
            continue

        ellipse = fit_head_ellipse_from_mask(mask_path)
        if ellipse is None:
            rows.append({
                "patient": pat_id,
                "image": used_label,
                "trimester": trimester,
                "pixel_mm": round(mm_per_px, 6),
                "BPD_mm": "", "OFD_mm": "", "HC_mm": "", "CI_pct": "", "SkullArea_mm2": "",
                "status": "SKIPPED: ellipse fit failed",
                "mm_source": source,
                "mask_path": mask_path
            })
            reasons["ellipse fit failed"] += 1
            continue

        bio = biometry_from_ellipse(ellipse, mm_per_px)
        rows.append({
            "patient": pat_id,
            "image": used_label,
            "trimester": trimester,
            "pixel_mm": round(mm_per_px, 6),
            "BPD_mm": round(bio["BPD_mm"], 1),
            "OFD_mm": round(bio["OFD_mm"], 1),
            "HC_mm":  round(bio["HC_mm"], 1),
            "CI_pct": round(bio["CI_pct"], 1),
            "SkullArea_mm2": round(bio["SkullArea_mm2"], 1),
            "status": "OK",
            "mm_source": source,
            "mask_path": mask_path
        })
        reasons["ok"] += 1

    cols = ["patient","image","trimester","pixel_mm","BPD_mm","OFD_mm","HC_mm","CI_pct","SkullArea_mm2","status","mm_source","mask_path"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved master CSV: {OUT_CSV}")

    patients = sorted(df["patient"].fillna("Unknown").unique().tolist()) if not df.empty else []
    sheetnames_used = set()
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=unique_sheetname(sheetnames_used, "Summary"), index=False)

    print(f"[INFO] Saved Excel workbook: {OUT_XLSX}")

    dbg = textwrap.dedent(f"""
    === DEBUG SUMMARY ===
    Input root: {INPUT_ROOT}
    CSV file:   {PIXEL_CSV}
    Pixel col:  {mm_col}

    Total masks collected: {len(collected)}
    Total rows written: {len(df)}

    Status counts:
      OK: {reasons['ok']}
      SKIPPED: pixel size missing: {reasons['pixel size missing']}
      SKIPPED: ellipse fit failed: {reasons['ellipse fit failed']}

    Trimester subfolders used: {TRIM_FOLDERS}
    """).strip()
    ensure_dir(OUT_DIR)
    with open(DEBUG_TXT, "w", encoding="utf-8") as f:
        f.write(dbg)
    print(dbg)

if __name__ == "__main__":
    main()
