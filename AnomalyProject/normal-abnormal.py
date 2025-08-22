# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# """
# WHO fetal growth comparison (per-patient, no Gaussian assumptions) — two-input version
#
# Inputs
# ------
# 1) Measurements Excel (e.g., tt_measurements.xlsx) with columns:
#    - patient (str)
#    - BPD_mm (float)   # biparietal diameter, outer–inner
#    - HC_mm  (float)   # head circumference
#    Optional passthrough: OFD_mm, CI_pct, SkullArea_mm2, image, trimester, pixel_mm, status, mm_source
#
# 2) GA CSV (e.g., ga_predictions_*.csv) with columns:
#    - patient (str)
#    - GA in weeks (column name auto-detected, or pass --ga_col)
#
# 3) WHO reference CSV (percentiles for integer GA), schema at bottom.
#
# What it does
# ------------
# • Merges Excel + GA CSV on 'patient'.
# • Interpolates WHO percentiles for BPD_OI & HC across percentiles AND between weeks.
# • Applies a 70/30 screen: inner [15,85] => NOT_AT_RISK; otherwise FLAGGED.
# • Optional CI guardrail: CI outside [74,83] => SHAPE_ALERT and upgrade to FLAGGED.
# • Writes a results CSV.
#
# Usage
# -----
# python who_compare_two_inputs.py \
#   --measures_xlsx tt_measurements.xlsx \
#   --ga_csv ga_predictions_20250821_144538.csv \
#   --who_csv who_hc_bpd_percentiles.csv \
#   --output_csv who_screened_patients.csv \
#   --ci_shape_check
# """
#
# import argparse
# import math
# import sys
# from typing import Dict, Tuple, Optional, List
#
# import numpy as np
# import pandas as pd
#
# # -----------------------------
# # Defaults & config
# # -----------------------------
# DEFAULT_MEASURES_XLSX = "tt_measurements.xlsx"
# DEFAULT_GA_CSV        = "ga_predictions.csv"
# DEFAULT_WHO_CSV       = "who_hc_bpd_percentiles.csv"
# DEFAULT_OUTPUT_CSV    = "who_screened_patients.csv"
#
# # 70/30 split
# INNER_LOW  = 15.0
# INNER_HIGH = 85.0
#
# # Optional CI band
# CI_LOW  = 74.0
# CI_HIGH = 83.0
#
# # Expected columns in measurements Excel
# COL_PATIENT   = "patient"
# COL_BPD       = "BPD_mm"
# COL_HC        = "HC_mm"
# COL_OFD       = "OFD_mm"
# COL_CI        = "CI_pct"
# COL_SKULLAREA = "SkullArea_mm2"
#
# # WHO measure labels
# WHO_MEASURE_BPD = "BPD_OI"
# WHO_MEASURE_HC  = "HC"
#
# # WHO percentiles included in the reference CSV
# WHO_PCT_COLS = ["p1","p5","p10","p25","p50","p75","p90","p95","p99"]
#
# # -----------------------------
# # WHO helpers (percentile math)
# # -----------------------------
# def _closest_two_weeks(avail_weeks: np.ndarray, ga_w: float) -> Tuple[int, int, float]:
#     lo_week = int(math.floor(ga_w))
#     hi_week = int(math.ceil(ga_w))
#     if lo_week < avail_weeks.min():
#         lo_week = hi_week = int(avail_weeks.min())
#     if hi_week > avail_weeks.max():
#         hi_week = lo_week = int(avail_weeks.max())
#     if lo_week == hi_week:
#         return lo_week, hi_week, 0.0
#     w = (ga_w - lo_week) / (hi_week - lo_week)
#     return lo_week, hi_week, float(w)
#
# def _value_to_percentile(bracket: Dict[float, float], value: float) -> float:
#     pts = sorted((float(p), float(v)) for p, v in bracket.items())
#     # edges: linear extrapolation
#     if value <= pts[0][1]:
#         p1, v1 = pts[0]; p2, v2 = pts[1]
#         if v2 == v1: return p1
#         frac = (value - v1) / (v2 - v1)
#         return max(0.0, p1 + frac * (p2 - p1))
#     if value >= pts[-1][1]:
#         p1, v1 = pts[-2]; p2, v2 = pts[-1]
#         if v2 == v1: return p2
#         frac = (value - v1) / (v2 - v1)
#         return min(100.0, p1 + frac * (p2 - p1))
#     # inside range
#     for i in range(len(pts) - 1):
#         p_lo, v_lo = pts[i]; p_hi, v_hi = pts[i + 1]
#         if v_lo <= value <= v_hi:
#             if v_hi == v_lo: return (p_lo + p_hi) / 2.0
#             frac = (value - v_lo) / (v_hi - v_lo)
#             return p_lo + frac * (p_hi - p_lo)
#     return 50.0
#
# def _blend_rows(row_lo: pd.Series, row_hi: pd.Series, w: float) -> Dict[float, float]:
#     blended = {}
#     for pc in WHO_PCT_COLS:
#         v_lo = float(row_lo[pc]); v_hi = float(row_hi[pc])
#         blended[float(pc.strip("p"))] = (1.0 - w) * v_lo + w * v_hi
#     return blended
#
# def percentile_for_measure(who_df: pd.DataFrame, measure: str, ga_weeks: float, value_mm: Optional[float]) -> Optional[float]:
#     if value_mm is None or pd.isna(value_mm):
#         return None
#     dfm = who_df[who_df["measure"] == measure]
#     if dfm.empty:
#         return None
#     avail_weeks = dfm["ga_weeks"].to_numpy()
#     lo_w, hi_w, w = _closest_two_weeks(avail_weeks, float(ga_weeks))
#     row_lo = dfm[dfm["ga_weeks"] == lo_w]
#     if row_lo.empty: return None
#     row_lo = row_lo.iloc[0]
#     if lo_w == hi_w:
#         bracket = {float(pc.strip("p")): float(row_lo[pc]) for pc in WHO_PCT_COLS}
#         return _value_to_percentile(bracket, float(value_mm))
#     row_hi = dfm[dfm["ga_weeks"] == hi_w]
#     if row_hi.empty:
#         bracket = {float(pc.strip("p")): float(row_lo[pc]) for pc in WHO_PCT_COLS}
#         return _value_to_percentile(bracket, float(value_mm))
#     bracket = _blend_rows(row_lo, row_hi.iloc[0], w)
#     return _value_to_percentile(bracket, float(value_mm))
#
# def flag_7030(pct: Optional[float]) -> str:
#     if pct is None or pd.isna(pct): return "UNKNOWN"
#     return "NOT_AT_RISK" if (INNER_LOW <= pct <= INNER_HIGH) else "FLAGGED"
#
# def ci_shape_flag(ci_pct: Optional[float]) -> str:
#     if ci_pct is None or pd.isna(ci_pct): return "UNKNOWN"
#     return "NORMAL_SHAPE" if (CI_LOW <= ci_pct <= CI_HIGH) else "SHAPE_ALERT"
#
# # -----------------------------
# # GA CSV handling
# # -----------------------------
# GA_NAME_CANDIDATES: List[str] = [
#     "GA_weeks","ga_weeks","GA","ga","gestational_age_weeks","gest_age_weeks",
#     "pred_ga_weeks","ga_pred_weeks","gest_weeks"
# ]
#
# def _infer_ga_column(df_ga: pd.DataFrame) -> Optional[str]:
#     # 1) named candidates
#     for c in GA_NAME_CANDIDATES:
#         if c in df_ga.columns:
#             return c
#     # 2) numeric heuristic: pick a numeric column with values mostly in [10,45]
#     numeric_cols = df_ga.select_dtypes(include=["number"]).columns.tolist()
#     for c in numeric_cols:
#         s = pd.to_numeric(df_ga[c], errors="coerce")
#         valid = s.between(10, 45).mean()  # share of values in plausible GA range
#         if valid > 0.5:
#             return c
#     return None
#
# def _deduplicate_ga(df_ga: pd.DataFrame, patient_col: str, ga_col: str) -> pd.DataFrame:
#     # If multiple rows per patient exist, prefer the one with highest 'confidence'/'prob' if present.
#     pref_cols = [c for c in df_ga.columns if c.lower() in ("confidence","prob","probability","score")]
#     if pref_cols:
#         key_col = pref_cols[0]
#         idx = df_ga.groupby(patient_col)[key_col].idxmax()
#         return df_ga.loc[idx].copy()
#     # Else keep the first occurrence per patient
#     return df_ga.drop_duplicates(subset=[patient_col], keep="first").copy()
#
# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser(description="Compare BPD_OI & HC to WHO percentiles with GA from a separate CSV.")
#     ap.add_argument("--measures_xlsx", default=DEFAULT_MEASURES_XLSX, help="Measurements Excel (no GA)")
#     ap.add_argument("--ga_csv",        default=DEFAULT_GA_CSV,        help="GA predictions CSV (has GA weeks)")
#     ap.add_argument("--who_csv",       default=DEFAULT_WHO_CSV,       help="WHO percentiles CSV (see schema below)")
#     ap.add_argument("--output_csv",    default=DEFAULT_OUTPUT_CSV,    help="Where to write results")
#     ap.add_argument("--ga_col",        default=None,                  help="Name of GA column in GA CSV (override autodetect)")
#     ap.add_argument("--ci_shape_check", action="store_true",          help="Flag CI outside [74,83] as SHAPE_ALERT")
#     args = ap.parse_args()
#
#     # Load measurements
#     try:
#         df_meas = pd.read_excel(args.measures_xlsx)
#     except Exception as e:
#         print(f"[ERROR] Failed to read Excel: {args.measures_xlsx}\n{e}", file=sys.stderr)
#         sys.exit(2)
#
#     if COL_PATIENT not in df_meas.columns:
#         print(f"[ERROR] Measurements file missing '{COL_PATIENT}' column.", file=sys.stderr)
#         sys.exit(2)
#
#     # Load GA CSV
#     try:
#         df_ga = pd.read_csv(args.ga_csv)
#     except Exception as e:
#         print(f"[ERROR] Failed to read GA CSV: {args.ga_csv}\n{e}", file=sys.stderr)
#         sys.exit(3)
#
#     if COL_PATIENT not in df_ga.columns:
#         print(f"[ERROR] GA CSV missing '{COL_PATIENT}' column.", file=sys.stderr)
#         sys.exit(3)
#
#     ga_col = args.ga_col or _infer_ga_column(df_ga)
#     if not ga_col:
#         print(f"[ERROR] Could not find GA column automatically. Use --ga_col to specify.", file=sys.stderr)
#         sys.exit(3)
#
#     # Deduplicate GA rows per patient if needed
#     df_ga_clean = _deduplicate_ga(df_ga[[COL_PATIENT, ga_col]].copy(), COL_PATIENT, ga_col)
#     # Coerce GA to numeric
#     df_ga_clean[ga_col] = pd.to_numeric(df_ga_clean[ga_col], errors="coerce")
#
#     # Merge on 'patient'
#     df = df_meas.merge(df_ga_clean.rename(columns={ga_col: "GA_weeks"}), on=COL_PATIENT, how="left")
#
#     # Load WHO reference
#     try:
#         if args.who_csv.endswith(".xlsx"):
#             who = pd.read_excel(args.who_csv)
#         else:
#             who = pd.read_csv(args.who_csv)
#     except Exception as e:
#         print(f"[ERROR] Failed to read WHO CSV: {args.who_csv}\n{e}", file=sys.stderr)
#         sys.exit(4)
#
#     must = {"measure", "ga_weeks", *WHO_PCT_COLS}
#     missing = must - set(who.columns)
#     if missing:
#         print(f"[ERROR] WHO CSV missing columns: {sorted(missing)}", file=sys.stderr)
#         sys.exit(4)
#     who["ga_weeks"] = who["ga_weeks"].astype(int)
#
#     # Process rows
#     out_rows = []
#     for i, r in df.iterrows():
#         patient = r.get(COL_PATIENT, f"row{i}")
#         ga      = r.get("GA_weeks", np.nan)
#         bpd     = r.get(COL_BPD, np.nan)
#         hc      = r.get(COL_HC,  np.nan)
#         ofd     = r.get(COL_OFD, np.nan)
#         ci      = r.get(COL_CI,  np.nan)
#         skull   = r.get(COL_SKULLAREA, np.nan)
#
#         if pd.isna(ga):
#             out_rows.append({
#                 "patient": patient, "GA_weeks": ga,
#                 "BPD_mm": bpd, "HC_mm": hc, "OFD_mm": ofd, "CI_pct": ci, "SkullArea_mm2": skull,
#                 "BPD_percentile": None, "HC_percentile": None,
#                 "BPD_flag": "UNKNOWN", "HC_flag": "UNKNOWN",
#                 "CI_shape_flag": "UNKNOWN" if args.ci_shape_check else None,
#                 "Final_screen": "UNKNOWN_GA"
#             })
#             continue
#
#         p_bpd = percentile_for_measure(who, WHO_MEASURE_BPD, ga, bpd)
#         p_hc  = percentile_for_measure(who, WHO_MEASURE_HC,  ga, hc)
#
#         f_bpd = flag_7030(p_bpd)
#         f_hc  = flag_7030(p_hc)
#
#         known = [f for f in (f_bpd, f_hc) if f != "UNKNOWN"]
#         final = "UNKNOWN" if not known else ("FLAGGED" if "FLAGGED" in known else "NOT_AT_RISK")
#
#         ci_flag = ci_shape_flag(ci) if args.ci_shape_check else None
#         if args.ci_shape_check and ci_flag == "SHAPE_ALERT" and final != "UNKNOWN":
#             final = "FLAGGED"
#
#         out_rows.append({
#             "patient": patient,
#             "GA_weeks": ga,
#             "BPD_mm": bpd,
#             "HC_mm": hc,
#             "OFD_mm": ofd,
#             "CI_pct": ci,
#             "SkullArea_mm2": skull,
#             "BPD_percentile": None if p_bpd is None else round(p_bpd, 2),
#             "HC_percentile":  None if p_hc  is None else round(p_hc,  2),
#             "BPD_flag": f_bpd,
#             "HC_flag":  f_hc,
#             "CI_shape_flag": ci_flag,
#             "Final_screen": final
#         })
#
#     out_df = pd.DataFrame(out_rows)
#     out_df.to_csv(args.output_csv, index=False)
#     print(f"[OK] Wrote {len(out_df)} rows to {args.output_csv}")
#
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHO fetal growth comparison (per-patient, no Gaussian assumptions) — PyCharm run
• Auto-normalizes WHO reference tables from pmed.1002220.s007.xlsx (and similar).
• Accepts GA CSV with `patient` or `mask_filename`.

Runs with hardcoded paths if no CLI args are provided.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, sys, math
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# =============================
# 🔧 HARDCODED PATHS (edit here)
# =============================
HARDCODED_MEASURES_XLSX = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\tt_reports\tt_measurements.xlsx"
HARDCODED_GA_CSV        = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\sorted\ga_predictions_20250821_144538.csv"
HARDCODED_WHO_FILE      = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\pmed.1002220.s007.xlsx"
HARDCODED_OUTPUT_CSV    = r"C:\Users\Lenovo\Desktop\Brain-Anomaly-Detection\AnomalyProject\who_screened_patients.csv"
RUN_WITH_HARDCODED_IF_NO_ARGS = True

# -----------------------------
# Defaults & config
# -----------------------------
DEFAULT_MEASURES_XLSX = "tt_measurements.xlsx"
DEFAULT_GA_CSV        = "ga_predictions.csv"
DEFAULT_WHO_CSV       = "who_hc_bpd_percentiles.csv"
DEFAULT_OUTPUT_CSV    = "who_screened_patients.csv"

INNER_LOW, INNER_HIGH = 15.0, 85.0
CI_LOW, CI_HIGH = 74.0, 83.0

COL_PATIENT   = "patient"
COL_BPD       = "BPD_mm"
COL_HC        = "HC_mm"
COL_OFD       = "OFD_mm"
COL_CI        = "CI_pct"
COL_SKULLAREA = "SkullArea_mm2"

WHO_MEASURE_BPD = "BPD_OI"
WHO_MEASURE_HC  = "HC"
WHO_PCTS = [1,5,10,25,50,75,90,95,99]

# -----------------------------
# Safe strings / utils
# -----------------------------
def _s(x) -> str:
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    return str(x)

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', _s(s).strip().lower())

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

# -----------------------------
# Percentile math
# -----------------------------
def _closest_two_weeks(avail_weeks: np.ndarray, ga_w: float) -> Tuple[int, int, float]:
    lo_week = int(math.floor(ga_w))
    hi_week = int(math.ceil(ga_w))
    if lo_week < avail_weeks.min():
        lo_week = hi_week = int(avail_weeks.min())
    if hi_week > avail_weeks.max():
        hi_week = lo_week = int(avail_weeks.max())
    if lo_week == hi_week:
        return lo_week, hi_week, 0.0
    w = (ga_w - lo_week) / (hi_week - lo_week)
    return lo_week, hi_week, float(w)

def _value_to_percentile(bracket: Dict[float, float], value: float) -> float:
    pts = sorted((float(p), float(v)) for p, v in bracket.items())
    if value <= pts[0][1]:
        p1, v1 = pts[0]; p2, v2 = pts[1]
        if v2 == v1: return p1
        frac = (value - v1) / (v2 - v1)
        return max(0.0, p1 + frac * (p2 - p1))
    if value >= pts[-1][1]:
        p1, v1 = pts[-2]; p2, v2 = pts[-1]
        if v2 == v1: return p2
        frac = (value - v1) / (v2 - v1)
        return min(100.0, p1 + frac * (p2 - p1))
    for i in range(len(pts) - 1):
        p_lo, v_lo = pts[i]; p_hi, v_hi = pts[i + 1]
        if v_lo <= value <= v_hi:
            if v_hi == v_lo: return (p_lo + p_hi) / 2.0
            frac = (value - v_lo) / (v_hi - v_lo)
            return p_lo + frac * (p_hi - p_lo)
    return 50.0

def _blend_rows(row_lo: pd.Series, row_hi: pd.Series, w: float) -> Dict[float, float]:
    return {float(p): (1.0 - w) * float(row_lo[f"p{p}"]) + w * float(row_hi[f"p{p}"]) for p in WHO_PCTS}

def percentile_for_measure(who_df: pd.DataFrame, measure: str, ga_weeks: float, value_mm: Optional[float]) -> Optional[float]:
    if value_mm is None or pd.isna(value_mm): return None
    dfm = who_df[who_df["measure"] == measure]
    if dfm.empty: return None
    avail_weeks = dfm["ga_weeks"].to_numpy()
    lo_w, hi_w, w = _closest_two_weeks(avail_weeks, float(ga_weeks))
    row_lo = dfm[dfm["ga_weeks"] == lo_w]
    if row_lo.empty: return None
    row_lo = row_lo.iloc[0]
    if lo_w == hi_w:
        bracket = {float(p): float(row_lo[f"p{p}"]) for p in WHO_PCTS}
        return _value_to_percentile(bracket, float(value_mm))
    row_hi = dfm[dfm["ga_weeks"] == hi_w]
    if row_hi.empty:
        bracket = {float(p): float(row_lo[f"p{p}"]) for p in WHO_PCTS}
        return _value_to_percentile(bracket, float(value_mm))
    bracket = _blend_rows(row_lo, row_hi.iloc[0], w)
    return _value_to_percentile(bracket, float(value_mm))

def flag_7030(pct: Optional[float]) -> str:
    if pct is None or pd.isna(pct): return "UNKNOWN"
    return "NOT_AT_RISK" if (INNER_LOW <= pct <= INNER_HIGH) else "FLAGGED"

def ci_shape_flag(ci_pct: Optional[float]) -> str:
    if ci_pct is None or pd.isna(ci_pct): return "UNKNOWN"
    return "NORMAL_SHAPE" if (CI_LOW <= ci_pct <= CI_HIGH) else "SHAPE_ALERT"

# -----------------------------
# GA CSV handling
# -----------------------------
GA_NAME_CANDIDATES: List[str] = [
    "GA_weeks","ga_weeks","GA","ga","gestational_age_weeks","gest_age_weeks",
    "pred_ga_weeks","ga_pred_weeks","gest_weeks"
]

def _infer_ga_column(df_ga: pd.DataFrame) -> Optional[str]:
    for c in GA_NAME_CANDIDATES:
        if c in df_ga.columns: return c
    numeric_cols = df_ga.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        s = pd.to_numeric(df_ga[c], errors="coerce")
        if s.between(10, 45).mean() > 0.5:
            return c
    return None

def _deduplicate_ga(df_ga: pd.DataFrame, patient_col: str, ga_col: str) -> pd.DataFrame:
    pref_cols = [c for c in df_ga.columns if _s(c).lower() in ("confidence","prob","probability","score")]
    if pref_cols:
        key_col = pref_cols[0]
        idx = df_ga.groupby(patient_col)[key_col].idxmax()
        return df_ga.loc[idx].copy()
    return df_ga.drop_duplicates(subset=[patient_col], keep="first").copy()

_PATIENT_RES = [
    re.compile(r"^(Patient\d+)_", re.IGNORECASE),
    re.compile(r"^(Patient\d+)(?:_Plane|$)", re.IGNORECASE),
]
def _patient_from_mask_filename(name: str) -> str:
    base = os.path.basename(_s(name))
    stem = os.path.splitext(base)[0]
    for rx in _PATIENT_RES:
        m = rx.search(stem)
        if m: return m.group(1)
    if "_Plane" in stem: return stem.split("_Plane", 1)[0]
    if "_" in stem: return stem.split("_", 1)[0]
    return stem

def _ensure_patient_column(df_ga: pd.DataFrame) -> pd.DataFrame:
    if COL_PATIENT in df_ga.columns: return df_ga
    mask_col = next((c for c in df_ga.columns if _s(c).lower()=="mask_filename"), None)
    if not mask_col:
        raise ValueError(f"GA CSV needs '{COL_PATIENT}' or 'mask_filename' column.")
    out = df_ga.copy()
    out[COL_PATIENT] = out[mask_col].apply(_patient_from_mask_filename)
    return out

# -----------------------------
# WHO loader (VERY robust) + optional manual hints
# -----------------------------
_PCT_SYNONYM_RX = re.compile(
    r'(?:(?:^|[^a-z0-9])p\s*(\d{1,2})\b)|\b(\d{1,2})(?:st|nd|rd|th)\b|\bperc(?:ent(?:ile)?)?\s*(\d{1,2})\b',
    re.I
)

def _extract_pct_from_col(col) -> Optional[int]:
    text = _s(col)
    m = _PCT_SYNONYM_RX.search(text)
    if m:
        for g in m.groups():
            if g is not None:
                try: return int(g)
                except: pass
    m2 = re.search(r'(\d{1,2})\s*$', text)
    if m2:
        try: return int(m2.group(1))
        except: return None
    return None

def _flatten_columns(df: pd.DataFrame) -> None:
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for tup in df.columns.tolist():
            parts = [p for p in map(_s, tup) if _s(p) and _s(p).lower() != "nan"]
            flat.append(" ".join(parts).strip())
        df.columns = flat
    else:
        df.columns = [_s(c) for c in df.columns]

def _guess_ga_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        cn = _norm(c)
        if any(k in cn for k in ["ga", "gest", "age"]) and any(w in cn for w in ["week","weeks","wk","wks"]):
            return c
    return None

def _find_ga_by_numbers(df: pd.DataFrame) -> Optional[str]:
    # Find a column that looks like 14..42 weeks (mostly integers, in-range)
    best_col, best_score = None, -1.0
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 6:  # need at least 6 data points
            continue
        in_range = s.between(10, 45)
        frac = in_range.mean()
        # prefer columns with many unique ints in the range (suggesting a GA axis)
        uniq_ints = s[in_range].round().dropna().astype(int).nunique()
        score = frac + 0.02 * uniq_ints
        if score > best_score:
            best_col, best_score = c, score
    return best_col if best_score >= 0.6 else None

def _detect_measure_from_context(col: str, sheet_name: Optional[str]) -> Optional[str]:
    c = _norm(col)
    sn = _norm(sheet_name or "")
    if "hc" in c or "head circum" in c or "hc" in sn or "head circum" in sn:
        return WHO_MEASURE_HC
    if "bpd oi" in c or "bpd" in c or "bpd" in sn:
        return WHO_MEASURE_BPD
    return None

def _normalize_one_sheet(df: pd.DataFrame, sheet_name: Optional[str]) -> pd.DataFrame:
    _flatten_columns(df)
    ga_col_name = _guess_ga_col(list(df.columns))
    if ga_col_name is None:
        ga_col_name = _find_ga_by_numbers(df)
    if ga_col_name is None:
        return pd.DataFrame()

    out_rows = []
    for col in df.columns:
        if col == ga_col_name: continue
        measure = _detect_measure_from_context(col, sheet_name)
        pct = _extract_pct_from_col(col)
        if measure and pct in WHO_PCTS:
            sub = df[[ga_col_name, col]].copy()
            sub.columns = ["ga_weeks", "value"]
            sub["ga_weeks"] = pd.to_numeric(sub["ga_weeks"], errors="coerce").astype("Int64")
            sub = sub.dropna(subset=["ga_weeks"]).copy()
            sub["measure"] = measure
            sub["pct"] = pct
            out_rows.append(sub)

    if not out_rows:
        return pd.DataFrame()

    long_df = pd.concat(out_rows, ignore_index=True)
    wide = long_df.pivot_table(index=["measure","ga_weeks"], columns="pct", values="value", aggfunc="first").reset_index()
    for p in WHO_PCTS:
        if p not in wide.columns:
            wide[p] = np.nan
    wide = wide[["measure","ga_weeks"] + WHO_PCTS].copy()
    wide = wide.rename(columns={p: f"p{p}" for p in WHO_PCTS})
    wide["ga_weeks"] = wide["ga_weeks"].astype(int)
    wide = wide.sort_values(["measure","ga_weeks"]).reset_index(drop=True)
    return wide

def _profile_workbook(path: str) -> None:
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WHO DEBUG] Could not open workbook: {e}", file=sys.stderr)
        return
    print("[WHO DEBUG] Sheets found:", ", ".join(xl.sheet_names))
    for name in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=name, nrows=5, header=None)
            headers = list(map(_s, df.iloc[0].tolist()))
            print(f"[WHO DEBUG] Sheet '{name}' – first row (possible header):")
            print("   ", headers[:10])
        except Exception as e:
            print(f"[WHO DEBUG] Failed to read preview of sheet '{name}': {e}", file=sys.stderr)

def load_who_reference(path: str,
                       sheet_hc: Optional[str]=None, sheet_bpd: Optional[str]=None,
                       ga_col_hc: Optional[str]=None, ga_col_bpd: Optional[str]=None) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        _flatten_columns(df)
        must = {"measure","ga_weeks", *{f"p{p}" for p in WHO_PCTS}}
        if must.issubset(set(df.columns)):
            df["ga_weeks"] = pd.to_numeric(df["ga_weeks"], errors="coerce").astype(int)
            return df[["measure","ga_weeks"]+[f"p{p}" for p in WHO_PCTS]].copy()
        # Treat as a single-sheet table to normalize
        norm = _normalize_one_sheet(df, sheet_name=None)
        if not norm.empty:
            return norm
        raise ValueError("CSV WHO reference is not in canonical format and could not be normalized.")

    sheets = pd.read_excel(path, sheet_name=None, header=0)
    collected = []

    # If manual hints are provided, try those first (very tolerant to headers)
    def _normalize_with_hints(sname: str, ga_col_hint: Optional[str], measure: str) -> pd.DataFrame:
        if sname not in sheets:
            return pd.DataFrame()
        raw = sheets[sname].copy()
        _flatten_columns(raw)
        # pick GA
        ga = ga_col_hint if (ga_col_hint and ga_col_hint in raw.columns) else (_guess_ga_col(list(raw.columns)) or _find_ga_by_numbers(raw))
        if ga is None:
            return pd.DataFrame()
        out_rows = []
        for col in raw.columns:
            if col == ga: continue
            pct = _extract_pct_from_col(col)
            if pct in WHO_PCTS:
                sub = raw[[ga, col]].copy()
                sub.columns = ["ga_weeks", "value"]
                sub["ga_weeks"] = pd.to_numeric(sub["ga_weeks"], errors="coerce").astype("Int64")
                sub = sub.dropna(subset=["ga_weeks"]).copy()
                sub["measure"] = measure
                sub["pct"] = pct
                out_rows.append(sub)
        if not out_rows:
            return pd.DataFrame()
        long_df = pd.concat(out_rows, ignore_index=True)
        wide = long_df.pivot_table(index=["measure","ga_weeks"], columns="pct", values="value", aggfunc="first").reset_index()
        for p in WHO_PCTS:
            if p not in wide.columns: wide[p] = np.nan
        wide = wide[["measure","ga_weeks"] + WHO_PCTS].rename(columns={p: f"p{p}" for p in WHO_PCTS})
        wide["ga_weeks"] = wide["ga_weeks"].astype(int)
        return wide.sort_values(["measure","ga_weeks"]).reset_index(drop=True)

    if sheet_hc or sheet_bpd:
        if sheet_hc:
            w_hc = _normalize_with_hints(sheet_hc, ga_col_hc, WHO_MEASURE_HC)
            if not w_hc.empty: collected.append(w_hc)
        if sheet_bpd:
            w_bpd = _normalize_with_hints(sheet_bpd, ga_col_bpd, WHO_MEASURE_BPD)
            if not w_bpd.empty: collected.append(w_bpd)

    # Auto mode: try all sheets with context-aware detection
    for name, sdf in sheets.items():
        normed = _normalize_one_sheet(sdf.copy(), sheet_name=name)
        if not normed.empty:
            collected.append(normed)
        else:
            # Retry with unknown header depth
            raw = pd.read_excel(path, sheet_name=name, header=None)
            # heuristic: choose first “dense” row as header (>= 4 non-empty)
            header_row_idx = None
            for i in range(min(15, len(raw))):
                row_vals = list(map(_s, raw.iloc[i].tolist()))
                non_empty = sum(1 for v in row_vals if _s(v))
                row_join = " ".join(row_vals).lower()
                looks_ga = any(k in row_join for k in ["ga","gest","age"]) and any(w in row_join for w in ["week","weeks","wk","wks"])
                if non_empty >= 4 or looks_ga:
                    header_row_idx = i
                    break
            if header_row_idx is not None:
                hdr = list(map(_s, raw.iloc[header_row_idx].tolist()))
                sdf2 = raw.iloc[header_row_idx+1:].copy()
                sdf2.columns = hdr
                normed2 = _normalize_one_sheet(sdf2, sheet_name=name)
                if not normed2.empty:
                    collected.append(normed2)

    if not collected:
        _profile_workbook(path)
        raise ValueError("Could not normalize WHO reference: no recognizable sheets/columns found.")

    who = pd.concat(collected, ignore_index=True).drop_duplicates(["measure","ga_weeks"]).reset_index(drop=True)
    needed = {"measure","ga_weeks", *{f"p{p}" for p in WHO_PCTS}}
    missing = needed - set(who.columns)
    if missing:
        raise ValueError(f"WHO reference missing columns after normalization: {sorted(missing)}")
    who["ga_weeks"] = who["ga_weeks"].astype(int)
    return who[["measure","ga_weeks"]+[f"p{p}" for p in WHO_PCTS]].copy()

# -----------------------------
# CLI & main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare BPD_OI & HC to WHO percentiles with GA from a separate CSV.")
    ap.add_argument("--measures_xlsx", default=DEFAULT_MEASURES_XLSX)
    ap.add_argument("--ga_csv",        default=DEFAULT_GA_CSV)
    ap.add_argument("--who_csv",       default=DEFAULT_WHO_CSV)
    ap.add_argument("--output_csv",    default=DEFAULT_OUTPUT_CSV)
    ap.add_argument("--ga_col",        default=None)
    ap.add_argument("--ci_shape_check", action="store_true")
    # 👇 Optional manual hints for tricky WHO files
    ap.add_argument("--who_sheet_hc",  default=None, help="Exact sheet name containing HC percentiles")
    ap.add_argument("--who_sheet_bpd", default=None, help="Exact sheet name containing BPD percentiles")
    ap.add_argument("--who_ga_col_hc",  default=None, help="GA column header in HC sheet (optional)")
    ap.add_argument("--who_ga_col_bpd", default=None, help="GA column header in BPD sheet (optional)")
    return ap

def _resolve_args_for_run(args: argparse.Namespace) -> argparse.Namespace:
    if RUN_WITH_HARDCODED_IF_NO_ARGS and len(sys.argv) == 1:
        args.measures_xlsx = HARDCODED_MEASURES_XLSX
        args.ga_csv        = HARDCODED_GA_CSV
        args.who_csv       = HARDCODED_WHO_FILE
        args.output_csv    = HARDCODED_OUTPUT_CSV
        if not getattr(args, "ci_shape_check", False):
            args.ci_shape_check = True
        print("[INFO] Running with hardcoded paths (no CLI args detected).")
    else:
        print("[INFO] Running with CLI arguments (or hardcoded bypass disabled).")
    return args

def main():
    ap = build_parser()
    args = ap.parse_args()
    args = _resolve_args_for_run(args)

    print("[INFO] Inputs:")
    print(f"       measures_xlsx = {args.measures_xlsx}")
    print(f"       ga_csv        = {args.ga_csv}")
    print(f"       who_csv       = {args.who_csv}")
    print(f"       output_csv    = {args.output_csv}")
    print(f"       ci_shape_check= {getattr(args, 'ci_shape_check', False)}")
    if args.who_sheet_hc or args.who_sheet_bpd:
        print(f"       who_sheet_hc  = {args.who_sheet_hc}")
        print(f"       who_sheet_bpd = {args.who_sheet_bpd}")
        print(f"       who_ga_col_hc = {args.who_ga_col_hc}")
        print(f"       who_ga_col_bpd= {args.who_ga_col_bpd}")

    # Measurements
    try:
        df_meas = pd.read_excel(args.measures_xlsx)
    except Exception as e:
        print(f"[ERROR] Failed to read Excel: {args.measures_xlsx}\n{e}", file=sys.stderr); sys.exit(2)
    if COL_PATIENT not in df_meas.columns:
        print(f"[ERROR] Measurements file missing '{COL_PATIENT}' column.", file=sys.stderr); sys.exit(2)

    # GA CSV
    try:
        df_ga_raw = pd.read_csv(args.ga_csv)
    except Exception as e:
        print(f"[ERROR] Failed to read GA CSV: {args.ga_csv}\n{e}", file=sys.stderr); sys.exit(3)
    try:
        df_ga = _ensure_patient_column(df_ga_raw)
    except ValueError as ve:
        print(f"[ERROR] {ve}", file=sys.stderr); sys.exit(3)

    ga_col = args.ga_col or _infer_ga_column(df_ga)
    if not ga_col:
        print(f"[ERROR] Could not find GA column automatically. Use --ga_col to specify.", file=sys.stderr); sys.exit(3)

    df_ga_clean = _deduplicate_ga(df_ga[[COL_PATIENT, ga_col]].copy(), COL_PATIENT, ga_col)
    df_ga_clean[ga_col] = pd.to_numeric(df_ga_clean[ga_col], errors="coerce")

    df = df_meas.merge(df_ga_clean.rename(columns={ga_col: "GA_weeks"}), on=COL_PATIENT, how="left")

    # WHO reference
    try:
        who = load_who_reference(args.who_csv,
                                 sheet_hc=args.who_sheet_hc, sheet_bpd=args.who_sheet_bpd,
                                 ga_col_hc=args.who_ga_col_hc, ga_col_bpd=args.who_ga_col_bpd)
    except Exception as e:
        print(f"[ERROR] WHO reference normalization failed: {e}", file=sys.stderr); sys.exit(4)

    # Process
    out_rows = []
    for i, r in df.iterrows():
        patient = r.get(COL_PATIENT, f"row{i}")
        ga      = r.get("GA_weeks", np.nan)
        bpd     = r.get(COL_BPD, np.nan)
        hc      = r.get(COL_HC,  np.nan)
        ofd     = r.get(COL_OFD, np.nan)
        ci      = r.get(COL_CI,  np.nan)
        skull   = r.get(COL_SKULLAREA, np.nan)

        if pd.isna(ga):
            out_rows.append({
                "patient": patient, "GA_weeks": ga,
                "BPD_mm": bpd, "HC_mm": hc, "OFD_mm": ofd, "CI_pct": ci, "SkullArea_mm2": skull,
                "BPD_percentile": None, "HC_percentile": None,
                "BPD_flag": "UNKNOWN", "HC_flag": "UNKNOWN",
                "CI_shape_flag": "UNKNOWN" if args.ci_shape_check else None,
                "Final_screen": "UNKNOWN_GA"
            })
            continue

        p_bpd = percentile_for_measure(who, WHO_MEASURE_BPD, ga, bpd)
        p_hc  = percentile_for_measure(who, WHO_MEASURE_HC,  ga, hc)

        f_bpd = flag_7030(p_bpd)
        f_hc  = flag_7030(p_hc)

        known = [f for f in (f_bpd, f_hc) if f != "UNKNOWN"]
        final = "UNKNOWN" if not known else ("FLAGGED" if "FLAGGED" in known else "NOT_AT_RISK")

        ci_flag = ci_shape_flag(ci) if args.ci_shape_check else None
        if args.ci_shape_check and ci_flag == "SHAPE_ALERT" and final != "UNKNOWN":
            final = "FLAGGED"

        out_rows.append({
            "patient": patient,
            "GA_weeks": ga,
            "BPD_mm": bpd,
            "HC_mm": hc,
            "OFD_mm": ofd,
            "CI_pct": ci,
            "SkullArea_mm2": skull,
            "BPD_percentile": None if p_bpd is None else round(p_bpd, 2),
            "HC_percentile":  None if p_hc  is None else round(p_hc,  2),
            "BPD_flag": f_bpd,
            "HC_flag":  f_hc,
            "CI_shape_flag": ci_flag,
            "Final_screen": final
        })

    out_df = pd.DataFrame(out_rows)
    _ensure_parent_dir(args.output_csv)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[OK] Wrote {len(out_df)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
