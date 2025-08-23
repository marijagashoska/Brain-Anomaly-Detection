import json
import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)


def outside_band(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = arr.astype(float)
    return np.clip(low - arr, 0, None) + np.clip(arr - high, 0, None)


def ci_outside(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = arr.astype(float)
    return np.clip(low - arr, 0, None) + np.clip(arr - high, 0, None)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))


def interp_percentile_from_quantiles(x: float, qs: list, vals: list) -> float:
    if len(qs) != len(vals) or len(qs) < 2:
        return np.nan
    pairs = sorted(zip(vals, qs))
    vals_sorted, qs_sorted = zip(*pairs)
    if x <= vals_sorted[0]:
        return 100.0 * qs_sorted[0]
    if x >= vals_sorted[-1]:
        return 100.0 * qs_sorted[-1]
    for i in range(len(vals_sorted) - 1):
        v0, v1 = vals_sorted[i], vals_sorted[i+1]
        if v0 <= x <= v1:
            q0, q1 = qs_sorted[i], qs_sorted[i+1]
            if v1 == v0:
                return 100.0 * q1
            t = (x - v0) / (v1 - v0)
            q = q0 + t * (q1 - q0)
            return 100.0 * q
    return np.nan


def compute_percentiles_from_who(df: pd.DataFrame, who_ref: pd.DataFrame) -> pd.DataFrame:
    if {"GA_weeks","BPD_mean","BPD_sd","HC_mean","HC_sd"}.issubset(set(who_ref.columns)):
        ref = who_ref.set_index("GA_weeks")
        def pct_from_mean_sd(row):
            ga = row["GA_weeks"]
            if ga not in ref.index: return pd.Series({"BPD_percentile": np.nan, "HC_percentile": np.nan})
            mu_b, sd_b = ref.loc[ga, ["BPD_mean","BPD_sd"]]
            mu_h, sd_h = ref.loc[ga, ["HC_mean","HC_sd"]]
            z_b = (row["BPD_cm"] - mu_b) / sd_b if sd_b > 0 else np.nan
            z_h = (row["HC_cm"]  - mu_h) / sd_h if sd_h > 0 else np.nan
            return pd.Series({
                "BPD_percentile": float(normal_cdf(np.array([z_b]))[0] * 100.0) if not np.isnan(z_b) else np.nan,
                "HC_percentile":  float(normal_cdf(np.array([z_h]))[0] * 100.0) if not np.isnan(z_h) else np.nan
            })
        add = df.apply(pct_from_mean_sd, axis=1)
        return pd.concat([df, add], axis=1)

    bpd_qs, hc_qs = [], []
    for q in [3,10,50,90,97]:
        bcol = f"BPD_p{q}"
        hcol = f"HC_p{q}"
        if bcol in who_ref.columns: bpd_qs.append(q/100.0)
        if hcol in who_ref.columns: hc_qs.append(q/100.0)
    if len(bpd_qs) >= 2 and len(hc_qs) >= 2 and "GA_weeks" in who_ref.columns:
        ref = who_ref.set_index("GA_weeks")
        def pct_from_percentiles(row):
            ga = row["GA_weeks"]
            if ga not in ref.index: return pd.Series({"BPD_percentile": np.nan, "HC_percentile": np.nan})
            b_vals, h_vals = [], []
            for q in [3,10,50,90,97]:
                bcol = f"BPD_p{q}"; hcol = f"HC_p{q}"
                if bcol in ref.columns: b_vals.append(ref.loc[ga, bcol])
                if hcol in ref.columns: h_vals.append(ref.loc[ga, hcol])
            bp = interp_percentile_from_quantiles(row["BPD_cm"], bpd_qs, b_vals) if len(b_vals)>=2 else np.nan
            hp = interp_percentile_from_quantiles(row["HC_cm"],  hc_qs, h_vals) if len(h_vals)>=2 else np.nan
            return pd.Series({"BPD_percentile": bp, "HC_percentile": hp})
        add = df.apply(pct_from_percentiles, axis=1)
        return pd.concat([df, add], axis=1)

    raise ValueError("WHO reference CSV not in a recognized format. See script docstring for supported schemas.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=Path, required=True, help="Path to new dataset CSV")
    ap.add_argument("--model_path", type=Path, required=True, help="Path to trained .joblib pipeline")
    ap.add_argument("--out_csv", type=Path, default=Path("eval_predictions.csv"))
    ap.add_argument("--out_json", type=Path, default=Path("eval_metrics.json"))
    ap.add_argument("--pct_low", type=float, default=15.0)
    ap.add_argument("--pct_high", type=float, default=85.0)
    ap.add_argument("--ci_low", type=float, default=74.0)
    ap.add_argument("--ci_high", type=float, default=83.0)
    ap.add_argument("--risk_yellow", type=float, default=0.5)
    ap.add_argument("--risk_red", type=float, default=0.8)
    ap.add_argument("--threshold", type=float, default=None, help="Decision threshold on probability; if omitted, 0.5 unless threshold_from_json provided")
    ap.add_argument("--threshold_from_json", type=Path, default=None, help="JSON file containing 'thresholding.chosen_threshold'")
    ap.add_argument("--who_ref_csv", type=Path, default=None, help="WHO reference table (mean/sd or percentiles) for converting cm to percentiles")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    colmap = {
        "patient": None,
        "GA_weeks": None,
        "BPD_percentile": None,
        "HC_percentile": None,
        "CI_pct": None,
        "Final_screen": None
    }

    for c in colmap.keys():
        if c in df.columns:
            colmap[c] = c

    if colmap["patient"] is None:
        for alt in ["PatientID","id","ID","patient_id"]:
            if alt in df.columns: colmap["patient"] = alt; break
    if colmap["GA_weeks"] is None:
        for alt in ["GA (weeks)","GA_weeks","Gestational_Age_weeks","GAw"]:
            if alt in df.columns: colmap["GA_weeks"] = alt; break
    if colmap["CI_pct"] is None:
        for alt in ["CI (%)","CI","CI_percent"]:
            if alt in df.columns: colmap["CI_pct"] = alt; break
    if colmap["Final_screen"] is None:
        for alt in ["Final screen","Status","Label","Outcome"]:
            if alt in df.columns: colmap["Final_screen"] = alt; break

    has_percentiles = ("BPD_percentile" in df.columns) and ("HC_percentile" in df.columns)
    has_raw = ("BPD (cm)" in df.columns) and ("HC (cm)" in df.columns)

    if not has_percentiles and has_raw:
        if colmap["GA_weeks"] is None:
            raise ValueError("Found raw BPD/HC columns but no GA weeks column. Please include GA (weeks).")
        w = pd.DataFrame({
            "GA_weeks": df[colmap["GA_weeks"]],
            "BPD_cm": df["BPD (cm)"],
            "HC_cm": df["HC (cm)"],
        })
        if args.who_ref_csv is None or not args.who_ref_csv.exists():
            raise ValueError("CSV has raw cm values. Please provide WHO reference with --who_ref_csv to compute percentiles.")
        who = pd.read_csv(args.who_ref_csv)
        w2 = compute_percentiles_from_who(w, who)
     
        df["BPD_percentile"] = w2["BPD_percentile"]
        df["HC_percentile"]  = w2["HC_percentile"]
        has_percentiles = True

    if colmap["CI_pct"] is not None and colmap["CI_pct"] != "CI_pct":
        df["CI_pct"] = df[colmap["CI_pct"]]

    if colmap["Final_screen"] is not None and colmap["Final_screen"] != "Final_screen":
        df["Final_screen"] = df[colmap["Final_screen"]]

    if "Final_screen" in df.columns and df["Final_screen"].dtype == object:
        vals = df["Final_screen"].astype(str).str.strip().str.lower()
        mask_norm = vals.isin(["normal","healthy","not_at_risk","not at risk","0"])
        mask_abn  = vals.isin(["abnormal","unhealthy","flagged","1"])
        df.loc[mask_norm, "Final_screen"] = "NOT_AT_RISK"
        df.loc[mask_abn,  "Final_screen"] = "FLAGGED"

    req = ["BPD_percentile","HC_percentile","CI_pct"]
    missing = [r for r in req if r not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after preprocessing: {missing}. If your CSV has raw cm, pass --who_ref_csv.")

    df["BPD_dev_out"] = outside_band(df["BPD_percentile"].to_numpy(), args.pct_low, args.pct_high)
    df["HC_dev_out"]  = outside_band(df["HC_percentile"].to_numpy(),  args.pct_low, args.pct_high)
    df["BPD_dev_low"] = np.clip(10.0 - df["BPD_percentile"].astype(float), 0, None)
    df["HC_dev_low"]  = np.clip(10.0 - df["HC_percentile"].astype(float), 0, None)
    df["CI_dev_out"]  = ci_outside(df["CI_pct"].to_numpy(), args.ci_low, args.ci_high)

    feature_cols = ["BPD_dev_out","HC_dev_out","BPD_dev_low","HC_dev_low","CI_dev_out"]
    X = df[feature_cols].to_numpy()

    pipe = load(args.model_path)

    threshold = 0.5
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif args.threshold_from_json is not None and args.threshold_from_json.exists():
        meta = json.loads(Path(args.threshold_from_json).read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            th = None
            if "thresholding" in meta and isinstance(meta["thresholding"], dict):
                th = meta["thresholding"].get("chosen_threshold", None)
            if th is None:
                th = meta.get("chosen_threshold", None)
            if th is not None:
                threshold = float(th)

    probs = pipe.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    def bucket(p: float) -> str:
        return "RED" if p >= args.risk_red else ("YELLOW" if p >= args.risk_yellow else "GREEN")
    buckets = [bucket(p) for p in probs]

    out_cols = []
    for c in ["patient","PatientID"]:
        if c in df.columns: out_cols.append(c); break
    for c in ["GA_weeks","GA (weeks)"]:
        if c in df.columns: out_cols.append(c); break
    for c in ["BPD_percentile","HC_percentile","CI_pct"]:
        out_cols.append(c)
    out_df = df[out_cols].copy()
    out_df["risk_probability_unhealthy"] = probs
    out_df["predicted_label"] = np.where(preds==1, "Unhealthy", "Healthy")
    out_df["decision_threshold"] = threshold
    out_df["risk_bucket"] = buckets

    metrics = {
        "used_threshold": float(threshold),
        "bands": {"BPD_HC":[args.pct_low,args.pct_high],"CI":[args.ci_low,args.ci_high]}
    }
    if "Final_screen" in df.columns:
        label_map = {"FLAGGED":1,"NOT_AT_RISK":0}
        y_true = df["Final_screen"].map(label_map)
        if y_true.notna().all():
            accuracy = accuracy_score(y_true, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", pos_label=1, zero_division=0)
            try:
                auc = roc_auc_score(y_true, probs) if len(np.unique(y_true))>1 else float("nan")
            except Exception:
                auc = None
            cm = confusion_matrix(y_true, preds)
            cm_dict = {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
            metrics.update({
                "accuracy": float(accuracy),
                "precision_unhealthy": float(prec),
                "recall_unhealthy": float(rec),
                "f1_unhealthy": float(f1),
                "roc_auc": float(auc) if auc is not None and not np.isnan(auc) else None,
                "confusion_matrix": cm_dict
            })
        else:
            metrics["note"] = "Ground-truth labels present but could not be mapped to FLAGGED/NOT_AT_RISK; metrics omitted."
    else:
        metrics["note"] = "No ground-truth labels found; only predictions were produced."

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
