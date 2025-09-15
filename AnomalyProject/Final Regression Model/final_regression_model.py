
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from joblib import dump

def outside_band(arr, low, high):
    arr = arr.astype(float)
    return np.clip(low - arr, 0, None) + np.clip(arr - high, 0, None)
def ci_outside(arr, low, high):
    arr = arr.astype(float)
    return np.clip(low - arr, 0, None) + np.clip(arr - high, 0, None)

ap = argparse.ArgumentParser()
ap.add_argument("--input_csv", type=Path, required=True)
ap.add_argument("--out_csv", type=Path, required=True)
ap.add_argument("--out_json", type=Path, required=True)
ap.add_argument("--out_model", type=Path, required=True)
ap.add_argument("--pct_low", type=float, default=15.0)
ap.add_argument("--pct_high", type=float, default=85.0)
ap.add_argument("--ci_low", type=float, default=74.0)
ap.add_argument("--ci_high", type=float, default=83.0)
ap.add_argument("--l1_ratio", type=float, default=0.5)
ap.add_argument("--C", type=float, default=1.0)
ap.add_argument("--test_size", type=float, default=0.2)
ap.add_argument("--random_state", type=int, default=42)
ap.add_argument("--optimize", choices=["none","recall"], default="recall")
ap.add_argument("--target_recall", type=float, default=0.98)
ap.add_argument("--risk_yellow", type=float, default=0.5)
ap.add_argument("--risk_red", type=float, default=0.8)
args = ap.parse_args()

df = pd.read_csv(args.input_csv)
label_map = {"FLAGGED":1,"NOT_AT_RISK":0}
df = df[df["Final_screen"].isin(label_map)].copy()
df["label"] = df["Final_screen"].map(label_map).astype(int)

df["BPD_dev_out"] = outside_band(df["BPD_percentile"].to_numpy(), args.pct_low, args.pct_high)
df["HC_dev_out"]  = outside_band(df["HC_percentile"].to_numpy(),  args.pct_low, args.pct_high)
df["BPD_dev_low"] = np.clip(10.0 - df["BPD_percentile"].astype(float), 0, None)
df["HC_dev_low"]  = np.clip(10.0 - df["HC_percentile"].astype(float), 0, None)
df["CI_dev_out"]  = ci_outside(df["CI_pct"].to_numpy(), args.ci_low, args.ci_high)

feature_cols = ["BPD_dev_out","HC_dev_out","BPD_dev_low","HC_dev_low","CI_dev_out"]
X = df[feature_cols].to_numpy(); y = df["label"].to_numpy()
for j in range(X.shape[1]):
    if np.allclose(X[:,j],0): X[:,j] += 1e-6*np.random.randn(X.shape[0])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                            l1_ratio=args.l1_ratio, C=args.C, max_iter=5000,
                                            class_weight="balanced", n_jobs=1))])
pipe.fit(Xtr, ytr)

probs = pipe.predict_proba(Xte)[:,1]
uniq = np.unique(np.round(probs,6))
def recall_at(th):
    pred = (probs>=th).astype(int)
    tp = ((pred==1)&(yte==1)).sum(); fn = ((pred==0)&(yte==1)).sum()
    return tp/(tp+fn) if (tp+fn)>0 else 0.0
chosen = None
for th in sorted(uniq):
    if recall_at(th) >= args.target_recall: chosen = th
if chosen is None: chosen = uniq.min() if len(uniq) else 0.0

pred = (probs>=chosen).astype(int)
acc = accuracy_score(yte, pred)
prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", pos_label=1, zero_division=0)
auc = roc_auc_score(yte, probs) if len(np.unique(yte))>1 else float("nan")
cm = confusion_matrix(yte, pred)
cm_dict = {"tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1])}

probs_full = pipe.predict_proba(df[feature_cols])[:,1]
pred_full = (probs_full>=chosen).astype(int)
def bucket(p):
    return "RED" if p>=args.risk_red else ("YELLOW" if p>=args.risk_yellow else "GREEN")
out = df[["patient","GA_weeks","BPD_percentile","HC_percentile","CI_pct","Final_screen"]].copy()
for c in feature_cols: out[c] = df[c]
out["risk_probability_unhealthy"] = [f"{p*100:.2f}%" for p in probs_full]
out["predicted_label"] = np.where(pred_full==1,"Unhealthy","Healthy")
out["decision_threshold"] = chosen
out["risk_bucket"] = [bucket(p) for p in probs_full]
out.to_csv(args.out_csv, index=False)

payload = {"feature_order": feature_cols,
           "weights": pipe.named_steps["clf"].coef_.ravel().tolist(),
           "intercept": float(pipe.named_steps["clf"].intercept_[0]),
           "bands":{"BPD_HC":[args.pct_low,args.pct_high],"CI":[args.ci_low,args.ci_high]},
           "thresholding":{"mode":args.optimize,"chosen_threshold":float(chosen),"target_recall":args.target_recall},
           "metrics_test":{"accuracy":float(acc),"precision_unhealthy":float(prec),"recall_unhealthy":float(rec),
                           "f1_unhealthy":float(f1),"roc_auc":float(auc),"confusion_matrix":cm_dict},
           "risk_buckets":{"yellow_lower":args.risk_yellow,"red_lower":args.risk_red,
                           "counts":out["risk_bucket"].value_counts().to_dict()}}
Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
dump(pipe, args.out_model)
print(json.dumps({"chosen_threshold":float(chosen),"metrics_test":payload["metrics_test"]}, indent=2))
