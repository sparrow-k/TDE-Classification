# Usage: python diagnostics/cv_feature_baseline_gbm.py outputs/cv/<run_dir>
# Non-DL reference on EXACTLY the same folds as a run_cv.py run (folds.json): for each fold, a gradient-boosting
# classifier is trained on causal prefix summary features of the other folds' objects (prefixes at all cutoffs pooled)
# and evaluated on the held-out fold. Pooled out-of-fold metrics are saved to <run_dir>/gbm_reference_metrics.csv.
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from tde.data import estimate_peak_mjd, load_mallorn_train, to_objects, truncate
from tde.evaluate import metrics_by_cutoff

RUN = sys.argv[1]
CUTOFFS = [-50, -20, -10, 0, 20, 50, 100, None]
META_COLS = ["object_id", "label", "spectype", "cutoff", "cutoff_days", "n_obs_seen"]


def prefix_features(o):
    """Summary features computed ONLY from the observations in the prefix."""
    m, f, e, b = o["mjd"], o["flux"], o["flux_err"], o["band"]
    out = {}
    if len(m) == 0:
        return out
    snr = f / e
    t_last = m[-1]
    for bi, bn in [(1, "g"), (2, "r"), (3, "i")]:
        s = b == bi
        rec = s & (m >= t_last - 30)
        out[f"max_flux_{bn}"] = f[s].max() if s.any() else np.nan
        out[f"median_flux_{bn}"] = np.median(f[s]) if s.any() else np.nan
        out[f"recent_flux_{bn}"] = f[rec].mean() if rec.any() else np.nan
    out["recent_g_minus_r"] = np.arcsinh(out["recent_flux_g"]) - np.arcsinh(out["recent_flux_r"])
    out["max_snr"] = snr.max()
    out["frac_snr5"] = (snr > 5).mean()
    out["mean_snr2"] = np.mean(snr ** 2)
    det = np.flatnonzero(snr > 5)
    out["days_since_first_det"] = t_last - m[det[0]] if len(det) else -1.0
    out["days_since_max"] = t_last - m[np.argmax(f)]
    out["n_obs"] = len(m)
    return out


def prefix_table(objs):
    rows = []
    for o in objs:
        tp = estimate_peak_mjd(o)
        for c in CUTOFFS:
            pre = o if c is None else truncate(o, tp + c)
            rows.append({"object_id": o["object_id"], "label": o["label"], "spectype": o["spectype"],
                         "cutoff": "full" if c is None else f"{c:+d}d", "cutoff_days": np.nan if c is None else c,
                         "n_obs_seen": len(pre["mjd"]), **prefix_features(pre)})
    return pd.DataFrame(rows)


folds = json.load(open(os.path.join(RUN, "folds.json")))
lc, meta = load_mallorn_train("data/mallorn-astronomical-classification-challenge (1)", "data/processed")
ids = [i for fold_ids in folds.values() for i in fold_ids]
meta = meta.set_index("object_id").loc[ids].reset_index()
objects = {o["object_id"]: o for o in to_objects(lc[lc["object_id"].isin(set(ids))], meta)}
tables = {k: prefix_table([objects[i] for i in fold_ids]) for k, fold_ids in folds.items()}

preds = []
for k, held in tables.items():
    train = pd.concat([t for kk, t in tables.items() if kk != k], ignore_index=True)
    features = [c for c in train.columns if c not in META_COLS]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    clf.fit(train[features], train["label"])
    preds.append(held[META_COLS].assign(prob=clf.predict_proba(held.reindex(columns=features))[:, 1]))
metrics = metrics_by_cutoff(pd.concat(preds, ignore_index=True))
metrics.to_csv(os.path.join(RUN, "gbm_reference_metrics.csv"), index=False)
print("GBM reference, pooled out-of-fold on the run_cv folds:")
print(metrics[["cutoff", "n", "n_pos", "pr_auc", "roc_auc", "f1", "f1_best"]].to_string(index=False, float_format="%.3f"))
