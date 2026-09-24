# Usage: python diagnostics/<script>.py outputs/runs/<run_dir>   (uses that run's splits.json / model.pt / predictions)
# Reference (non-DL) baseline: gradient boosting on causal summary features of each prefix.
# Same split, same cutoffs, same inputs (no metadata). Answers: is there learnable signal at these cutoffs?
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from tde.data import estimate_peak_mjd, load_mallorn_train, to_objects, truncate
from tde.evaluate import compute_metrics

RUN = sys.argv[1]  # run directory, e.g. outputs/runs/<timestamp>_stage0_mallorn_gru_seed42
CUTOFFS = [-50, -20, -10, 0, 20, 50, 100, None]


def prefix_features(o):
    m, f, e, b = o["mjd"], o["flux"], o["flux_err"], o["band"]
    out = {"n_obs": len(m)}
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
    out["mean_snr2"] = np.mean(snr ** 2)  # variability / chi2 w.r.t. zero flux
    det = np.flatnonzero(snr > 5)
    out["days_since_first_det"] = t_last - m[det[0]] if len(det) else -1.0
    out["days_since_max"] = t_last - m[np.argmax(f)]
    return out


def table(objs):
    rows = []
    for o in objs:
        tp = estimate_peak_mjd(o)
        for c in CUTOFFS:
            pre = o if c is None else truncate(o, tp + c)
            rows.append({"object_id": o["object_id"], "label": o["label"], "cutoff": "full" if c is None else f"{c:+d}d",
                         **prefix_features(pre)})
    return pd.DataFrame(rows)


lc, meta = load_mallorn_train("data/mallorn-astronomical-classification-challenge (1)", "data/processed")
splits = json.load(open(f"{RUN}/splits.json"))
objs = {o["object_id"]: o for o in to_objects(lc, meta)}
tr, va = table([objs[i] for i in splits["train"]]), table([objs[i] for i in splits["val"]])
X = [c for c in tr.columns if c not in ("object_id", "label", "cutoff")]
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
clf.fit(tr[X], tr.label)  # trained on prefixes at all cutoffs pooled
va["prob"] = clf.predict_proba(va[X])[:, 1]
rows = []
for c, g in va.groupby("cutoff", sort=False):
    rows.append({"cutoff": c, **compute_metrics(g.label, g.prob)})
print("GBM on causal prefix features (val):")
print(pd.DataFrame(rows)[["cutoff", "n_pos", "pr_auc", "roc_auc", "f1", "f1_best"]].to_string(index=False, float_format="%.3f"))
