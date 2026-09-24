# Usage: python diagnostics/cv_class_breakdown.py outputs/cv/<run_dir>
# Per-class behaviour of pooled out-of-fold predictions from run_cv.py: mean P(TDE) per true class, and whether
# each mode separates TDEs from AGN and from non-AGN transients (supernovae etc.). Metrics averaged over seeds.
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RUN = sys.argv[1]
preds = pd.read_csv(os.path.join(RUN, "oof_predictions.csv"))
preds["prob"] = preds["prob"].fillna(0.0)
modes = list(dict.fromkeys(preds["mode"]))
pd.set_option("display.width", 200)

for cutoff in ["-20d", "+0d", "+50d", "full"]:
    g = preds[preds["cutoff"] == cutoff]
    means = g.groupby(["spectype", "mode"])["prob"].mean().unstack()[modes]
    means["n_objects"] = g[g["mode"] == modes[0]].groupby("spectype")["object_id"].nunique()
    print(f"\n=== cutoff {cutoff}: mean P(TDE) by true class (averaged over seeds)")
    print(means.sort_values(modes[-1], ascending=False).round(3).to_string())
    rows = []
    for (mode, seed), s in g.groupby(["mode", "seed"], sort=False):
        vs_agn = s[(s["label"] == 1) | (s["spectype"] == "AGN")]
        vs_other = s[(s["label"] == 1) | (s["spectype"] != "AGN")]
        rows.append({"mode": mode,
                     "AP_vs_AGN": average_precision_score(vs_agn["label"], vs_agn["prob"]),
                     "ROC_vs_AGN": roc_auc_score(vs_agn["label"], vs_agn["prob"]),
                     "AP_vs_nonAGN": average_precision_score(vs_other["label"], vs_other["prob"]),
                     "ROC_vs_nonAGN": roc_auc_score(vs_other["label"], vs_other["prob"])})
    table = pd.DataFrame(rows).groupby("mode", sort=False).mean()
    prev_agn = vs_agn["label"].mean()
    prev_other = vs_other["label"].mean()
    print(f"TDE vs AGN (chance AP {prev_agn:.3f}) and TDE vs non-AGN classes (chance AP {prev_other:.3f}):")
    print(table.round(3).to_string())
