# Usage: python diagnostics/<script>.py outputs/runs/<run_dir>   (uses that run's splits.json / model.pt / predictions)
# Diagnostics on the Stage 0 run: per-class behaviour on val, and fit on the training set.
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tde.data import N_FEATURES, load_mallorn_train, to_objects
from tde.evaluate import metrics_by_cutoff, predict_at_cutoffs
from tde.model import GRUClassifier

RUN = sys.argv[1]  # run directory, e.g. outputs/runs/<timestamp>_stage0_mallorn_gru_seed42
preds = pd.read_csv(f"{RUN}/val_predictions.csv")
for c in ["-20d", "+0d", "full"]:
    g = preds[preds.cutoff == c]
    print(f"\n== val cutoff {c}: mean prob by class")
    print(g.groupby("spectype").prob.agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())
    vs_agn = g[(g.label == 1) | (g.spectype == "AGN")]
    vs_sn = g[(g.label == 1) | (g.spectype != "AGN")]
    print(f"TDE vs AGN only: AP={average_precision_score(vs_agn.label, vs_agn.prob):.3f} (prev {vs_agn.label.mean():.3f}) "
          f"ROC={roc_auc_score(vs_agn.label, vs_agn.prob):.3f}")
    print(f"TDE vs non-AGN: AP={average_precision_score(vs_sn.label, vs_sn.prob):.3f} (prev {vs_sn.label.mean():.3f}) "
          f"ROC={roc_auc_score(vs_sn.label, vs_sn.prob):.3f}")

lc, meta = load_mallorn_train("data/mallorn-astronomical-classification-challenge (1)", "data/processed")
splits = json.load(open(f"{RUN}/splits.json"))
objs = {o["object_id"]: o for o in to_objects(lc, meta)}
model = GRUClassifier(N_FEATURES, 64, 2, 0.1).cuda()
model.load_state_dict(torch.load(f"{RUN}/model.pt"))
tr = [objs[i] for i in splits["train"]]
m = metrics_by_cutoff(predict_at_cutoffs(model, tr, [-20, 0, 20], 1.0, "cuda", ["g", "r", "i"], 3.0))
print("\n== TRAIN set metrics (best checkpoint)")
print(m[["cutoff", "n_pos", "pr_auc", "roc_auc", "f1"]].to_string(index=False, float_format="%.3f"))
