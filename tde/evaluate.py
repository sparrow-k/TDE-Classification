"""Prediction and early-time evaluation.

Early predictions are made by physically truncating the input light curve at the cutoff
and running the model on the truncated sequence only. The model therefore cannot see any
observation after the cutoff, regardless of its architecture.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, \
    recall_score, roc_auc_score
from torch.utils.data import DataLoader

from tde.data import LightCurveDataset, build_features, collate, estimate_peak_mjd, truncate


@torch.no_grad()
def predict_last_step(model, objects, flux_scale, device, batch_size=256):
    """TDE probability after the last observation of each light curve (NaN if it has none)."""
    model.eval()
    loader = DataLoader(LightCurveDataset(objects, flux_scale), batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    probs = []
    for x, _, _, lengths in loader:
        if x.shape[1] == 0:  # every curve in the batch is empty; the GRU needs >= 1 step (result is NaN anyway)
            x = torch.zeros(x.shape[0], 1, x.shape[2])
        step_probs = model.tde_probability(model(x.to(device))).cpu().double()
        p = step_probs[torch.arange(len(lengths)), (lengths - 1).clamp(min=0)]
        p[lengths == 0] = float("nan")
        probs.append(p.numpy())
    return np.concatenate(probs) if probs else np.array([])


@torch.no_grad()
def predict_steps(model, obj, flux_scale, device):
    """Per-step TDE probability track for one full light curve."""
    model.eval()
    x = torch.from_numpy(build_features(obj["mjd"], obj["flux"], obj["flux_err"], obj["band"], flux_scale))
    return model.tde_probability(model(x[None].to(device))).cpu().double().numpy()[0]


def cutoff_label(c):
    return "full" if c is None else f"{c:+d}d"


def predict_at_cutoffs(model, objects, cutoffs_days, flux_scale, device, peak_bands, peak_min_snr):
    """Predictions for every object at every cutoff (days relative to estimated peak) plus 'full'."""
    peaks = np.array([estimate_peak_mjd(o, peak_bands, peak_min_snr) for o in objects])
    rows = []
    for c in list(cutoffs_days) + [None]:
        if c is None:
            prefixes = objects
        else:
            prefixes = [truncate(o, p + c) for o, p in zip(objects, peaks)]
        probs = predict_last_step(model, prefixes, flux_scale, device)
        for o, pre, p in zip(objects, prefixes, probs):
            rows.append({
                "object_id": o["object_id"], "label": o["label"], "spectype": o["spectype"],
                "cutoff": cutoff_label(c), "cutoff_days": np.nan if c is None else c,
                "n_obs_seen": len(pre["mjd"]), "prob": p,
            })
    return pd.DataFrame(rows)


def compute_metrics(labels, probs, threshold=0.5):
    """PR-AUC, ROC-AUC, F1 at a fixed threshold and best-possible F1 (optimistic, threshold tuned on same data).

    Objects with no observations before the cutoff (prob = NaN) are scored 0: nothing can be flagged yet.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.nan_to_num(np.asarray(probs, dtype=float), nan=0.0)
    out = {"n": len(labels), "n_pos": int(labels.sum()), "prevalence": float(labels.mean())}
    if 0 < labels.sum() < len(labels):
        out["pr_auc"] = float(average_precision_score(labels, probs))
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        prec, rec, thr = precision_recall_curve(labels, probs)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        out["f1_best"] = float(f1.max())
        out["threshold_best"] = float(thr[f1.argmax()])
    else:
        out.update(pr_auc=np.nan, roc_auc=np.nan, f1_best=np.nan, threshold_best=np.nan)
    pred = probs >= threshold
    out["threshold"] = threshold
    out["f1"] = float(f1_score(labels, pred, zero_division=0))
    out["precision"] = float(precision_score(labels, pred, zero_division=0))
    out["recall"] = float(recall_score(labels, pred, zero_division=0))
    return out


def metrics_by_cutoff(preds, threshold=0.5):
    rows = []
    for cutoff, g in preds.groupby("cutoff", sort=False):
        m = compute_metrics(g["label"], g["prob"], threshold)
        m.update(cutoff=cutoff, cutoff_days=g["cutoff_days"].iloc[0],
                 median_obs_seen=float(g["n_obs_seen"].median()), n_empty=int((g["n_obs_seen"] == 0).sum()))
        rows.append(m)
    cols = ["cutoff", "cutoff_days", "n", "n_pos", "prevalence", "pr_auc", "roc_auc", "f1", "precision",
            "recall", "f1_best", "threshold_best", "threshold", "median_obs_seen", "n_empty"]
    return pd.DataFrame(rows)[cols]


def check_causality(model, objects, flux_scale, device, n_checks=20, seed=0, tol=1e-4):
    """Runtime guard: prediction from a truncated curve must equal the full-curve output at the same step.

    If a model (or feature) used future observations, the two would differ.
    Returns the maximum absolute difference found.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in rng.choice(len(objects), size=min(n_checks, len(objects)), replace=False):
        obj = objects[i]
        full_track = predict_steps(model, obj, flux_scale, device)
        k = int(rng.integers(1, len(obj["mjd"]) + 1))
        prefix = truncate(obj, obj["mjd"][k - 1])  # includes any simultaneous observations at that time
        n = len(prefix["mjd"])
        p_trunc = predict_last_step(model, [prefix], flux_scale, device)[0]
        worst = max(worst, abs(p_trunc - full_track[n - 1]))
    if worst > tol:
        raise AssertionError(f"Causality check failed: max |p_truncated - p_full| = {worst:.2e}")
    return worst
