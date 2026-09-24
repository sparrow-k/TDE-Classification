"""Stage 2: the single final evaluation on the locked test set.

Protocol (fixed in configs/default.yaml before this script was ever run):
  * Data: the 609-object test set produced by make_splits(meta, split_seed=42) - the same set that has been
    excluded from every reported experiment since Stage 0. The remaining 2,434 objects are the development set.
  * Models: trained here on the development set only.
      frozen  = PLAsTiCC pre-trained encoder, frozen, new TDE head   (primary model, DECISIONS D20)
      scratch = same architecture trained from random initialisation (control)
      GBM     = gradient boosting on causal prefix features          (non-deep-learning reference)
    Seeds 0, 1, 2; seed 0 is the declared final model, the other seeds only measure training variability.
  * Early stopping: on an inner split of the development set (never on test data).
  * Thresholds: 0.5 (used throughout the project) and a threshold chosen per cutoff on the cross-validation
    out-of-fold predictions - i.e. from development data only.
  * Metrics: PR-AUC (primary, with a bootstrap confidence interval), ROC-AUC, precision/recall/F1 at both
    thresholds, TDE-vs-AGN and TDE-vs-other sub-problems, and top-k purity.

Everything is written to a new timestamped directory under outputs/final_test/; no previous result is touched.

Usage:  python run_final_test.py
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from run_cv import build_model
from tde.config import load_config, make_run_dir
from tde.data import (BAND_TO_INDEX, LightCurveDataset, collate, estimate_peak_mjd, grouped_stratified_folds,
                      load_mallorn_train, make_splits, to_objects, truncate)
from tde.evaluate import check_causality, compute_metrics, metrics_by_cutoff, predict_at_cutoffs
from tde.train import fit, set_seed

ORDER = ["-50d", "-20d", "-10d", "+0d", "+20d", "+50d", "+100d", "full"]
META_COLS = ["object_id", "label", "spectype", "cutoff", "cutoff_days", "n_obs_seen"]


def prefix_features(o):
    """Causal summary features of one (already truncated) light curve - mirrors diagnostics/cv_feature_baseline_gbm.py."""
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


def prefix_table(objects, cutoffs, peak_bands, peak_min_snr):
    rows = []
    for o in objects:
        tp = estimate_peak_mjd(o, peak_bands, peak_min_snr)
        for c in list(cutoffs) + [None]:
            pre = o if c is None else truncate(o, tp + c)
            rows.append({"object_id": o["object_id"], "label": o["label"], "spectype": o["spectype"],
                         "cutoff": "full" if c is None else f"{c:+d}d",
                         "cutoff_days": np.nan if c is None else c,
                         "n_obs_seen": len(pre["mjd"]), **prefix_features(pre)})
    return pd.DataFrame(rows)


def bootstrap_pr_auc(labels, probs, n_resamples, seed=0):
    """Percentile bootstrap confidence interval for PR-AUC, resampling objects."""
    labels, probs = np.asarray(labels), np.nan_to_num(np.asarray(probs), nan=0.0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(labels), len(labels))
        if 0 < labels[idx].sum() < len(idx):
            vals.append(average_precision_score(labels[idx], probs[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sub_problem_roc(g):
    """ROC-AUC of TDE vs AGN and TDE vs every non-AGN class, on one set of predictions."""
    a = g[(g["label"] == 1) | (g["spectype"] == "AGN")]
    b = g[(g["label"] == 1) | (g["spectype"] != "AGN")]
    return (float(roc_auc_score(a["label"], a["prob"])), float(roc_auc_score(b["label"], b["prob"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fcfg, ccfg, dcfg, ecfg = cfg["final_test"], cfg["cv"], cfg["data"], cfg["eval"]
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    run_dir = make_run_dir(fcfg["output_dir"], "final_test", cfg)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    print(f"Run dir: {run_dir}   device: {device}   git commit: {commit[:10]}")

    # ---- data: the locked test set, and the development set that everything was built on
    lc, meta = load_mallorn_train(dcfg["mallorn_dir"], dcfg["processed_dir"])
    splits = make_splits(meta, ccfg["split_seed"], dcfg["n_folds"])
    test_ids = set(splits["test"])
    dev_meta = meta[~meta["object_id"].isin(test_ids)].reset_index(drop=True)
    test_meta = meta[meta["object_id"].isin(test_ids)].reset_index(drop=True)

    # the development set must be exactly the set the cross-validation used
    cv_folds = json.load(open(os.path.join(os.path.dirname(fcfg["threshold_source"]), "folds.json")))
    cv_dev_ids = {i for v in cv_folds.values() for i in v}
    assert cv_dev_ids == set(dev_meta["object_id"]), "development set differs from the cross-validation one"
    assert not (test_ids & cv_dev_ids), "locked test objects appear in the cross-validation folds"
    print(f"development set: {len(dev_meta)} objects ({int(dev_meta['target'].sum())} TDE)   "
          f"locked TEST set: {len(test_meta)} objects ({int(test_meta['target'].sum())} TDE)")

    dev_objects = to_objects(lc[lc["object_id"].isin(set(dev_meta["object_id"]))], dev_meta)
    test_objects = to_objects(lc[lc["object_id"].isin(test_ids)], test_meta)
    y_dev = dev_meta["target"].to_numpy()
    groups_dev = dev_meta.groupby(["z", "ebv"]).ngroup().to_numpy()
    eval_kwargs = dict(cutoffs_days=ecfg["cutoffs_days"], flux_scale=dcfg["flux_scale"], device=device,
                       peak_bands=ecfg["peak_bands"], peak_min_snr=ecfg["peak_min_snr"])

    # ---- thresholds chosen on DEVELOPMENT data (cross-validation out-of-fold predictions)
    oof = pd.read_csv(fcfg["threshold_source"])
    oof["prob"] = oof["prob"].fillna(0.0)
    dev_primary = oof[(oof["mode"] == fcfg["primary_mode"]) & (oof["seed"] == fcfg["primary_seed"])]
    dev_thresholds = {}
    for cutoff in ORDER:
        g = dev_primary[dev_primary["cutoff"] == cutoff]
        prec, rec, thr = precision_recall_curve(g["label"], g["prob"])
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        dev_thresholds[cutoff] = float(thr[int(np.argmax(f1))])
    print("thresholds from development data:", {k: round(v, 3) for k, v in dev_thresholds.items()})

    # ---- train the final models on the development set, then predict the test set once
    checkpoint = torch.load(fcfg["pretrained_model"], map_location="cpu")
    all_preds, rows = [], []
    for mode in fcfg["modes"]:
        for seed in fcfg["seeds"]:
            t0 = time.time()
            set_seed(seed)
            inner = grouped_stratified_folds(y_dev, groups_dev, ccfg["inner_val_folds"], seed)[0]
            stop_mask = np.zeros(len(y_dev), dtype=bool)
            stop_mask[inner] = True
            fit_objs = [o for o, s in zip(dev_objects, stop_mask) if not s]
            stop_objs = [o for o, s in zip(dev_objects, stop_mask) if s]

            model = build_model(mode, cfg, checkpoint, device)
            n_pos = sum(o["label"] for o in fit_objs)
            pos_weight = torch.tensor((len(fit_objs) - n_pos) / max(n_pos, 1), device=device)
            fit_loader = DataLoader(LightCurveDataset(fit_objs, dcfg["flux_scale"]), batch_size=ccfg["batch_size"],
                                    shuffle=True, collate_fn=collate,
                                    generator=torch.Generator().manual_seed(seed))
            stop_loader = DataLoader(LightCurveDataset(stop_objs, dcfg["flux_scale"]), batch_size=256,
                                     shuffle=False, collate_fn=collate)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=ccfg["lr"][mode], weight_decay=ccfg["weight_decay"])
            result = fit(model, fit_loader, stop_loader, optimizer, pos_weight, device, ccfg["epochs"],
                         ccfg["patience"], cfg["train"]["grad_clip"])
            torch.save({"state_dict": model.state_dict(), "mode": mode, "seed": seed,
                        "model_config": checkpoint["model_config"], "best_epoch": result["best_epoch"],
                        "n_fit": len(fit_objs), "n_early_stop": len(stop_objs), "git_commit": commit},
                       os.path.join(run_dir, "models", f"{mode}_seed{seed}.pt"))

            max_diff = check_causality(model, test_objects, dcfg["flux_scale"], device, n_checks=20)
            preds = predict_at_cutoffs(model, test_objects, **eval_kwargs)
            preds.insert(0, "seed", seed)
            preds.insert(0, "mode", mode)
            all_preds.append(preds)
            m = metrics_by_cutoff(preds).assign(mode=mode, seed=seed, best_epoch=result["best_epoch"],
                                                causality_max_abs_diff=max_diff)
            rows.append(m)
            print(f"{mode:8s} seed {seed} | fit {len(fit_objs)} / stop {len(stop_objs)} | "
                  f"best epoch {result['best_epoch']:3d} | TEST PR-AUC peak "
                  f"{m.set_index('cutoff').loc['+0d', 'pr_auc']:.3f} full "
                  f"{m.set_index('cutoff').loc['full', 'pr_auc']:.3f} | {time.time() - t0:.0f}s")

    preds = pd.concat(all_preds, ignore_index=True)

    # ---- gradient-boosting reference: trained on the development set, evaluated on the test set
    t0 = time.time()
    dev_tab = prefix_table(dev_objects, ecfg["cutoffs_days"], ecfg["peak_bands"], ecfg["peak_min_snr"])
    test_tab = prefix_table(test_objects, ecfg["cutoffs_days"], ecfg["peak_bands"], ecfg["peak_min_snr"])
    features = [c for c in dev_tab.columns if c not in META_COLS]
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    gbm.fit(dev_tab[features], dev_tab["label"])
    gbm_preds = test_tab[META_COLS].assign(prob=gbm.predict_proba(test_tab.reindex(columns=features))[:, 1],
                                           mode="gbm", seed=0)
    preds = pd.concat([preds, gbm_preds], ignore_index=True)
    rows.append(metrics_by_cutoff(gbm_preds).assign(mode="gbm", seed=0, best_epoch=np.nan,
                                                    causality_max_abs_diff=np.nan))
    print(f"gbm      reference | TEST PR-AUC peak "
          f"{rows[-1].set_index('cutoff').loc['+0d', 'pr_auc']:.3f} full "
          f"{rows[-1].set_index('cutoff').loc['full', 'pr_auc']:.3f} | {time.time() - t0:.0f}s")

    # ---- metrics: per model, per cutoff; plus development-threshold operating points and bootstrap CIs
    metrics = pd.concat(rows, ignore_index=True)
    extra = []
    for (mode, seed, cutoff), g in preds.groupby(["mode", "seed", "cutoff"], sort=False):
        thr = dev_thresholds[cutoff]
        at_dev = compute_metrics(g["label"], g["prob"], threshold=thr)
        row = {"mode": mode, "seed": seed, "cutoff": cutoff, "threshold_dev": thr,
               "precision_at_dev_threshold": at_dev["precision"], "recall_at_dev_threshold": at_dev["recall"],
               "f1_at_dev_threshold": at_dev["f1"]}
        if seed == fcfg["primary_seed"]:
            lo, hi = bootstrap_pr_auc(g["label"], g["prob"], fcfg["bootstrap"])
            roc_agn, roc_other = sub_problem_roc(g.assign(prob=g["prob"].fillna(0.0)))
            order = g.sort_values("prob", ascending=False)
            row.update({"pr_auc_ci_low": lo, "pr_auc_ci_high": hi, "roc_tde_vs_agn": roc_agn,
                        "roc_tde_vs_other": roc_other,
                        **{f"tdes_in_top_{k}": int(order.head(k)["label"].sum()) for k in (20, 50, 100)}})
        extra.append(row)
    metrics = metrics.merge(pd.DataFrame(extra), on=["mode", "seed", "cutoff"], how="left")
    metrics["cutoff"] = pd.Categorical(metrics["cutoff"], ORDER, ordered=True)
    metrics = metrics.sort_values(["mode", "seed", "cutoff"])

    # ---- development (cross-validation) numbers next to the final test numbers
    cv_pooled = pd.read_csv(os.path.join(os.path.dirname(fcfg["threshold_source"]), "pooled_metrics_by_seed.csv"))
    cv_gbm = pd.read_csv(os.path.join(os.path.dirname(fcfg["threshold_source"]), "gbm_reference_metrics.csv"))
    comp = []
    for cutoff in ORDER:
        row = {"cutoff": cutoff}
        for mode in list(fcfg["modes"]):
            cv = cv_pooled[(cv_pooled["mode"] == mode) & (cv_pooled["cutoff"] == cutoff)]["pr_auc"]
            te = metrics[(metrics["mode"] == mode) & (metrics["cutoff"] == cutoff)]["pr_auc"]
            row[f"{mode}_cv_mean"] = float(cv.mean())
            row[f"{mode}_cv_std"] = float(cv.std())
            row[f"{mode}_test_mean"] = float(te.mean())
            row[f"{mode}_test_std"] = float(te.std())
        row["gbm_cv"] = float(cv_gbm[cv_gbm["cutoff"] == cutoff]["pr_auc"].iloc[0])
        row["gbm_test"] = float(metrics[(metrics["mode"] == "gbm") & (metrics["cutoff"] == cutoff)]["pr_auc"].iloc[0])
        comp.append(row)
    comparison = pd.DataFrame(comp)

    # ---- save everything
    preds.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
    metrics.to_csv(os.path.join(run_dir, "test_metrics_by_cutoff.csv"), index=False)
    comparison.to_csv(os.path.join(run_dir, "cv_vs_test_pr_auc.csv"), index=False)
    json.dump(sorted(test_ids), open(os.path.join(run_dir, "test_object_ids.json"), "w"))
    json.dump({"dev_thresholds": dev_thresholds}, open(os.path.join(run_dir, "dev_thresholds.json"), "w"), indent=2)
    primary = metrics[(metrics["mode"] == fcfg["primary_mode"]) & (metrics["seed"] == fcfg["primary_seed"])]
    summary = {
        "git_commit": commit, "device": device, "run_dir": run_dir,
        "protocol": {"split_seed": ccfg["split_seed"], "modes": fcfg["modes"], "seeds": fcfg["seeds"],
                     "primary_model": f"{fcfg['primary_mode']} seed {fcfg['primary_seed']}",
                     "pretrained_model": fcfg["pretrained_model"], "threshold_source": fcfg["threshold_source"],
                     "bootstrap_resamples": fcfg["bootstrap"]},
        "test_set": {"n": len(test_meta), "n_tde": int(test_meta["target"].sum()),
                     "prevalence": float(test_meta["target"].mean()),
                     "class_counts": test_meta["spectype"].value_counts().to_dict()},
        "development_set": {"n": len(dev_meta), "n_tde": int(dev_meta["target"].sum())},
        "primary_model_test_metrics": {r.cutoff: {"pr_auc": r.pr_auc, "pr_auc_ci": [r.pr_auc_ci_low, r.pr_auc_ci_high],
                                                  "roc_auc": r.roc_auc, "f1_at_0.5": r.f1,
                                                  "precision_at_0.5": r.precision, "recall_at_0.5": r.recall,
                                                  "threshold_dev": r.threshold_dev,
                                                  "precision_at_dev_threshold": r.precision_at_dev_threshold,
                                                  "recall_at_dev_threshold": r.recall_at_dev_threshold,
                                                  "roc_tde_vs_agn": r.roc_tde_vs_agn,
                                                  "roc_tde_vs_other": r.roc_tde_vs_other,
                                                  "tdes_in_top_20": r.tdes_in_top_20}
                                      for r in primary.itertuples()},
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2, default=float)

    with open(os.path.join(run_dir, "README.md"), "w") as fh:
        fh.write(f"""# Final held-out test evaluation

Run: `python run_final_test.py` on git commit `{commit}` (device: {device}).

**These are the only test-set numbers in the project.** Everything under `outputs/cv/` and `outputs/runs/` is
development / cross-validation work and was produced without touching this test set.

* Test set: {len(test_meta)} objects, {int(test_meta['target'].sum())} TDEs
  (`make_splits(meta, seed={ccfg['split_seed']})["test"]`, listed in `test_object_ids.json`).
* Development set: {len(dev_meta)} objects, {int(dev_meta['target'].sum())} TDEs - identical to the
  cross-validation set of `{os.path.dirname(fcfg['threshold_source'])}`.
* Models trained here on development data only: {', '.join(fcfg['modes'])} x seeds {fcfg['seeds']},
  plus a gradient-boosting reference. Primary model: **{fcfg['primary_mode']} seed {fcfg['primary_seed']}**
  (`models/{fcfg['primary_mode']}_seed{fcfg['primary_seed']}.pt`).
* Decision thresholds come from development data only (`dev_thresholds.json`).

Files: `test_predictions.csv`, `test_metrics_by_cutoff.csv`, `cv_vs_test_pr_auc.csv`, `summary.json`,
`config.yaml`, `models/`.
""")

    cols = ["cutoff", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc", "precision", "recall", "f1",
            "roc_tde_vs_agn", "roc_tde_vs_other", "tdes_in_top_20"]
    print(f"\nFINAL TEST RESULTS - primary model ({fcfg['primary_mode']} seed {fcfg['primary_seed']}), "
          f"{len(test_meta)} objects, {int(test_meta['target'].sum())} TDEs "
          f"(chance PR-AUC {test_meta['target'].mean():.3f}):")
    print(primary[cols].to_string(index=False, float_format="%.3f"))
    print("\nDevelopment (cross-validation) vs. final test, PR-AUC:")
    print(comparison.to_string(index=False, float_format="%.3f"))
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
