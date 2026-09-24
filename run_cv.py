"""Stage 1b: cross-validated comparison on MALLORN (the locked test set is excluded).

Modes
  scratch   random initialisation, all weights trained                 (Stage 0 model, better protocol)
  frozen    PLAsTiCC encoder loaded and frozen, only a new head trained  (proposal's method)
  finetune  PLAsTiCC encoder loaded, all weights trained at a lower LR

Protocol: 5 grouped-stratified folds over the non-test objects. For each fold, 1/5 of the training
portion is held out for early stopping on validation loss; the model is then evaluated on the
held-out fold at every cutoff. Out-of-fold predictions from all folds are pooled (all TDEs used)
and metrics are reported as mean ± std over training seeds.

Usage:
  python run_cv.py --modes scratch frozen finetune --pretrained outputs/pretrain/<run>/model.pt
  python run_cv.py --smoke --modes scratch frozen finetune --pretrained <smoke pretrain model.pt>
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tde.config import load_config, make_run_dir
from tde.data import (N_FEATURES, LightCurveDataset, collate, grouped_stratified_folds, load_mallorn_train,
                      make_splits, stratified_subset, to_objects)
from tde.evaluate import check_causality, metrics_by_cutoff, predict_at_cutoffs
from tde.model import GRUClassifier, load_encoder_weights
from tde.train import fit, set_seed

MODES = ("scratch", "frozen", "finetune")


def build_model(mode, cfg, checkpoint, device):
    if mode == "scratch":
        m = cfg["model"]
        model = GRUClassifier(N_FEATURES, m["hidden_size"], m["num_layers"], m["dropout"])
    else:
        m = checkpoint["model_config"]
        model = GRUClassifier(N_FEATURES, m["hidden_size"], m["num_layers"], m["dropout"])  # new binary head
        load_encoder_weights(model, checkpoint["state_dict"])
        if mode == "frozen":
            model.freeze_encoder()
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["scratch"], choices=MODES)
    parser.add_argument("--pretrained", default=None, help="model.pt from pretrain_plasticc.py")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.smoke)
    ccfg, dcfg, ecfg = cfg["cv"], cfg["data"], cfg["eval"]
    seeds = args.seeds or ccfg["seeds"]
    pretrained = args.pretrained or ccfg["pretrained_model"]
    if any(m != "scratch" for m in args.modes) and not pretrained:
        parser.error("--pretrained is required for frozen/finetune modes")
    checkpoint = torch.load(pretrained, map_location="cpu") if pretrained else None
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    cfg["cv_run"] = {"modes": args.modes, "seeds": seeds, "pretrained": pretrained, "smoke": args.smoke}
    run_dir = make_run_dir(ccfg["output_dir"], args.run_name or ("cv_" + "_".join(args.modes)), cfg)
    print(f"Run dir: {run_dir}  device: {device}")

    # ---- data: remove the locked test set FIRST (same split as Stage 0), then build CV folds
    lc, meta = load_mallorn_train(dcfg["mallorn_dir"], dcfg["processed_dir"])
    test_ids = set(make_splits(meta, ccfg["split_seed"], dcfg["n_folds"])["test"])
    dev = meta[~meta["object_id"].isin(test_ids)].reset_index(drop=True)
    dev = stratified_subset(dev, dcfg["max_objects"], ccfg["split_seed"])
    objects = to_objects(lc[lc["object_id"].isin(set(dev["object_id"]))], dev)
    y = dev["target"].to_numpy()
    groups = dev.groupby(["z", "ebv"]).ngroup().to_numpy()
    folds = grouped_stratified_folds(y, groups, ccfg["n_folds"], ccfg["split_seed"])
    assert not test_ids & set(dev["object_id"])
    with open(os.path.join(run_dir, "folds.json"), "w") as f:
        json.dump({f"fold_{k}": dev["object_id"].iloc[idx].tolist() for k, idx in enumerate(folds)}, f)
    print(f"dev objects: {len(dev)} ({y.sum()} TDE) in {len(folds)} folds | locked test objects excluded: "
          f"{len(test_ids)}")

    eval_kwargs = dict(cutoffs_days=ecfg["cutoffs_days"], flux_scale=dcfg["flux_scale"], device=device,
                       peak_bands=ecfg["peak_bands"], peak_min_snr=ecfg["peak_min_snr"])
    all_preds, fold_rows = [], []
    all_idx = np.arange(len(y))

    for mode in args.modes:
        for seed in seeds:
            for k, held in enumerate(folds):
                t0 = time.time()
                set_seed(seed)
                train_idx = np.setdiff1d(all_idx, held)
                inner = grouped_stratified_folds(y[train_idx], groups[train_idx], ccfg["inner_val_folds"], seed)[0]
                stop_idx = train_idx[inner]
                fit_idx = np.setdiff1d(train_idx, stop_idx)
                fit_objs = [objects[i] for i in fit_idx]
                stop_objs = [objects[i] for i in stop_idx]
                held_objs = [objects[i] for i in held]

                model = build_model(mode, cfg, checkpoint, device)
                n_pos = sum(o["label"] for o in fit_objs)
                pos_weight = torch.tensor((len(fit_objs) - n_pos) / max(n_pos, 1), device=device)
                fit_loader = DataLoader(LightCurveDataset(fit_objs, dcfg["flux_scale"]),
                                        batch_size=ccfg["batch_size"], shuffle=True, collate_fn=collate,
                                        generator=torch.Generator().manual_seed(seed))
                stop_loader = DataLoader(LightCurveDataset(stop_objs, dcfg["flux_scale"]), batch_size=256,
                                         shuffle=False, collate_fn=collate)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                              lr=ccfg["lr"][mode], weight_decay=ccfg["weight_decay"])
                result = fit(model, fit_loader, stop_loader, optimizer, pos_weight, device, ccfg["epochs"],
                             ccfg["patience"], cfg["train"]["grad_clip"])
                check_causality(model, held_objs, dcfg["flux_scale"], device, n_checks=5, seed=k)

                preds = predict_at_cutoffs(model, held_objs, **eval_kwargs)
                preds.insert(0, "fold", k)
                preds.insert(0, "seed", seed)
                preds.insert(0, "mode", mode)
                all_preds.append(preds)
                m = metrics_by_cutoff(preds).set_index("cutoff")
                fold_rows.append(m.reset_index().assign(mode=mode, seed=seed, fold=k, best_epoch=result["best_epoch"]))
                print(f"{mode:8s} seed {seed} fold {k} | best epoch {result['best_epoch']:3d} "
                      f"(val loss {result['best_val_loss']:.3f}) | PR-AUC -20d {m.loc['-20d', 'pr_auc']:.3f} "
                      f"0d {m.loc['+0d', 'pr_auc']:.3f} full {m.loc['full', 'pr_auc']:.3f} | {time.time() - t0:.0f}s")

    # ---- aggregate: pooled out-of-fold metrics per (mode, seed), then mean ± std over seeds
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(os.path.join(run_dir, "oof_predictions.csv"), index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(os.path.join(run_dir, "fold_metrics.csv"), index=False)
    pooled = pd.concat([metrics_by_cutoff(g).assign(mode=mode, seed=seed)
                        for (mode, seed), g in preds.groupby(["mode", "seed"], sort=False)], ignore_index=True)
    pooled.to_csv(os.path.join(run_dir, "pooled_metrics_by_seed.csv"), index=False)
    summary = (pooled.groupby(["mode", "cutoff"], sort=False)[["pr_auc", "roc_auc", "f1", "f1_best"]]
               .agg(["mean", "std"]))
    summary.to_csv(os.path.join(run_dir, "summary_mean_std.csv"))

    print(f"\nPooled out-of-fold results, mean +/- std over seeds {seeds} "
          f"(prevalence {y.mean():.3f} = chance PR-AUC)")
    for metric in ["pr_auc", "roc_auc", "f1"]:
        table = pd.DataFrame({
            mode: [f"{summary.loc[(mode, c), (metric, 'mean')]:.3f} +/- {np.nan_to_num(summary.loc[(mode, c), (metric, 'std')]):.3f}"
                   for c in pooled["cutoff"].unique()]
            for mode in args.modes}, index=pooled["cutoff"].unique())
        print(f"\n{metric}:\n{table.to_string()}")


if __name__ == "__main__":
    main()
