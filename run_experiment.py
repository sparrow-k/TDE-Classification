"""End-to-end experiment: MALLORN -> features -> GRU training -> validation -> early-time evaluation.

Usage:
  python run_experiment.py --smoke                  # quick check on a 400-object subset (~1 min)
  python run_experiment.py                          # full Stage 0 run
  python run_experiment.py --config configs/default.yaml --seed 1
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tde.data import (N_FEATURES, LightCurveDataset, collate, load_mallorn_train, make_splits, stratified_subset,
                      to_objects)
from tde.evaluate import check_causality, metrics_by_cutoff, predict_at_cutoffs
from tde.model import GRUClassifier
from tde.train import set_seed, train_one_epoch


def deep_update(base, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true", help="small subset, few epochs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    smoke_overrides = cfg.pop("smoke", {})
    if args.smoke:
        deep_update(cfg, smoke_overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    seed = cfg["seed"]
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(cfg["output_dir"], f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['run_name']}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Run dir: {run_dir}  device: {device}")

    # ---- data
    dcfg, ecfg, tcfg = cfg["data"], cfg["eval"], cfg["train"]
    t0 = time.time()
    lc, meta = load_mallorn_train(dcfg["mallorn_dir"], dcfg["processed_dir"])
    meta = stratified_subset(meta, dcfg["max_objects"], seed)
    lc = lc[lc["object_id"].isin(set(meta["object_id"]))]
    splits = make_splits(meta, seed, dcfg["n_folds"])
    with open(os.path.join(run_dir, "splits.json"), "w") as f:
        json.dump(splits, f)
    objects = {o["object_id"]: o for o in to_objects(lc, meta)}
    train_objs = [objects[i] for i in splits["train"]]
    val_objs = [objects[i] for i in splits["val"]]
    for name, objs in [("train", train_objs), ("val", val_objs)]:
        n_pos = sum(o["label"] for o in objs)
        print(f"{name}: {len(objs)} objects, {n_pos} TDE ({n_pos / len(objs):.1%})")
    print(f"test: {len(splits['test'])} objects (held out, not evaluated in this run)")
    print(f"data ready in {time.time() - t0:.1f}s")

    # ---- model and training setup
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(LightCurveDataset(train_objs, dcfg["flux_scale"]), batch_size=tcfg["batch_size"],
                              shuffle=True, collate_fn=collate, generator=g)
    mcfg = cfg["model"]
    model = GRUClassifier(N_FEATURES, mcfg["hidden_size"], mcfg["num_layers"], mcfg["dropout"]).to(device)
    n_pos = sum(o["label"] for o in train_objs)
    pw = (len(train_objs) - n_pos) / max(n_pos, 1) if tcfg["pos_weight"] == "auto" else float(tcfg["pos_weight"])
    pos_weight = torch.tensor(pw, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  pos_weight: {pw:.2f}")

    eval_kwargs = dict(cutoffs_days=ecfg["cutoffs_days"], flux_scale=dcfg["flux_scale"], device=device,
                       peak_bands=ecfg["peak_bands"], peak_min_snr=ecfg["peak_min_snr"])
    select_labels = [f"{c:+d}d" for c in ecfg["select_cutoffs_days"]]

    # ---- training loop with model selection on validation early-time PR-AUC
    history, best_score, best_state, best_epoch = [], -np.inf, None, -1
    for epoch in range(1, tcfg["epochs"] + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, pos_weight, device, tcfg["grad_clip"])
        val_metrics = metrics_by_cutoff(predict_at_cutoffs(model, val_objs, **eval_kwargs)).set_index("cutoff")
        score = float(val_metrics.loc[select_labels, "pr_auc"].mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_select_pr_auc": score,
                        **{f"val_pr_auc_{c}": float(v) for c, v in val_metrics["pr_auc"].items()}})
        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val PR-AUC early(mean) {score:.3f} "
              f"| @0d {val_metrics.loc['+0d', 'pr_auc']:.3f} | full {val_metrics.loc['full', 'pr_auc']:.3f} "
              f"| {time.time() - t_ep:.1f}s")
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "model.pt"))
    print(f"best epoch: {best_epoch} (val early PR-AUC {best_score:.3f})")

    # ---- final validation evaluation, predictions, causality guard
    max_diff = check_causality(model, val_objs, dcfg["flux_scale"], device)
    print(f"causality check passed (max |p_truncated - p_full| = {max_diff:.2e})")
    preds = predict_at_cutoffs(model, val_objs, **eval_kwargs)
    metrics = metrics_by_cutoff(preds)
    preds.to_csv(os.path.join(run_dir, "val_predictions.csv"), index=False)
    metrics.to_csv(os.path.join(run_dir, "val_metrics_by_cutoff.csv"), index=False)
    # Training-set fit, to tell underfitting (train ~ val, both low) from overfitting (train >> val).
    train_metrics = metrics_by_cutoff(predict_at_cutoffs(model, train_objs, **eval_kwargs))
    train_metrics.to_csv(os.path.join(run_dir, "train_metrics_by_cutoff.csv"), index=False)
    print("\nTrain-set fit: " + ", ".join(f"{r.cutoff} PR-AUC {r.pr_auc:.3f}" for r in train_metrics.itertuples()
                                          if r.cutoff in ("+0d", "full")))
    summary = {"best_epoch": best_epoch, "best_val_select_pr_auc": best_score, "pos_weight": pw,
               "causality_max_abs_diff": max_diff, "n_train": len(train_objs), "n_val": len(val_objs),
               "history": history}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with np.printoptions(precision=3):
        print("\nValidation metrics by cutoff (days relative to estimated peak):")
        print(metrics[["cutoff", "n", "n_pos", "pr_auc", "roc_auc", "f1", "precision", "recall", "f1_best",
                       "median_obs_seen", "n_empty"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
