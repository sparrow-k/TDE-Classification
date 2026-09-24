"""NEW post-completion experiment: 14-class pre-training on the FULL PLAsTiCC release.

Same model, same loss, same class weighting, same optimiser, batch size, learning rate, early stopping
(validation loss, patience 10, max 100 epochs), same 80/20 stratified validation split and same seed as the
original `pretrain_plasticc.py`. Only the data changes: the original 7,848-object training set plus the
unblinded PLAsTiCC test set (objects of the 14 training classes). Results go to
results/full_plasticc_pretraining/; the original pipeline and its outputs are not touched.

Usage (from the repository root):
  python pretrain_plasticc_full.py --build-store     # one-off: stream the 11 .csv.gz files into a memmap store
  python pretrain_plasticc_full.py --smoke           # quick end-to-end check on a small subset
  python pretrain_plasticc_full.py                   # the full pre-training run
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pretrain_plasticc import last_step_accuracy  # reused unchanged from the original script
from tde.config import deep_update, make_run_dir
from tde.data import (N_FEATURES, PLASTICC_CODES, PLASTICC_TDE_INDEX, PlasticcFullStore, StoreLightCurveDataset,
                      build_plasticc_full_store, collate, grouped_stratified_folds)
from tde.evaluate import check_causality, metrics_by_cutoff, predict_at_cutoffs
from tde.model import GRUClassifier
from tde.train import evaluate_loss, set_seed, train_one_epoch

CONFIG = "configs/full_plasticc_pretraining.yaml"


def load_experiment_config(path=CONFIG, smoke=False):
    """configs/default.yaml with this experiment's overrides applied on top."""
    with open(path) as f:
        overrides = yaml.safe_load(f)
    with open(overrides.pop("base")) as f:
        cfg = yaml.safe_load(f)
    smoke_overrides = cfg.pop("smoke", None) or {}
    deep_update(cfg, overrides)
    if smoke:
        deep_update(cfg, smoke_overrides)
    return cfg


def build_store(cfg):
    ex = cfg["experiment"]
    files = sorted(glob.glob(ex["test_lc_glob"]))
    if len(files) != ex["n_test_lc_files"]:
        raise SystemExit(f"expected {ex['n_test_lc_files']} light-curve files, found {len(files)}: {files}")
    t0 = time.time()
    manifest = build_plasticc_full_store(cfg["plasticc"]["lc_path"], cfg["plasticc"]["meta_path"],
                                         ex["test_meta_path"], files, ex["store_dir"],
                                         cfg["data"]["processed_dir"])
    manifest["build_seconds"] = round(time.time() - t0)
    manifest["test_light_curve_files"] = [os.path.basename(f) for f in files]
    with open(os.path.join(ex["store_dir"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    os.makedirs(ex["results_dir"], exist_ok=True)
    with open(os.path.join(ex["results_dir"], "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


def fit_with_log(model, train_loader, val_loader, optimizer, weight, device, max_epochs, patience, grad_clip,
                 run_dir):
    """Exactly the stopping rule of `tde.train.fit` (validation loss, min improvement 1e-4, patience, restore
    best), plus a per-epoch history file and best-so-far checkpoint, because one epoch here takes minutes."""
    best_loss, best_state, best_epoch, bad_epochs, history = float("inf"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, weight, device, grad_clip)
        val_loss = evaluate_loss(model, val_loader, weight, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": round(time.time() - t0, 1)})
        improved = val_loss < best_loss - 1e-4
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"{time.time() - t0:.0f}s{' | best' if improved else ''}", flush=True)
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "epoch": epoch}, os.path.join(run_dir, "best_so_far.pt"))
        else:
            bad_epochs += 1
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_run": len(history), "history": history}


def diagnostics(model, store, val_idx, cfg, device, run_dir, flux_scale, workers):
    """Validation diagnostics of the original script: 14-class accuracy, causality guard, TDE-vs-rest cutoffs."""
    ex, ecfg = cfg["experiment"], cfg["eval"]
    val_loader = DataLoader(StoreLightCurveDataset(store, val_idx, flux_scale), batch_size=256, shuffle=False,
                            collate_fn=collate, num_workers=workers, persistent_workers=workers > 0)
    acc, bacc = last_step_accuracy(model, val_loader, device)
    rng = np.random.default_rng(cfg["seed"])
    diag_idx = np.sort(rng.choice(val_idx, size=min(ex["diag_val_objects"], len(val_idx)), replace=False))
    diag_objs = [store.get(int(i)) for i in diag_idx]
    max_diff = check_causality(model, diag_objs, flux_scale, device)
    tde_objs = [{**o, "label": int(o["label"] == PLASTICC_TDE_INDEX)} for o in diag_objs]
    metrics = metrics_by_cutoff(predict_at_cutoffs(model, tde_objs, ecfg["cutoffs_days"], flux_scale, device,
                                                   ecfg["peak_bands"], ecfg["peak_min_snr"]))
    metrics.to_csv(os.path.join(run_dir, "val_tde_metrics_by_cutoff.csv"), index=False)
    return acc, bacc, max_diff, len(diag_idx), metrics


def finalize(cfg, run_dir, workers):
    """Write model.pt and the diagnostics for a run that was interrupted before it could finish.

    The training loop checkpoints every improvement to best_so_far.pt, so the best epoch is recoverable.
    The resulting model.pt is marked `interrupted` and records why, so no later reader can mistake it for a
    run that stopped on its own early-stopping criterion.
    """
    ex, pcfg, mcfg = cfg["experiment"], cfg["plasticc"], cfg["model"]
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    history = json.load(open(os.path.join(run_dir, "history.json")))
    data_info = json.load(open(os.path.join(run_dir, "data_info.json")))
    best = min(history, key=lambda e: e["val_loss"])
    ckpt = torch.load(os.path.join(run_dir, "best_so_far.pt"), map_location="cpu", weights_only=False)
    if ckpt["epoch"] != best["epoch"]:
        raise SystemExit(f"best_so_far.pt is epoch {ckpt['epoch']} but history says {best['epoch']}")

    model_config = {"hidden_size": mcfg["hidden_size"], "num_layers": mcfg["num_layers"],
                    "dropout": mcfg["dropout"], "n_outputs": len(PLASTICC_CODES),
                    "tde_class": PLASTICC_TDE_INDEX}
    model = GRUClassifier(N_FEATURES, **model_config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    flux_scale = pcfg["flux_scale"]
    print(f"restored epoch {best['epoch']} (val loss {best['val_loss']:.4f}) from best_so_far.pt", flush=True)

    store = PlasticcFullStore(ex["store_dir"])
    val_idx = np.load(os.path.join(run_dir, "val_store_indices.npy"))
    acc, bacc, max_diff, n_diag, metrics = diagnostics(model, store, val_idx, cfg, device, run_dir, flux_scale,
                                                       workers)
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "flux_scale": flux_scale,
                "seed": cfg["seed"], "experiment": ex["name"], "n_train": data_info["n_train"],
                "best_epoch": best["epoch"], "interrupted": True,
                "interrupted_note": ("training was stopped by the environment for system memory pressure after "
                                     f"{len(history)} of max {cfg['pretrain']['epochs']} epochs, before its own "
                                     "early-stopping criterion; the best epoch was restored")},
               os.path.join(run_dir, "model.pt"))
    summary = {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "epochs_run": len(history),
               "history": history, "train_minutes": round(sum(e["seconds"] for e in history) / 60, 1),
               "val_accuracy_last_step": acc, "val_balanced_accuracy_last_step": bacc,
               "causality_max_abs_diff": max_diff, "n_train": data_info["n_train"], "n_val": data_info["n_val"],
               "diag_val_objects": int(n_diag), "data": data_info, "interrupted": True,
               "interrupted_after_epochs": len(history),
               "interrupt_reason": "killed by the environment (system memory pressure), not early stopping"}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"causality check passed (max |p_truncated - p_full| = {max_diff:.2e})")
    print(f"PLAsTiCC val 14-class accuracy (last step): {acc:.3f}  balanced accuracy: {bacc:.3f}")
    print(metrics[["cutoff", "n", "n_pos", "pr_auc", "roc_auc", "median_obs_seen"]]
          .to_string(index=False, float_format="%.3f"))
    print(f"\nPre-trained model: {os.path.join(run_dir, 'model.pt')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--build-store", action="store_true", help="build the memory-mapped data store and exit")
    parser.add_argument("--finalize", metavar="RUN_DIR", default=None,
                        help="write model.pt + diagnostics for an interrupted run from its best_so_far.pt")
    parser.add_argument("--workers", type=int, default=None, help="override experiment.num_workers")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-objects", type=int, default=20000)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config, args.smoke)
    if args.workers is not None:
        cfg["experiment"]["num_workers"] = args.workers
    if args.build_store:
        build_store(cfg)
        return
    if args.finalize:
        finalize(cfg, args.finalize, cfg["experiment"]["num_workers"])
        return

    ex, pcfg, ptr, mcfg, ecfg = cfg["experiment"], cfg["plasticc"], cfg["pretrain"], cfg["model"], cfg["eval"]
    seed = cfg["seed"]
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    run_name = f"{ptr['run_name']}{'_smoke' if args.smoke else ''}_seed{seed}"
    run_dir = make_run_dir(ptr["output_dir"], run_name, cfg)
    print(f"Run dir: {run_dir}  device: {device}", flush=True)

    # ---- data: the whole store, then the SAME stratified 80/20 split rule as pretrain_plasticc.py
    store = PlasticcFullStore(ex["store_dir"])
    idx_all = np.arange(len(store))
    if args.smoke:  # stratified-ish small subset, for a quick check of the machinery only
        idx_all = np.random.default_rng(seed).choice(idx_all, size=min(args.smoke_objects, len(idx_all)),
                                                     replace=False)
        idx_all.sort()
    y = store.index["target"].to_numpy()[idx_all].astype(np.int64)
    t0 = time.time()
    val_pos = grouped_stratified_folds(y, np.arange(len(y)), ptr["val_folds"], seed)[0]
    is_val = np.zeros(len(y), dtype=bool)
    is_val[val_pos] = True
    train_idx, val_idx = idx_all[~is_val], idx_all[is_val]
    print(f"split in {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(run_dir, "val_store_indices.npy"), val_idx)

    n_classes = len(PLASTICC_CODES)
    counts = np.bincount(y[~is_val], minlength=n_classes)
    lengths = store.index["length"].to_numpy()
    data_info = {
        "store_dir": ex["store_dir"], "n_objects": int(len(idx_all)), "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)), "n_observations": int(lengths[idx_all].sum()),
        "n_train_observations": int(lengths[train_idx].sum()),
        "train_class_counts": {PLASTICC_CODES[i]: int(c) for i, c in enumerate(counts)},
        "train_tde": int(counts[PLASTICC_TDE_INDEX]),
        "objects_from_original_training_set": int((store.index["source"].to_numpy()[idx_all] == "train").sum()),
    }
    with open(os.path.join(run_dir, "data_info.json"), "w") as f:
        json.dump(data_info, f, indent=2)
    print(f"train: {len(train_idx):,} objects ({counts[PLASTICC_TDE_INDEX]:,} TDE) | val: {len(val_idx):,} | "
          f"observations: {data_info['n_observations']:,}", flush=True)

    # Same "balanced" class weights as the original: every class contributes equally to the loss.
    class_weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32,
                                device=device)
    flux_scale = pcfg["flux_scale"]
    workers = ex["num_workers"]
    loader_kw = dict(collate_fn=collate, num_workers=workers, persistent_workers=workers > 0)
    train_loader = DataLoader(StoreLightCurveDataset(store, train_idx, flux_scale), batch_size=ptr["batch_size"],
                              shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_kw)
    val_loader = DataLoader(StoreLightCurveDataset(store, val_idx, flux_scale), batch_size=256, shuffle=False,
                            **loader_kw)

    model_config = {"hidden_size": mcfg["hidden_size"], "num_layers": mcfg["num_layers"], "dropout": mcfg["dropout"],
                    "n_outputs": n_classes, "tde_class": PLASTICC_TDE_INDEX}
    model = GRUClassifier(N_FEATURES, **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ptr["lr"], weight_decay=ptr["weight_decay"])

    t_train = time.time()
    result = fit_with_log(model, train_loader, val_loader, optimizer, class_weight, device, ptr["epochs"],
                          ptr["patience"], cfg["train"]["grad_clip"], run_dir)
    train_minutes = (time.time() - t_train) / 60
    print(f"best epoch {result['best_epoch']} (val loss {result['best_val_loss']:.4f}), "
          f"{result['epochs_run']} epochs, {train_minutes:.0f} min", flush=True)
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "flux_scale": flux_scale,
                "seed": seed, "experiment": ex["name"], "n_train": int(len(train_idx))},
               os.path.join(run_dir, "model.pt"))

    # ---- diagnostics on PLAsTiCC validation objects (same quantities as the original script)
    acc, bacc = last_step_accuracy(model, val_loader, device)
    rng = np.random.default_rng(seed)
    diag_idx = np.sort(rng.choice(val_idx, size=min(ex["diag_val_objects"], len(val_idx)), replace=False))
    diag_objs = [store.get(int(i)) for i in diag_idx]
    max_diff = check_causality(model, diag_objs, flux_scale, device)
    tde_objs = [{**o, "label": int(o["label"] == PLASTICC_TDE_INDEX)} for o in diag_objs]
    metrics = metrics_by_cutoff(predict_at_cutoffs(model, tde_objs, ecfg["cutoffs_days"], flux_scale, device,
                                                   ecfg["peak_bands"], ecfg["peak_min_snr"]))
    metrics.to_csv(os.path.join(run_dir, "val_tde_metrics_by_cutoff.csv"), index=False)
    summary = {**{k: v for k, v in result.items() if k != "history"}, "history": result["history"],
               "train_minutes": round(train_minutes, 1), "val_accuracy_last_step": acc,
               "val_balanced_accuracy_last_step": bacc, "causality_max_abs_diff": max_diff,
               "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
               "diag_val_objects": int(len(diag_idx)), "data": data_info}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"causality check passed (max |p_truncated - p_full| = {max_diff:.2e})")
    print(f"PLAsTiCC val 14-class accuracy (last step): {acc:.3f}  balanced accuracy: {bacc:.3f}")
    print(metrics[["cutoff", "n", "n_pos", "pr_auc", "roc_auc", "median_obs_seen"]]
          .to_string(index=False, float_format="%.3f"))
    print(f"\nPre-trained model: {os.path.join(run_dir, 'model.pt')}")


if __name__ == "__main__":
    main()
