"""Stage 1a: pre-train the GRU on PLAsTiCC (14 classes, prediction at every step).

The saved encoder is later loaded by run_cv.py for MALLORN fine-tuning.

Usage:
  python pretrain_plasticc.py --smoke     # ~10 s check on 800 objects
  python pretrain_plasticc.py             # full pre-training
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader

from tde.config import load_config, make_run_dir
from tde.data import (N_FEATURES, PLASTICC_CODES, PLASTICC_TDE_INDEX, LightCurveDataset, collate,
                      grouped_stratified_folds, load_plasticc_train, stratified_subset, to_objects)
from tde.evaluate import check_causality, metrics_by_cutoff, predict_at_cutoffs
from tde.model import GRUClassifier
from tde.train import fit, set_seed


@torch.no_grad()
def last_step_accuracy(model, loader, device):
    """14-class accuracy and balanced accuracy using the prediction after the final observation."""
    model.eval()
    y_true, y_pred = [], []
    for x, _, labels, lengths in loader:
        logits = model(x.to(device)).cpu()
        last = logits[torch.arange(len(lengths)), lengths - 1]
        y_pred += last.argmax(-1).tolist()
        y_true += labels.long().tolist()
    return accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.smoke)
    seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    pcfg, ptr, mcfg, ecfg = cfg["plasticc"], cfg["pretrain"], cfg["model"], cfg["eval"]
    run_dir = make_run_dir(ptr["output_dir"], f"{ptr['run_name']}_seed{seed}", cfg)
    print(f"Run dir: {run_dir}  device: {device}")

    # ---- data: stratified 80/20 split of PLAsTiCC objects
    lc, meta = load_plasticc_train(pcfg["lc_path"], pcfg["meta_path"], cfg["data"]["processed_dir"])
    meta = stratified_subset(meta, pcfg["max_objects"], seed)
    objects = to_objects(lc[lc["object_id"].isin(set(meta["object_id"]))], meta)
    y = meta["target"].to_numpy()
    is_val = np.zeros(len(y), dtype=bool)
    is_val[grouped_stratified_folds(y, np.arange(len(y)), ptr["val_folds"], seed)[0]] = True
    train_objs = [o for o, v in zip(objects, is_val) if not v]
    val_objs = [o for o, v in zip(objects, is_val) if v]
    n_classes = len(PLASTICC_CODES)
    counts = np.bincount([o["label"] for o in train_objs], minlength=n_classes)
    print(f"train: {len(train_objs)} objects ({counts[PLASTICC_TDE_INDEX]} TDE) | val: {len(val_objs)} objects")

    # "Balanced" class weights: every class contributes equally to the loss despite different counts.
    class_weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32,
                                device=device)

    flux_scale = pcfg["flux_scale"]
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(LightCurveDataset(train_objs, flux_scale), batch_size=ptr["batch_size"], shuffle=True,
                              collate_fn=collate, generator=g)
    val_loader = DataLoader(LightCurveDataset(val_objs, flux_scale), batch_size=256, shuffle=False,
                            collate_fn=collate)

    model_config = {"hidden_size": mcfg["hidden_size"], "num_layers": mcfg["num_layers"], "dropout": mcfg["dropout"],
                    "n_outputs": n_classes, "tde_class": PLASTICC_TDE_INDEX}
    model = GRUClassifier(N_FEATURES, **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ptr["lr"], weight_decay=ptr["weight_decay"])

    # ---- training with early stopping on validation loss
    result = fit(model, train_loader, val_loader, optimizer, class_weight, device, ptr["epochs"], ptr["patience"],
                 cfg["train"]["grad_clip"], log_every=1)
    print(f"best epoch {result['best_epoch']} (val loss {result['best_val_loss']:.4f}), "
          f"{result['epochs_run']} epochs run")
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "flux_scale": flux_scale,
                "seed": seed}, os.path.join(run_dir, "model.pt"))

    # ---- diagnostics on the PLAsTiCC validation objects
    acc, bacc = last_step_accuracy(model, val_loader, device)
    max_diff = check_causality(model, val_objs, flux_scale, device)
    tde_objs = [{**o, "label": int(o["label"] == PLASTICC_TDE_INDEX)} for o in val_objs]  # TDE vs rest
    metrics = metrics_by_cutoff(predict_at_cutoffs(model, tde_objs, ecfg["cutoffs_days"], flux_scale, device,
                                                   ecfg["peak_bands"], ecfg["peak_min_snr"]))
    metrics.to_csv(os.path.join(run_dir, "val_tde_metrics_by_cutoff.csv"), index=False)
    summary = {**{k: v for k, v in result.items() if k != "history"}, "history": result["history"],
               "val_accuracy_last_step": acc, "val_balanced_accuracy_last_step": bacc,
               "causality_max_abs_diff": max_diff, "n_train": len(train_objs), "n_val": len(val_objs)}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"causality check passed (max |p_truncated - p_full| = {max_diff:.2e})")
    print(f"PLAsTiCC val 14-class accuracy (last step): {acc:.3f}  balanced accuracy: {bacc:.3f}")
    print("PLAsTiCC val TDE-vs-rest by cutoff:")
    print(metrics[["cutoff", "n", "n_pos", "pr_auc", "roc_auc", "f1", "median_obs_seen"]]
          .to_string(index=False, float_format="%.3f"))
    print(f"\nPre-trained model: {os.path.join(run_dir, 'model.pt')}")


if __name__ == "__main__":
    main()
