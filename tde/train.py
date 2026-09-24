"""Seeding, masked per-step loss (binary or multi-class), epoch loop and early stopping."""
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    # Must be set before the first cuBLAS call for deterministic CUDA kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def masked_step_loss(logits, labels, mask, weight):
    """Cross-entropy at every real (unpadded) time step.

    Binary head:      logits (B, T),    labels 0/1,          weight = pos_weight (scalar tensor).
    Multi-class head: logits (B, T, C), labels class index,  weight = per-class weights (C,).

    Every step of an object is trained towards that object's label, so the model learns to
    classify every prefix of the light curve (this is what makes early predictions possible).
    Losses are averaged within each sequence first, so long light curves do not dominate.
    """
    if logits.dim() == 2:
        targets = labels[:, None].expand_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight, reduction="none")
    else:
        b, t, c = logits.shape
        targets = labels.long()[:, None].expand(b, t)
        loss = F.cross_entropy(logits.reshape(b * t, c), targets.reshape(b * t), weight=weight,
                               reduction="none").reshape(b, t)
    mask = mask.float()
    per_sequence = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return per_sequence.mean()


def train_one_epoch(model, loader, optimizer, weight, device, grad_clip=1.0):
    model.train()
    total, n = 0.0, 0
    for x, mask, labels, _ in loader:
        x, mask, labels = x.to(device), mask.to(device), labels.to(device)
        loss = masked_step_loss(model(x), labels, mask, weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss(model, loader, weight, device):
    model.eval()
    total, n = 0.0, 0
    for x, mask, labels, _ in loader:
        loss = masked_step_loss(model(x.to(device)), labels.to(device), mask.to(device), weight)
        total += loss.item() * len(labels)
        n += len(labels)
    return total / max(n, 1)


def fit(model, train_loader, val_loader, optimizer, weight, device, max_epochs, patience, grad_clip=1.0,
        log_every=0, log_prefix=""):
    """Train with early stopping on validation LOSS, then restore the best weights.

    Validation loss is used (not PR-AUC) because it is a smooth, continuous criterion and is far
    less noisy than a ranking metric computed from a couple of dozen TDEs.
    """
    best_loss, best_state, best_epoch, bad_epochs, history = float("inf"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, weight, device, grad_clip)
        val_loss = evaluate_loss(model, val_loader, weight, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if log_every and epoch % log_every == 0:
            print(f"{log_prefix}epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_run": len(history), "history": history}
