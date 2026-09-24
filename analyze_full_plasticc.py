"""NEW post-completion experiment: compare original vs full-PLAsTiCC pre-training on the MALLORN CV.

Reads only saved outputs (no training):
  baseline   outputs/cv/20260913-015131_stage1_cv_all_modes                 (the original Stage 1 CV, EXP-006)
  repro      results/full_plasticc_pretraining/cv/*_baseline_repro          (original encoder, re-run here)
  full       results/full_plasticc_pretraining/cv/*_full_plasticc           (full-PLAsTiCC encoder)
  pretrain   results/full_plasticc_pretraining/pretrain/*_pretrain_plasticc_full_seed42
Writes tables to results/full_plasticc_pretraining/ and figures to results/full_plasticc_pretraining/figures/.

Both CV runs use the same folds and the same seeds, so differences are also reported per seed (paired).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pretrain_plasticc_full import CONFIG, load_experiment_config

CUTOFFS = ["-50d", "-20d", "-10d", "+0d", "+20d", "+50d", "+100d", "full"]
LABELS = ["-50", "-20", "-10", "peak", "+20", "+50", "+100", "full"]
KEY = ["-20d", "+0d", "+50d", "full"]
X = np.arange(len(CUTOFFS))
OLD, NEW, SCRATCH, GBM, RED = "#1f4e79", "#c0392b", "#d1651d", "#7a7a7a", "#b3242b"
EXP_TITLE = "Full-PLAsTiCC pre-training experiment"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False})


def latest(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"nothing matches {pattern}")
    return hits[-1]


def pooled(run_dir):
    return pd.read_csv(os.path.join(run_dir, "pooled_metrics_by_seed.csv"))


def subproblem_roc(run_dir):
    """TDE-vs-AGN and TDE-vs-non-AGN ROC-AUC per (mode, seed, cutoff), as in diagnostics/cv_class_breakdown.py."""
    p = pd.read_csv(os.path.join(run_dir, "oof_predictions.csv"))
    p["prob"] = p["prob"].fillna(0.0)
    rows = []
    for (mode, seed, cutoff), s in p.groupby(["mode", "seed", "cutoff"], sort=False):
        agn = s[(s["label"] == 1) | (s["spectype"] == "AGN")]
        oth = s[(s["label"] == 1) | (s["spectype"] != "AGN")]
        rows.append({"mode": mode, "seed": seed, "cutoff": cutoff,
                     "roc_tde_vs_agn": roc_auc_score(agn["label"], agn["prob"]),
                     "roc_tde_vs_other": roc_auc_score(oth["label"], oth["prob"])})
    return pd.DataFrame(rows), p


def mean_std(df, mode, metric):
    g = df[df["mode"] == mode].groupby("cutoff")[metric]
    return g.mean().reindex(CUTOFFS), g.std().reindex(CUTOFFS)


def main():
    cfg = load_experiment_config(CONFIG)
    ex = cfg["experiment"]
    out = ex["results_dir"]
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    base_dir = ex["baseline_cv"]
    repro_dir = latest(os.path.join(out, "cv", "*_baseline_repro"))
    full_dir = latest(os.path.join(out, "cv", "*_full_plasticc"))
    pre_dir = latest(os.path.join(out, "pretrain", "*_pretrain_plasticc_full_seed*"))
    base, repro, full = pooled(base_dir), pooled(repro_dir), pooled(full_dir)
    gbm = pd.read_csv(os.path.join(base_dir, "gbm_reference_metrics.csv")).set_index("cutoff")

    # ---- 1. reproduction check: the original encoder through the re-run protocol must match the original CV
    keycols = ["mode", "seed", "cutoff"]
    m = base.merge(repro, on=keycols, suffixes=("_orig", "_repro"))
    repro_diff = {met: float((m[f"{met}_orig"] - m[f"{met}_repro"]).abs().max()) for met in ["pr_auc", "roc_auc"]}
    repro_rows = len(m)

    # ---- 2. comparison table (mean +/- std over seeds 0,1,2) and paired per-seed differences
    rows, paired = [], []
    for mode in ["frozen", "finetune"]:
        for name, df, data in [("original pre-training", base, "PLAsTiCC training set (7,848 objects)"),
                               ("full-PLAsTiCC pre-training", full, "full PLAsTiCC release")]:
            for metric in ["pr_auc", "roc_auc"]:
                mu, sd = mean_std(df, mode, metric)
                rows.append({"experiment": name, "plasticc_data": data, "mallorn_mode": mode, "metric": metric,
                             **{c: f"{mu[c]:.3f} +/- {sd[c]:.3f}" for c in CUTOFFS},
                             **{f"{c}_mean": mu[c] for c in CUTOFFS}, **{f"{c}_std": sd[c] for c in CUTOFFS}})
        for metric in ["pr_auc", "roc_auc"]:
            b = base[base["mode"] == mode].set_index(["seed", "cutoff"])[metric]
            f = full[full["mode"] == mode].set_index(["seed", "cutoff"])[metric]
            d = (f - b).unstack("cutoff")[CUTOFFS]
            for c in CUTOFFS:
                paired.append({"mallorn_mode": mode, "metric": metric, "cutoff": c,
                               "delta_mean": d[c].mean(), "delta_seed0": d[c].loc[0], "delta_seed1": d[c].loc[1],
                               "delta_seed2": d[c].loc[2], "seeds_improved": int((d[c] > 0).sum()),
                               "baseline_seed_std": b.unstack("cutoff")[c].std()})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(out, "comparison_all_cutoffs.csv"), index=False)
    paired = pd.DataFrame(paired)
    paired.to_csv(os.path.join(out, "comparison_paired_deltas.csv"), index=False)

    # ---- 3. per-class / sub-problem behaviour
    sub_base, preds_base = subproblem_roc(base_dir)
    sub_full, preds_full = subproblem_roc(full_dir)
    sub = pd.concat([sub_base.assign(experiment="original"), sub_full.assign(experiment="full_plasticc")])
    sub.to_csv(os.path.join(out, "subproblem_roc_by_seed.csv"), index=False)
    cls = []
    for name, p in [("original", preds_base), ("full_plasticc", preds_full)]:
        g = p[(p["mode"] == "frozen") & (p["cutoff"].isin(["-20d", "full"]))]
        t = g.groupby(["cutoff", "spectype"])["prob"].mean().unstack("cutoff")
        t.columns = [f"{name}_{c}" for c in t.columns]
        cls.append(t)
    cls = pd.concat(cls, axis=1)
    cls["n_objects"] = preds_base[preds_base["mode"] == "frozen"].groupby("spectype")["object_id"].nunique()
    cls = cls.sort_values("n_objects", ascending=False)
    cls.to_csv(os.path.join(out, "class_mean_prob_frozen.csv"))

    # ---- 4. pre-training summaries
    new_pre = json.load(open(os.path.join(pre_dir, "summary.json")))
    old_pre = json.load(open(os.path.join(ex["baseline_pretrain_dir"], "summary.json")))
    new_diag = pd.read_csv(os.path.join(pre_dir, "val_tde_metrics_by_cutoff.csv")).set_index("cutoff")
    old_diag = pd.read_csv(os.path.join(ex["baseline_pretrain_dir"], "val_tde_metrics_by_cutoff.csv")).set_index("cutoff")

    # ---- figures ----------------------------------------------------------------------------------------
    def ax_cutoffs(ax, ylabel):
        ax.set_xticks(X)
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("evaluation cutoff (days relative to peak)")
        ax.set_ylabel(ylabel)
        ax.axvline(3, color="#dddddd", lw=0.8, zorder=0)

    # A / B: PR-AUC and ROC-AUC vs cutoff, frozen (primary) and fine-tuned
    for tag, metric, ylabel, chance in [("A", "pr_auc", "PR-AUC", 0.0485), ("B", "roc_auc", "ROC-AUC", 0.5)]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, mode in zip(axes, ["frozen", "finetune"]):
            for df, col, lab in [(base, OLD, "original pre-training (7,848 PLAsTiCC objects)"),
                                 (full, NEW, f"full-PLAsTiCC pre-training ({new_pre['n_train']:,} train objects)")]:
                mu, sd = mean_std(df, mode, metric)
                ax.plot(X, mu, "-o", ms=3.5, color=col, label=lab)
                ax.fill_between(X, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
            mu, _ = mean_std(base, "scratch", metric)
            ax.plot(X, mu, ":", color=SCRATCH, label="from scratch (no pre-training)")
            if metric == "pr_auc":
                ax.plot(X, gbm["pr_auc"].reindex(CUTOFFS), "--", color=GBM, label="gradient boosting")
            ax.axhline(chance, color=RED, ls=":", lw=0.9)
            ax.set_title(f"MALLORN CV, {mode} encoder" + (" (primary)" if mode == "frozen" else ""), loc="left")
            ax_cutoffs(ax, ylabel)
        axes[0].legend(loc="upper left", fontsize=7.5)
        fig.suptitle(f"{EXP_TITLE}: {ylabel} vs time relative to peak "
                     f"(pooled out-of-fold, mean ± std over seeds 0-2; dotted red = chance)", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig_{tag}_{metric}_vs_cutoff.png"))
        plt.close(fig)

    # C: pre-training curves (new run, and the original run for reference; different validation sets)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, s, title in [(axes[0], new_pre, f"full PLAsTiCC ({new_pre['n_train']:,} train / {new_pre['n_val']:,} val)"),
                         (axes[1], old_pre, f"original ({old_pre['n_train']:,} train / {old_pre['n_val']:,} val)")]:
        h = pd.DataFrame(s["history"])
        ax.plot(h["epoch"], h["train_loss"], "-o", ms=2.5, label="train loss")
        ax.plot(h["epoch"], h["val_loss"], "-s", ms=2.5, label="validation loss")
        ax.axvline(s["best_epoch"], color="#999999", ls="--", lw=0.9)
        ax.text(s["best_epoch"], ax.get_ylim()[1], f" best epoch {s['best_epoch']}", va="top", fontsize=8,
                color="#555555")
        ax.set_xlabel("epoch")
        ax.set_ylabel("class-weighted per-step cross-entropy")
        ax.set_title(f"14-class pre-training: {title}", loc="left", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(f"{EXP_TITLE}: pre-training loss curves (validation sets differ, so levels are not directly "
                 f"comparable)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_C_pretraining_curves.png"))
    plt.close(fig)

    # D: sub-problems (frozen), original vs full
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, metric, title in [(axes[0], "roc_tde_vs_agn", "TDE vs AGN"),
                              (axes[1], "roc_tde_vs_other", "TDE vs supernova-like (non-AGN) classes")]:
        for df, col, lab in [(sub_base, OLD, "original pre-training"), (sub_full, NEW, "full-PLAsTiCC pre-training")]:
            mu, sd = mean_std(df, "frozen", metric)
            ax.plot(X, mu, "-o", ms=3.5, color=col, label=lab)
            ax.fill_between(X, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ax.axhline(0.5, color=RED, ls=":", lw=0.9)
        ax.set_title(f"{title}, frozen encoder", loc="left")
        ax_cutoffs(ax, "ROC-AUC")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{EXP_TITLE}: what changes by sub-problem (MALLORN CV, mean ± std over seeds)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_D_subproblem_roc.png"))
    plt.close(fig)

    # E: paired per-seed differences (full minus original), frozen and fine-tuned, PR-AUC
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, mode in zip(axes, ["frozen", "finetune"]):
        d = paired[(paired["mallorn_mode"] == mode) & (paired["metric"] == "pr_auc")].set_index("cutoff").loc[CUTOFFS]
        for k, mk in zip([0, 1, 2], ["o", "s", "^"]):
            ax.plot(X + (k - 1) * 0.12, d[f"delta_seed{k}"], mk, ms=4.5, color=NEW, alpha=0.75,
                    label=f"seed {k}" if mode == "frozen" else None)
        ax.bar(X, d["delta_mean"], width=0.55, color=NEW, alpha=0.18, label="mean" if mode == "frozen" else None)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{mode} encoder: PR-AUC(full) - PR-AUC(original), same folds and seeds", loc="left",
                     fontsize=9)
        ax_cutoffs(ax, "paired difference in PR-AUC")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{EXP_TITLE}: paired per-seed change (positive = full PLAsTiCC better)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_E_paired_delta_pr_auc.png"))
    plt.close(fig)

    # ---- markdown table + machine-readable summary -----------------------------------------------------
    def cell(name, mode, metric, c):
        r = comp[(comp["experiment"] == name) & (comp["mallorn_mode"] == mode) & (comp["metric"] == metric)].iloc[0]
        return r[c]

    lines = ["| Experiment | PLAsTiCC data | MALLORN setup | -20 d | peak | +50 d | full curve |",
             "|---|---|---|--:|--:|--:|--:|"]
    for metric, mname in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]:
        for mode in ["frozen", "finetune"]:
            for name, data in [("original pre-training", "training set, 7,848 objects"),
                               ("full-PLAsTiCC pre-training", f"full release, {new_pre['data']['n_objects']:,} objects")]:
                lines.append(f"| {name} ({mname}) | {data} | {mode}, same CV protocol | "
                             + " | ".join(cell(name, mode, metric, c) for c in KEY) + " |")
    with open(os.path.join(out, "comparison_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    summary = {
        "runs": {"baseline_cv": base_dir, "baseline_repro_cv": repro_dir, "full_plasticc_cv": full_dir,
                 "full_plasticc_pretrain": pre_dir, "baseline_pretrain": ex["baseline_pretrain_dir"]},
        "reproduction_check": {"rows_compared": repro_rows, "max_abs_diff": repro_diff},
        "pretrain_new": {k: new_pre[k] for k in ["best_epoch", "epochs_run", "best_val_loss", "train_minutes",
                                                   "val_accuracy_last_step", "val_balanced_accuracy_last_step",
                                                   "causality_max_abs_diff", "n_train", "n_val", "diag_val_objects"]},
        "pretrain_old": {k: old_pre[k] for k in ["best_epoch", "epochs_run", "best_val_loss",
                                                   "val_accuracy_last_step", "val_balanced_accuracy_last_step",
                                                   "n_train", "n_val"]},
        "pretrain_val_tde_pr_auc": {"new": new_diag["pr_auc"].to_dict(), "old": old_diag["pr_auc"].to_dict(),
                                    "new_prevalence": float(new_diag["prevalence"].iloc[0]),
                                    "old_prevalence": float(old_diag["prevalence"].iloc[0])},
        "pretrain_val_tde_roc_auc": {"new": new_diag["roc_auc"].to_dict(), "old": old_diag["roc_auc"].to_dict()},
        "cv_frozen_pr_auc": {n: {c: float(mean_std(df, "frozen", "pr_auc")[0][c]) for c in CUTOFFS}
                             for n, df in [("original", base), ("full_plasticc", full)]},
        "cv_frozen_pr_auc_std": {n: {c: float(mean_std(df, "frozen", "pr_auc")[1][c]) for c in CUTOFFS}
                                 for n, df in [("original", base), ("full_plasticc", full)]},
        "cv_finetune_pr_auc": {n: {c: float(mean_std(df, "finetune", "pr_auc")[0][c]) for c in CUTOFFS}
                               for n, df in [("original", base), ("full_plasticc", full)]},
        "cv_frozen_roc_auc": {n: {c: float(mean_std(df, "frozen", "roc_auc")[0][c]) for c in CUTOFFS}
                              for n, df in [("original", base), ("full_plasticc", full)]},
        "subproblem_frozen_mean": {n: {m_: mean_std(df, "frozen", m_)[0].round(4).to_dict()
                                       for m_ in ["roc_tde_vs_agn", "roc_tde_vs_other"]}
                                   for n, df in [("original", sub_base), ("full_plasticc", sub_full)]},
    }
    with open(os.path.join(out, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n".join(lines))
    print("\nreproduction check:", summary["reproduction_check"])
    print(paired[paired["metric"] == "pr_auc"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
