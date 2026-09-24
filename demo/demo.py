"""Out-of-the-box demo: streaming early classification of a light curve with the FINAL trained model.

    cd demo
    python demo.py                      # runs the bundled examples in examples/
    python demo.py my_lightcurve.csv    # runs your own CSV (columns: time,flux,flux_err,band)

Nothing is trained here. The demo loads the exact checkpoint that produced the final held-out test
results (`outputs/final_test/20260916-172737_final_test/models/frozen_seed0.pt`, the frozen-transfer
model, seed 0) and feeds it light curves one observation at a time.

Preprocessing is NOT re-implemented: the CSV is turned into the same arrays the training pipeline uses
(`read_lightcurve_csv` below is a thin reader), and the features and the per-step predictions come from
the project code itself - `tde.data.build_features` via `tde.evaluate.predict_steps`.

Because the GRU is unidirectional, the probability printed after observation k depends only on
observations 1..k. That is the point of the plot: the curve on the bottom panel is what the model would
have said in real time, as the data arrived.
"""
import argparse
import json
import os
import sys
import textwrap

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(DEMO_DIR, ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import torch

from tde.data import BANDS, BAND_TO_INDEX, N_FEATURES
from tde.evaluate import predict_steps
from tde.model import GRUClassifier

# ---- the final model of the locked held-out evaluation (Stage 2, EXP-008); see demo/README.md
CHECKPOINT = os.path.join(REPO, "outputs", "final_test", "20260916-172737_final_test",
                          "models", "frozen_seed0.pt")
EXAMPLES_DIR = os.path.join(DEMO_DIR, "examples")
OUTPUT_DIR = os.path.join(DEMO_DIR, "output")

# MALLORN fluxes are used as they are: the model was trained with this fixed scale (configs/default.yaml).
FLUX_SCALE = 1.0
THRESHOLD = 0.5          # the project's default operating point
MILESTONES = (1, 5, 10, 25, 50, 100, 150)   # observation counts printed in the terminal

# Plot style: same band colours as the presentation figures, plus a marker per band so six
# overlapping series stay readable on a projector. (Styling only - no effect on the model.)
BAND_STYLE = {"u": ("#4a3aa7", "v"), "g": ("#2a78d6", "o"), "r": ("#eb6834", "s"),
              "i": ("#1baf7a", "^"), "z": ("#e87ba4", "D"), "y": ("#eda100", "P")}
MODEL_COLOR, INK2, MUTED, GRID = "#2a78d6", "#52514e", "#898781", "#e1e0d9"


# ----------------------------------------------------------------------------- input

def read_lightcurve_csv(path):
    """Read a human-readable light-curve CSV into the arrays the project pipeline uses.

    Required columns: time (MJD), flux, flux_err, band (u/g/r/i/z/y, or 0-5).
    The three steps below are exactly what `tde.data.load_mallorn_train` does to the raw survey
    tables: drop rows without a flux measurement, map the filter to its index, and sort by
    (time, band) so that "everything observed up to time t" is always a prefix of the arrays.
    """
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {"mjd": "time", "t": "time", "err": "flux_err", "fluxerr": "flux_err",
               "flux_error": "flux_err", "filter": "band", "passband": "band"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})

    missing = [c for c in ("time", "flux", "flux_err", "band") if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing column(s) {missing}. "
                         f"Expected a header line: time,flux,flux_err,band")

    df = df.dropna(subset=["time", "flux", "flux_err", "band"])
    if df.empty:
        raise ValueError(f"{os.path.basename(path)}: no usable rows (every row had a missing value)")

    band = df["band"].astype(str).str.strip()
    if band.str.fullmatch(r"[0-5]").all():          # numeric band indices are accepted too
        band_idx = band.astype(int)
    else:
        unknown = sorted(set(band) - set(BAND_TO_INDEX))
        if unknown:
            raise ValueError(f"{os.path.basename(path)}: unknown filter(s) {unknown}. "
                             f"Use one of {BANDS} (or the indices 0-5)")
        band_idx = band.map(BAND_TO_INDEX)

    df = df.assign(band=band_idx.to_numpy()).sort_values(["time", "band"], kind="mergesort")
    return {
        "object_id": os.path.splitext(os.path.basename(path))[0],
        "mjd": df["time"].to_numpy(np.float64),
        "flux": df["flux"].to_numpy(np.float32),
        "flux_err": df["flux_err"].to_numpy(np.float32),
        "band": df["band"].to_numpy(np.int64),
    }


def load_example_info():
    """Descriptions of the bundled examples (true class, one-line comment). Empty for user files."""
    path = os.path.join(EXAMPLES_DIR, "examples.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    return {e["file"]: e for e in info.get("examples", [])}


# ----------------------------------------------------------------------------- model

def load_final_model(device="cpu"):
    """Load the frozen-transfer seed-0 checkpoint used for the final held-out evaluation.

    The stored `model_config` comes from the PLAsTiCC pre-training run, so it still says
    `n_outputs: 14`; the fine-tuned weights are binary. We therefore build the binary model
    (as `run_cv.build_model` does) and check the head shape before trusting the checkpoint.
    """
    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}\n"
                         f"It is tracked in this repository; run the demo from a full clone.")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = GRUClassifier(N_FEATURES, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])

    head_shape = tuple(ckpt["state_dict"]["head.3.weight"].shape)
    if head_shape != (1, cfg["hidden_size"]):
        raise SystemExit(f"Unexpected checkpoint: binary head expected, got head.3.weight {head_shape}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {ckpt['mode']} transfer, seed {ckpt['seed']}, best epoch {ckpt['best_epoch']}, "
          f"{n_params:,} parameters (GRU {cfg['num_layers']}x{cfg['hidden_size']}, unidirectional)")
    print(f"       trained on {ckpt['n_fit']:,} + {ckpt['n_early_stop']:,} development objects "
          f"at commit {ckpt['git_commit'][:7]}")
    print(f"       checkpoint: {os.path.relpath(CHECKPOINT, REPO)}")
    return model


# ----------------------------------------------------------------------------- plotting

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figure(plt, obj, probs, title, subtitle):
    """Top: the light curve. Bottom: P(TDE) after each observation, on the same time axis."""
    days = obj["mjd"] - obj["mjd"][0]
    fig, (ax_lc, ax_p) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [1.2, 1]})

    for name, (color, marker) in BAND_STYLE.items():
        sel = obj["band"] == BAND_TO_INDEX[name]
        if sel.any():
            ax_lc.errorbar(days[sel], obj["flux"][sel], yerr=obj["flux_err"][sel], fmt=marker, ms=4,
                           color=color, ecolor=color, elinewidth=0.9, alpha=0.9, linestyle="none",
                           label=f"{name} band")
    ax_lc.set_ylabel("flux", color=INK2)
    ax_lc.legend(frameon=False, loc="upper right", fontsize=8, ncol=3, labelcolor=INK2)
    ax_lc.set_title(title, loc="left", fontsize=12)
    style_axes(ax_lc)

    # P(TDE) only changes when an observation arrives, so it is a step function.
    ax_p.step(days, probs, where="post", color=MODEL_COLOR, linewidth=2)
    ax_p.plot(days, probs, ".", ms=3.5, color=MODEL_COLOR, alpha=0.7)
    ax_p.axhline(THRESHOLD, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_p.text(days[-1], THRESHOLD + 0.02, f"threshold {THRESHOLD}", color=MUTED, ha="right",
              va="bottom", fontsize=8)
    # Milestone markers, skipping any label that would sit on top of the previous one.
    shown = [n for n in MILESTONES if n <= len(probs)] + [len(probs)]
    last_x = -np.inf
    for n in shown:
        x, y = days[n - 1], probs[n - 1]
        ax_p.plot(x, y, "o", ms=6, mfc="white", mec=MODEL_COLOR, mew=1.6, zorder=4)
        if x - last_x > 0.045 * (days[-1] - days[0] + 1e-9):
            ax_p.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                          fontsize=8, color=INK2)
            last_x = x
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("P(TDE)", color=INK2)
    ax_p.set_xlabel("days since the first observation   (labels = number of observations seen)", color=INK2)
    style_axes(ax_p)

    head = "Each prediction uses only the observations available at that time"
    text = head + (f"  |  {subtitle}" if subtitle else "")
    fig.suptitle(textwrap.fill(text, 118), x=0.02, y=0.985, ha="left", va="top",
                 fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ----------------------------------------------------------------------------- one light curve

def run_one(plt, model, path, info, save_dir, device):
    obj = read_lightcurve_csv(path)
    probs = predict_steps(model, obj, FLUX_SCALE, device)     # real project code, one pass, causal
    n = len(probs)
    days = obj["mjd"] - obj["mjd"][0]
    name = os.path.basename(path)
    entry = info.get(name, {})
    truth = entry.get("true_class")

    print(f"\n--- {name} " + "-" * max(0, 58 - len(name)))
    if entry.get("description"):
        print(f"    {entry['description']}")
    bands_present = " ".join(b for b in BANDS if (obj["band"] == BAND_TO_INDEX[b]).any())
    print(f"    {n} observations over {days[-1]:.0f} days | bands: {bands_present}"
          + (f" | true class: {truth}" if truth else ""))
    for m in [m for m in MILESTONES if m <= n]:
        print(f"      after {m:4d} observation{'s' if m > 1 else ' '} "
              f"({days[m - 1]:6.0f} d): P(TDE) = {probs[m - 1]:.2f}")
    verdict = "flagged as a TDE candidate" if probs[-1] >= THRESHOLD else "not flagged"
    print(f"      after all {n:4d} observations ({days[-1]:6.0f} d): P(TDE) = {probs[-1]:.2f}  ->  {verdict}"
          f" (threshold {THRESHOLD})")

    title = f"{name}" + (f"  -  true class: {truth}" if truth else "")
    fig = make_figure(plt, obj, probs, title, entry.get("description", ""))
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"demo_{os.path.splitext(name)[0]}.png")
    fig.savefig(out, dpi=140)
    shown_path = os.path.relpath(out, REPO)
    print(f"      figure: {out if shown_path.startswith('..') else shown_path}")
    return {"file": name, "true_class": truth or "?", "n_obs": n,
            "p_first": float(probs[0]), "p_final": float(probs[-1])}


def main():
    parser = argparse.ArgumentParser(
        description="Streaming TDE classification with the project's final trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="*", help="light-curve CSV(s); default: the bundled examples/")
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening windows")
    parser.add_argument("--save-dir", default=OUTPUT_DIR, help="where the figures are written")
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = args.csv or sorted(os.path.join(EXAMPLES_DIR, f)
                               for f in os.listdir(EXAMPLES_DIR) if f.endswith(".csv"))
    if not paths:
        raise SystemExit(f"No light curves to run. Put a CSV in {EXAMPLES_DIR} or pass one on the command line.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("Early-epoch TDE classification - prediction after every observation")
    print("=" * 78)
    model = load_final_model(device)
    print(f"       device: {device}\n")

    info = load_example_info()
    summary = []
    for path in paths:
        try:
            summary.append(run_one(plt, model, path, info, args.save_dir, device))
        except (ValueError, FileNotFoundError) as err:   # a readable message beats a traceback
            print(f"\n--- {os.path.basename(path)}\n    SKIPPED: {err}")
    if not summary:
        raise SystemExit("\nNothing could be read. See demo/README.md for the expected CSV format.")

    print("\n" + "=" * 78)
    print(f"{'file':26s} {'true class':11s} {'n obs':>6s} {'P after 1':>10s} {'P at end':>9s}")
    for s in summary:
        print(f"{s['file']:26s} {s['true_class']:11s} {s['n_obs']:6d} {s['p_first']:10.2f} {s['p_final']:9.2f}")
    print("=" * 78)
    print("The bundled examples are development objects the final model was trained on: they show how the\n"
          "model behaves, not how well it generalises. The measured performance is the locked-test result\n"
          "in the project report (full-curve PR-AUC 0.62, 95% CI 0.46-0.78).")

    if not args.no_show:
        print("\nClose the figure windows to finish.")
        plt.show()


if __name__ == "__main__":
    main()
