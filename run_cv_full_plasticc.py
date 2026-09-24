"""NEW post-completion experiment: repeat the original MALLORN cross-validation with the full-PLAsTiCC encoder.

Runs the ORIGINAL, unmodified `run_cv.py` (same locked-test exclusion, same 5 grouped-stratified folds, same
seeds 0/1/2, same inner early-stopping split, cutoffs and metrics). Only two things differ from the original
Stage 1 run: the `--pretrained` checkpoint and the output folder. The locked MALLORN test set is removed by
run_cv.py before anything else and is never evaluated here.

Two runs:
  baseline_repro   original PLAsTiCC encoder, modes scratch/frozen/finetune. Must reproduce the original
                   Stage 1 CV numbers exactly; it proves the downstream protocol is unchanged.
  full_plasticc    full-PLAsTiCC encoder, modes frozen/finetune (scratch does not depend on pre-training).

Usage:  python run_cv_full_plasticc.py --pretrained results/full_plasticc_pretraining/pretrain/<run>/model.pt
"""
import argparse
import os
import subprocess
import sys

import yaml

from pretrain_plasticc_full import CONFIG, load_experiment_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True, help="model.pt written by pretrain_plasticc_full.py")
    parser.add_argument("--skip-baseline-repro", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(CONFIG)
    ex = cfg["experiment"]
    os.makedirs(ex["results_dir"], exist_ok=True)
    merged = os.path.join(ex["results_dir"], "cv_config_merged.yaml")
    with open(merged, "w") as f:  # a complete config file, so run_cv.py needs no changes
        yaml.safe_dump(cfg, f, sort_keys=False)

    runs = [] if args.skip_baseline_repro else [
        ("baseline_repro", ex["baseline_pretrain"], ["scratch", "frozen", "finetune"])]
    runs.append(("full_plasticc", args.pretrained, ["frozen", "finetune"]))
    for name, ckpt, modes in runs:
        cmd = [sys.executable, "-u", "run_cv.py", "--config", merged, "--pretrained", ckpt,
               "--modes", *modes, "--run-name", name]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
