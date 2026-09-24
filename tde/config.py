"""Config loading (YAML + optional `smoke` overrides) and run-directory creation."""
import os
import time

import yaml


def deep_update(base, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path, smoke=False):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    overrides = cfg.pop("smoke", None) or {}
    if smoke:
        deep_update(cfg, overrides)
    return cfg


def make_run_dir(output_dir, name, cfg):
    """Create outputs/<...>/<timestamp>_<name>/ and store the exact config used."""
    run_dir = os.path.join(output_dir, f"{time.strftime('%Y%m%d-%H%M%S')}_{name}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir
