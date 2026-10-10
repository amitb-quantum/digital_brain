"""Load a tier config deep-merged on top of config/default.yaml."""
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    path = Path(path)
    if path.resolve() != (CONFIG_DIR / "default.yaml").resolve():
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg
