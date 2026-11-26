import random
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    return ROOT / path


REPLACE_KEYS = {"model"}


def deep_merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path="configs/baseline.yaml"):
    path = resolve(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    parent = load_config(path.parent / base)
    for key in REPLACE_KEYS & cfg.keys():
        parent.pop(key, None)
    return deep_merge(parent, cfg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(1 - ss_res / ss_tot),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def experiment_dir(cfg):
    path = resolve(cfg["paths"]["reports"]) / cfg["experiment"]
    path.mkdir(parents=True, exist_ok=True)
    return path
