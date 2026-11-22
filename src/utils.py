import random
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    return ROOT / path


def load_config(path="configs/baseline.yaml"):
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


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
