import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils import load_config, resolve


def load_train(cfg):
    return pd.read_csv(resolve(cfg["paths"]["train"]))


def load_test(cfg):
    return pd.read_csv(resolve(cfg["paths"]["test"]))


def make_folds(df, n_splits, n_bins, seed):
    bins = pd.qcut(df["target"], q=n_bins, labels=False)
    folds = np.full(len(df), -1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(df, bins)):
        folds[val_idx] = k
    return folds


def build_folds(cfg):
    df = load_train(cfg)
    out = df[["id"]].copy()
    for seed in cfg["cv"]["seeds"]:
        out[f"fold_{seed}"] = make_folds(df, cfg["cv"]["n_splits"], cfg["cv"]["n_bins"], seed)
    path = resolve(cfg["paths"]["folds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_folds(cfg):
    return pd.read_csv(resolve(cfg["paths"]["folds"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    print(build_folds(load_config(args.config)).head())
