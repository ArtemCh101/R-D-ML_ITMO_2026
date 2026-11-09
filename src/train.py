import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.data import load_folds, load_test, load_train
from src.models import build_model
from src.utils import compute_metrics, experiment_dir, load_config, resolve, rmse, set_seed


def run_cv(cfg, train, folds):
    X = train["excerpt"].values
    y = train["target"].values
    oof = train[["id", "target", "standard_error"]].copy()
    oof["has_url"] = train["url_legal"].notna()
    fold_rows = []
    for seed in cfg["cv"]["seeds"]:
        set_seed(seed)
        fold_col = folds[f"fold_{seed}"].values
        pred = np.full(len(train), np.nan)
        for k in sorted(np.unique(fold_col)):
            tr = fold_col != k
            va = fold_col == k
            model = build_model(cfg)
            start = time.perf_counter()
            model.fit(X[tr], y[tr])
            fit_time = time.perf_counter() - start
            pred[va] = model.predict(X[va])
            fold_rows.append({
                "seed": seed,
                "fold": int(k),
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                **compute_metrics(y[va], pred[va]),
                "dummy_rmse": rmse(y[va], np.full(va.sum(), y[tr].mean())),
                "fit_time_s": fit_time,
            })
        assert not np.isnan(pred).any()
        oof[f"pred_{seed}"] = pred
    return oof, pd.DataFrame(fold_rows)


def summarize(cfg, oof, fold_df):
    metric_cols = ["rmse", "mae", "r2", "pearson", "spearman", "dummy_rmse", "fit_time_s"]
    per_fold = {c: {"mean": float(fold_df[c].mean()), "std": float(fold_df[c].std(ddof=1))} for c in metric_cols}
    per_seed = {}
    for seed in cfg["cv"]["seeds"]:
        per_seed[str(seed)] = compute_metrics(oof["target"], oof[f"pred_{seed}"])
    seed_rmse = [per_seed[str(s)]["rmse"] for s in cfg["cv"]["seeds"]]
    return {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "cv": cfg["cv"],
        "n_samples": int(len(oof)),
        "per_fold": per_fold,
        "oof_per_seed": per_seed,
        "oof_rmse_across_seeds": {"mean": float(np.mean(seed_rmse)), "std": float(np.std(seed_rmse, ddof=1))},
    }


def fit_full_and_predict(cfg, train):
    model = build_model(cfg)
    model.fit(train["excerpt"].values, train["target"].values)
    model_dir = resolve(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{cfg['experiment']}.joblib")
    test = load_test(cfg)
    return pd.DataFrame({"id": test["id"], "target": model.predict(test["excerpt"].values)})


def main(config_path):
    cfg = load_config(config_path)
    out = experiment_dir(cfg)
    train = load_train(cfg)
    folds = load_folds(cfg)
    assert (folds["id"].values == train["id"].values).all()
    oof, fold_df = run_cv(cfg, train, folds)
    oof.to_csv(out / "oof.csv", index=False)
    fold_df.to_csv(out / "fold_metrics.csv", index=False)
    summary = summarize(cfg, oof, fold_df)
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    fit_full_and_predict(cfg, train).to_csv(out / "submission.csv", index=False)
    pf = summary["per_fold"]
    print(f"{cfg['experiment']}: RMSE {pf['rmse']['mean']:.4f} ± {pf['rmse']['std']:.4f} "
          f"(dummy {pf['dummy_rmse']['mean']:.4f} ± {pf['dummy_rmse']['std']:.4f}), "
          f"{len(fold_df)} folds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
