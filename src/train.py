import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.data import load_folds, load_test, load_train
from src.models import build_model, coefficients, inner_cv_curve, model_inputs, selected_alpha
from src.utils import compute_metrics, experiment_dir, load_config, resolve, rmse, set_seed


def is_experiment(cfg):
    return cfg["model"]["name"] != "tfidf_ridge"


def run_cv(cfg, train, folds):
    X = model_inputs(cfg, train)
    y = train["target"].values
    oof = train[["id", "target", "standard_error"]].copy()
    oof["has_url"] = train["url_legal"].notna()
    fold_rows, alpha_rows, curve_rows, coef_rows = [], [], [], []
    for seed in cfg["cv"]["seeds"]:
        set_seed(seed)
        fold_col = folds[f"fold_{seed}"].values
        pred = np.full(len(train), np.nan)
        for k in sorted(np.unique(fold_col)):
            tr = fold_col != k
            va = fold_col == k
            model = build_model(cfg)
            X_tr = X.iloc[np.where(tr)[0]] if isinstance(X, pd.DataFrame) else X[tr]
            X_va = X.iloc[np.where(va)[0]] if isinstance(X, pd.DataFrame) else X[va]
            start = time.perf_counter()
            model.fit(X_tr, y[tr])
            fit_time = time.perf_counter() - start
            pred[va] = model.predict(X_va)
            row = {
                "seed": seed,
                "fold": int(k),
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                **compute_metrics(y[va], pred[va]),
                "dummy_rmse": rmse(y[va], np.full(va.sum(), y[tr].mean())),
                "fit_time_s": fit_time,
            }
            if is_experiment(cfg):
                row["alpha"] = selected_alpha(model)
                curve = inner_cv_curve(model) if cfg["model"]["tuning"] == "nested_loo" else None
                if curve is not None:
                    alpha_rows.append({"seed": seed, "fold": int(k), "alpha": row["alpha"]})
                    curve_rows += [{"seed": seed, "fold": int(k), "alpha": float(a), "inner_rmse": float(r)} for a, r in curve]
                names, coef = coefficients(model)
                coef_rows += [{"seed": seed, "fold": int(k), "feature": n.split("__", 1)[1], "coef": float(c)}
                              for n, c in zip(names, coef) if n.startswith("feats__")]
            fold_rows.append(row)
        assert not np.isnan(pred).any()
        oof[f"pred_{seed}"] = pred
    extras = {"alphas": pd.DataFrame(alpha_rows), "alpha_curve": pd.DataFrame(curve_rows),
              "feature_coefs": pd.DataFrame(coef_rows)}
    return oof, pd.DataFrame(fold_rows), extras


def summarize(cfg, oof, fold_df):
    metric_cols = ["rmse", "mae", "r2", "pearson", "spearman", "dummy_rmse", "fit_time_s"]
    per_fold = {c: {"mean": float(fold_df[c].mean()), "std": float(fold_df[c].std(ddof=1))} for c in metric_cols}
    per_seed = {}
    for seed in cfg["cv"]["seeds"]:
        per_seed[str(seed)] = compute_metrics(oof["target"], oof[f"pred_{seed}"])
    seed_rmse = [per_seed[str(s)]["rmse"] for s in cfg["cv"]["seeds"]]
    summary = {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "cv": cfg["cv"],
        "n_samples": int(len(oof)),
        "per_fold": per_fold,
        "oof_per_seed": per_seed,
        "oof_rmse_across_seeds": {"mean": float(np.mean(seed_rmse)), "std": float(np.std(seed_rmse, ddof=1))},
    }
    if "alpha" in fold_df:
        a = fold_df["alpha"]
        summary["alpha"] = {"median": float(a.median()), "min": float(a.min()), "max": float(a.max())}
    return summary


def fit_full_and_predict(cfg, train):
    model = build_model(cfg)
    model.fit(model_inputs(cfg, train), train["target"].values)
    model_dir = resolve(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{cfg['experiment']}.joblib")
    test = load_test(cfg)
    return pd.DataFrame({"id": test["id"], "target": model.predict(model_inputs(cfg, test))})


def main(config_path):
    cfg = load_config(config_path)
    out = experiment_dir(cfg)
    train = load_train(cfg)
    folds = load_folds(cfg)
    assert (folds["id"].values == train["id"].values).all()
    if is_experiment(cfg):
        from src.registry import load_experiments, validate
        validate(load_experiments())
        from src.tracking import run_info
        info = run_info(cfg)
    oof, fold_df, extras = run_cv(cfg, train, folds)
    oof.to_csv(out / "oof.csv", index=False)
    fold_df.to_csv(out / "fold_metrics.csv", index=False)
    for name, df in extras.items():
        if not df.empty:
            df.to_csv(out / f"{name}.csv", index=False)
    summary = summarize(cfg, oof, fold_df)
    if is_experiment(cfg):
        summary["run_info"] = info
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    fit_full_and_predict(cfg, train).to_csv(out / "submission.csv", index=False)
    pf = summary["per_fold"]
    msg = (f"{cfg['experiment']}: RMSE {pf['rmse']['mean']:.4f} ± {pf['rmse']['std']:.4f} "
           f"(dummy {pf['dummy_rmse']['mean']:.4f} ± {pf['dummy_rmse']['std']:.4f}), {len(fold_df)} folds")
    if "alpha" in summary and cfg["model"].get("tuning") == "nested_loo":
        a = summary["alpha"]
        msg += f", alpha median {a['median']:.3g} [{a['min']:.3g}, {a['max']:.3g}]"
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
