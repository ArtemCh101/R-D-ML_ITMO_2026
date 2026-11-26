import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_train
from src.utils import experiment_dir, load_config, resolve, rmse


def md_table(df, floatfmt="{:.4f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bootstrap_rmse(y, p, iters, ci, seed):
    rng = np.random.default_rng(seed)
    sq = (np.asarray(y) - np.asarray(p)) ** 2
    idx = rng.integers(0, len(sq), size=(iters, len(sq)))
    samples = np.sqrt(sq[idx].mean(axis=1))
    alpha = (1 - ci) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def quantile_bins(x, q):
    codes, edges = pd.qcut(x, q, labels=False, retbins=True, duplicates="drop")
    labels = [f"Q{i + 1} [{edges[i]:.2f}, {edges[i + 1]:.2f}]" for i in range(len(edges) - 1)]
    return pd.Categorical.from_codes(np.asarray(codes).astype(int), labels, ordered=True)


def slice_table(df, by, pred_col):
    rows = []
    for key, g in df.groupby(by, observed=True):
        res = g[pred_col] - g["target"]
        rows.append({
            "slice": str(key),
            "n": len(g),
            "rmse": rmse(g["target"], g[pred_col]),
            "mae": float(res.abs().mean()),
            "bias": float(res.mean()),
        })
    return pd.DataFrame(rows)


def plot_pred_vs_true(df, pred_col, path):
    lim = [min(df["target"].min(), df[pred_col].min()) - 0.2, max(df["target"].max(), df[pred_col].max()) + 0.2]
    plt.figure(figsize=(5.5, 5))
    plt.scatter(df["target"], df[pred_col], s=5, alpha=0.4)
    plt.plot(lim, lim, color="black", lw=1)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("target")
    plt.ylabel("OOF prediction")
    plt.title("Prediction vs target")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_residuals(df, pred_col, bins, path):
    res = df[pred_col] - df["target"]
    binned = df.assign(res=res, bin=quantile_bins(df["target"], bins)).groupby("bin", observed=True).agg(
        x=("target", "mean"), y=("res", "mean"))
    plt.figure(figsize=(6, 4))
    plt.scatter(df["target"], res, s=5, alpha=0.3)
    plt.plot(binned["x"], binned["y"], color="red", marker="o", label="mean residual per bin")
    plt.axhline(0, color="black", lw=1)
    plt.xlabel("target")
    plt.ylabel("prediction - target")
    plt.legend()
    plt.title("Residuals vs target")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_slice_rmse(tables, path):
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4))
    for ax, (name, t) in zip(np.atleast_1d(axes), tables.items()):
        ax.bar(t["slice"], t["rmse"])
        ax.set_title(f"RMSE by {name}")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_fold_rmse(fold_df, path):
    plt.figure(figsize=(6, 4))
    for seed, g in fold_df.groupby("seed"):
        plt.scatter(g["fold"], g["rmse"], label=f"seed {seed}")
    plt.axhline(fold_df["rmse"].mean(), color="black", lw=1, ls="--", label="mean")
    plt.xlabel("fold")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("RMSE per fold and seed")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def error_examples(df, train, pred_col, n):
    d = df.assign(excerpt=train["excerpt"].values, residual=df[pred_col] - df["target"])
    cols = ["id", "target", pred_col, "residual", "standard_error", "has_url", "excerpt"]
    over = d.nlargest(n, "residual")[cols].assign(kind="predicted too easy")
    under = d.nsmallest(n, "residual")[cols].assign(kind="predicted too hard")
    return pd.concat([over, under], ignore_index=True)


def top_terms(model, n):
    names = np.asarray(model[:-1].get_feature_names_out())
    coef = np.asarray(model[-1].coef_)
    is_term = np.array([not x.startswith("feats__") for x in names])
    terms = np.array([x.split("__", 1)[1] if x.startswith("tfidf__") else x for x in names])
    feats = pd.DataFrame({"feature": [x.split("__", 1)[1] for x in names[~is_term]], "coef": coef[~is_term]})
    feats = feats.reindex(feats["coef"].abs().sort_values(ascending=False).index)
    if not is_term.any():
        return None, None, feats
    t_names, t_coef = terms[is_term], coef[is_term]
    order = np.argsort(t_coef)
    hard = pd.DataFrame({"term": t_names[order[:n]], "coef": t_coef[order[:n]]})
    easy = pd.DataFrame({"term": t_names[order[::-1][:n]], "coef": t_coef[order[::-1][:n]]})
    return easy, hard, feats


def main(config_path):
    cfg = load_config(config_path)
    ev = cfg["eval"]
    out = experiment_dir(cfg)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    train = load_train(cfg)
    oof = pd.read_csv(out / "oof.csv")
    fold_df = pd.read_csv(out / "fold_metrics.csv")
    with open(out / "metrics.json") as f:
        metrics = json.load(f)
    assert (oof["id"].values == train["id"].values).all()

    seeds = cfg["cv"]["seeds"]
    pred_col = f"pred_{seeds[0]}"
    y, p = oof["target"].values, oof[pred_col].values

    ci_rows = []
    for seed in seeds:
        lo, hi = bootstrap_rmse(y, oof[f"pred_{seed}"].values, ev["bootstrap_iters"], ev["ci"], ev["seed"])
        ci_rows.append({"seed": seed, "oof_rmse": rmse(y, oof[f"pred_{seed}"]), "ci_low": lo, "ci_high": hi})
    ci_df = pd.DataFrame(ci_rows)

    oof["target_bin"] = quantile_bins(oof["target"], ev["target_bins"])
    oof["se_bin"] = quantile_bins(oof["standard_error"], ev["se_bins"])
    slices = {
        "has_url": slice_table(oof, "has_url", pred_col),
        "target quantile": slice_table(oof, "target_bin", pred_col),
        "standard_error quantile": slice_table(oof, "se_bin", pred_col),
    }

    pred_std_ratio = float(np.std(p) / np.std(y))
    slope = float(np.polyfit(y, p - y, 1)[0])

    plot_pred_vs_true(oof, pred_col, fig_dir / "pred_vs_true.png")
    plot_residuals(oof, pred_col, ev["target_bins"], fig_dir / "residuals_vs_target.png")
    plot_slice_rmse(slices, fig_dir / "rmse_by_slice.png")
    plot_fold_rmse(fold_df, fig_dir / "rmse_per_fold.png")

    errors = error_examples(oof, train, pred_col, ev["n_error_examples"])
    errors.to_csv(out / "error_examples.csv", index=False)

    model = joblib.load(resolve(cfg["paths"]["models"]) / f"{cfg['experiment']}.joblib")
    easy, hard, feat_coefs = top_terms(model, ev["n_top_terms"])

    pf = metrics["per_fold"]
    summary_df = pd.DataFrame([
        {"metric": m, "mean": pf[m]["mean"], "std": pf[m]["std"]}
        for m in ["rmse", "mae", "r2", "pearson", "spearman", "dummy_rmse", "fit_time_s"]
    ])
    n_folds = len(fold_df)

    lines = [
        f"# Results: {cfg['experiment']}",
        "",
        f"Model: `{cfg['model']['name']}`, params: `{json.dumps(cfg['model'].get('params', cfg['model']))}`",
        f"CV: {cfg['cv']['n_splits']}-fold stratified by {cfg['cv']['n_bins']} target quantiles, seeds {seeds} "
        f"({n_folds} folds total).",
        "",
        "## Per-fold metrics (mean ± std over all folds and seeds)",
        "",
        md_table(summary_df),
        "",
        "`dummy_rmse` – sanity check: constant prediction equal to the train-fold mean.",
        "",
        f"## OOF RMSE per seed with {int(ev['ci'] * 100)}% bootstrap CI ({ev['bootstrap_iters']} resamples)",
        "",
        md_table(ci_df),
        "",
        f"OOF RMSE across seeds: {metrics['oof_rmse_across_seeds']['mean']:.4f} ± "
        f"{metrics['oof_rmse_across_seeds']['std']:.4f}",
        "",
        f"## Error analysis by slices (OOF, seed {seeds[0]})",
        "",
        "`bias` = mean(prediction - target); positive means predicted easier than labeled.",
        "",
    ]
    for name, t in slices.items():
        lines += [f"### {name}", "", md_table(t), ""]
    lines += [
        "## Shrinkage toward the mean",
        "",
        f"std(prediction) / std(target) = {pred_std_ratio:.3f}; "
        f"slope of residual vs target = {slope:.3f} (0 means no systematic shrinkage).",
        "",
        f"## Largest errors (OOF, seed {seeds[0]})",
        "",
    ]
    for r in errors.itertuples(index=False):
        text = r.excerpt.replace("\n", " ")[:300]
        lines += [
            f"- **{r.kind}** | id `{r.id}` | target {r.target:.3f} | pred {getattr(r, pred_col):.3f} | "
            f"residual {r.residual:+.3f} | SE {r.standard_error:.3f} | has_url {r.has_url}",
            f"  > {text}...",
            "",
        ]
    if easy is not None:
        lines += [
            "## Top terms of the full-data model",
            "",
            "### Easier (positive coefficients)",
            "",
            md_table(easy),
            "",
            "### Harder (negative coefficients)",
            "",
            md_table(hard),
            "",
        ]
    if not feat_coefs.empty:
        lines += [
            "## Readability feature coefficients of the full-data model (standardized features)",
            "",
            md_table(feat_coefs),
            "",
        ]
    lines += [
        "## Figures",
        "",
        "![pred vs true](figures/pred_vs_true.png)",
        "![residuals](figures/residuals_vs_target.png)",
        "![rmse by slice](figures/rmse_by_slice.png)",
        "![rmse per fold](figures/rmse_per_fold.png)",
    ]
    (out / "results.md").write_text("\n".join(lines))

    metrics["bootstrap_ci"] = ci_df.to_dict(orient="records")
    metrics["slices"] = {k: v.to_dict(orient="records") for k, v in slices.items()}
    metrics["pred_std_ratio"] = pred_std_ratio
    metrics["residual_slope"] = slope
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {out / 'results.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)
