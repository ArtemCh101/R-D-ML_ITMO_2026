import argparse
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.registry import alpha_grid, comparison_pairs, describe, feature_columns, load_experiments, validate
from src.utils import ROOT, compute_metrics, resolve, rmse

def load_oof(path, seeds):
    oof = pd.read_csv(path)
    missing = [f"pred_{s}" for s in seeds if f"pred_{s}" not in oof.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return oof


def check_alignment(oofs, folds):
    names = list(oofs)
    ref = oofs[names[0]]
    for name in names[1:]:
        o = oofs[name]
        if not (o["id"].values == ref["id"].values).all():
            raise ValueError(f"{name}: OOF ids differ from {names[0]}")
        if not np.allclose(o["target"].values, ref["target"].values):
            raise ValueError(f"{name}: OOF targets differ from {names[0]}")
    if not (folds["id"].values == ref["id"].values).all():
        raise ValueError("folds.csv ids differ from OOF ids")


def check_folds_hash(metrics, current_hash):
    problems = []
    for name, m in metrics.items():
        h = (m or {}).get("run_info", {}).get("folds_sha256")
        if h is None:
            problems.append(f"{name}: no folds_sha256 recorded")
        elif h != current_hash:
            problems.append(f"{name}: trained on different folds ({h[:12]} != {current_hash[:12]})")
    return problems


def fold_rmse(oof, folds, seeds):
    rows = []
    for seed in seeds:
        fold_col = folds[f"fold_{seed}"].values
        for k in sorted(np.unique(fold_col)):
            mask = fold_col == k
            rows.append({"seed": seed, "fold": int(k),
                         "rmse": rmse(oof["target"].values[mask], oof[f"pred_{seed}"].values[mask])})
    return pd.DataFrame(rows)


def paired_bootstrap(y, pred_model, pred_ref, iters, ci, seed, chunk=500):
    y = np.asarray(y)
    sq_m = (y - np.asarray(pred_model)) ** 2
    sq_r = (y - np.asarray(pred_ref)) ** 2
    n = len(y)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iters)
    for start in range(0, iters, chunk):
        size = min(chunk, iters - start)
        idx = rng.integers(0, n, size=(size, n))
        deltas[start:start + size] = np.sqrt(sq_m[idx].mean(axis=1)) - np.sqrt(sq_r[idx].mean(axis=1))
    alpha = (1 - ci) / 2
    return {
        "delta": float(np.sqrt(sq_m.mean()) - np.sqrt(sq_r.mean())),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1 - alpha)),
        "p_model_better": float(np.mean(deltas < 0)),
    }


def verdict(rows, min_effect):
    mean_delta = float(np.mean([r["delta"] for r in rows]))
    if all(r["ci_high"] < 0 for r in rows):
        return "значимо лучше" if abs(mean_delta) >= min_effect else "значимо, но пренебрежимо"
    if all(r["ci_low"] > 0 for r in rows):
        return "значимо хуже" if abs(mean_delta) >= min_effect else "значимо, но пренебрежимо"
    return "незначимо"


def model_summary(oof, folds, seeds, target_bins):
    y = oof["target"].values
    q = pd.qcut(y, target_bins, labels=False)
    url = oof["has_url"].astype(bool).values if "has_url" in oof else None
    per_seed = []
    for seed in seeds:
        p = oof[f"pred_{seed}"].values
        m = compute_metrics(y, p)
        m["std_ratio"] = float(np.std(p) / np.std(y))
        m["calib_slope"] = float(np.cov(y, p)[0, 1] / np.var(p, ddof=1))
        m["rmse_q1"] = rmse(y[q == 0], p[q == 0])
        m["rmse_q_last"] = rmse(y[q == target_bins - 1], p[q == target_bins - 1])
        m["bias_q1"] = float(np.mean(p[q == 0] - y[q == 0]))
        m["bias_q_last"] = float(np.mean(p[q == target_bins - 1] - y[q == target_bins - 1]))
        if url is not None:
            m["rmse_url"] = rmse(y[url], p[url])
            m["rmse_no_url"] = rmse(y[~url], p[~url])
        per_seed.append(m)
    seed_df = pd.DataFrame(per_seed)
    fr = fold_rmse(oof, folds, seeds)
    out = {f"{k}": float(seed_df[k].mean()) for k in seed_df.columns}
    out["oof_rmse_seed_std"] = float(seed_df["rmse"].std(ddof=1)) if len(seeds) > 1 else 0.0
    out["fold_rmse_mean"] = float(fr["rmse"].mean())
    out["fold_rmse_std"] = float(fr["rmse"].std(ddof=1))
    return out, fr


def stored_table(metrics_entry, key):
    rows = (metrics_entry or {}).get(key)
    return pd.DataFrame(rows) if rows else None


def alpha_files(res, name):
    m = res["metrics"].get(name)
    return stored_table(m, "alphas"), stored_table(m, "alpha_curve")


def compute(exps, reports, folds_path, require_hash=True):
    first = next(iter(exps.values()))
    seeds = first["cv"]["seeds"]
    comp = first["comparison"]
    boot = comp["bootstrap"]
    target_bins = first["eval"]["target_bins"]
    folds = pd.read_csv(folds_path)
    oofs, metrics = {}, {}
    for name in exps:
        d = reports / name
        if not (d / "oof.csv").exists():
            raise FileNotFoundError(f"{d / 'oof.csv'} not found – run training for {name} first")
        oofs[name] = load_oof(d / "oof.csv", seeds)
        metrics[name] = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else None
    check_alignment(oofs, folds)

    from src.tracking import file_sha256
    hash_problems = check_folds_hash(metrics, file_sha256(folds_path))
    if hash_problems:
        msg = "folds consistency check failed:\n  " + "\n  ".join(hash_problems)
        if require_hash:
            raise ValueError(msg)
        print(f"[warn] {msg}")

    summaries, fold_tables = {}, {}
    for name in exps:
        summaries[name], fold_tables[name] = model_summary(oofs[name], folds, seeds, target_bins)
        alphas = stored_table(metrics[name], "alphas")
        if alphas is not None:
            grid = alpha_grid(first)
            a = alphas["alpha"].values
            summaries[name]["alpha_median"] = float(np.median(a))
            summaries[name]["alpha_min"] = float(a.min())
            summaries[name]["alpha_max"] = float(a.max())
            summaries[name]["alpha_edge_share"] = float(np.mean(np.isclose(a, grid[0]) | np.isclose(a, grid[-1])))

    pair_rows = []
    for pair in comparison_pairs(exps):
        m, r = pair["model"], pair["ref"]
        y = oofs[m]["target"].values
        seed_rows = []
        for seed in seeds:
            res = paired_bootstrap(y, oofs[m][f"pred_{seed}"].values, oofs[r][f"pred_{seed}"].values,
                                   boot["iters"], boot["ci"], boot["seed"] + seed)
            seed_rows.append({"seed": seed, **res})
        fm, fr = fold_tables[m], fold_tables[r]
        merged = fm.merge(fr, on=["seed", "fold"], suffixes=("_m", "_r"))
        better_folds = int((merged["rmse_m"] < merged["rmse_r"]).sum())
        v = verdict(seed_rows, comp["min_effect"])
        for row in seed_rows:
            pair_rows.append({**pair, **row, "verdict": v, "folds_better": better_folds, "n_folds": len(merged)})
    pairs_df = pd.DataFrame(pair_rows)

    reproduction = None
    hw1 = resolve(comp["hw1_baseline_oof"])
    if hw1.exists():
        base = pd.read_csv(hw1)
        e0 = oofs[comp["reference"]]
        if (base["id"].values == e0["id"].values).all():
            y = e0["target"].values
            reproduction = {
                "max_abs_pred": max(float(np.max(np.abs(base[f"pred_{s}"].values - e0[f"pred_{s}"].values))) for s in seeds),
                "max_abs_rmse": max(abs(rmse(y, base[f"pred_{s}"]) - rmse(y, e0[f"pred_{s}"])) for s in seeds),
            }
            reproduction["ok"] = reproduction["max_abs_rmse"] < comp["reproduction_rmse_tol"]

    boundary = []
    share_limit = first["model_defaults"]["alpha_boundary_max_share"]
    for name, m in summaries.items():
        if m.get("alpha_edge_share", 0) > share_limit:
            boundary.append(f"{name}: выбранная alpha на краю сетки в {m['alpha_edge_share']:.0%} фолдов "
                            f"(порог {share_limit:.0%}) – по правилу сетку нужно расширить на 2 декады")

    return {"exps": exps, "seeds": seeds, "comp": comp, "summaries": summaries, "fold_tables": fold_tables,
            "oofs": oofs, "pairs": pairs_df, "reports": reports, "hash_problems": hash_problems,
            "reproduction": reproduction, "boundary": boundary, "metrics": metrics}


def build_tables(res):
    exps, s = res["exps"], res["summaries"]
    rows = []
    for name, cfg in exps.items():
        role = "baseline" if "improvement" not in cfg and "ablation" not in cfg else (
            "улучшение" if "improvement" in cfg else "абляция")
        if "improvement" in cfg and "ablation" in cfg:
            role = "улучшение / абляция"
        m = s[name]
        rows.append({
            "id": name, "role": role, "title": cfg.get("title", ""), "model": describe(cfg),
            "seeds": ",".join(map(str, res["seeds"])),
            "fold_rmse_mean": m["fold_rmse_mean"], "fold_rmse_std": m["fold_rmse_std"],
            "oof_rmse": m["rmse"], "oof_rmse_seed_std": m["oof_rmse_seed_std"],
            "mae": m["mae"], "r2": m["r2"], "spearman": m["spearman"],
            "std_ratio": m["std_ratio"], "calib_slope": m["calib_slope"],
            "rmse_q1": m["rmse_q1"], "rmse_q_last": m["rmse_q_last"],
            "rmse_url": m.get("rmse_url"), "alpha_median": m.get("alpha_median"),
            "alpha_edge_share": m.get("alpha_edge_share"),
            "hypothesis": " ".join(str(cfg.get("hypothesis", "")).split()),
            "expected": " ".join(str(cfg.get("expected", cfg.get("ablation", {}).get("expected", ""))).split()),
        })
    exp_df = pd.DataFrame(rows)

    pairs = res["pairs"]
    comp_rows = []
    for (m, r, kind), g in pairs.groupby(["model", "ref", "kind"], sort=False):
        comp_rows.append({
            "kind": kind, "model": m, "ref": r,
            "delta_mean": g["delta"].mean(),
            "delta_per_seed": ", ".join(f"{d:+.4f}" for d in g["delta"]),
            "ci_per_seed": ", ".join(f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(g["ci_low"], g["ci_high"])),
            "folds_better": f"{g['folds_better'].iloc[0]}/{g['n_folds'].iloc[0]}",
            "verdict": g["verdict"].iloc[0],
        })
    comp_df = pd.DataFrame(comp_rows)
    return exp_df, comp_df


def build_checks(res):
    rep = res["reproduction"]
    return {
        "hw1_reproduction": rep,
        "folds_hash_consistent": not res["hash_problems"],
        "folds_hash_problems": res["hash_problems"],
        "alpha_boundary_rule_triggered": res["boundary"],
        "decision_rule": res["comp"]["decision_rule"],
        "min_effect": res["comp"]["min_effect"],
        "bootstrap": res["comp"]["bootstrap"],
    }


def error_examples(res, n=3):
    comp = res["comp"]
    first = res["exps"][comp["final"]]
    seed = res["seeds"][0]
    train = pd.read_csv(resolve(first["paths"]["train"]))[["id", "excerpt"]]
    final = res["oofs"][comp["final"]]
    base = res["oofs"][comp["reference"]]
    d = final[["id", "target", "standard_error", "has_url"]].copy()
    d["pred_final"] = final[f"pred_{seed}"]
    d["pred_baseline"] = base[f"pred_{seed}"]
    d["residual_final"] = d["pred_final"] - d["target"]
    d = d.merge(train, on="id", how="left")
    d["excerpt"] = d["excerpt"].str.replace("\n", " ").str.slice(0, 300)
    over = d.nlargest(n, "residual_final").assign(kind="predicted too easy")
    under = d.nsmallest(n, "residual_final").assign(kind="predicted too hard")
    out = pd.concat([over, under], ignore_index=True)
    out.insert(0, "seed", seed)
    return out


def plot_rmse_by_experiment(res, path):
    s = res["summaries"]
    names = list(res["exps"])[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.8))
    means = [s[n]["fold_rmse_mean"] for n in names]
    stds = [s[n]["fold_rmse_std"] for n in names]
    ax.errorbar(means, range(len(names)), xerr=stds, fmt="s", color="tab:blue", capsize=3, ms=5,
                label="mean ± std по 15 фолдам")
    for i, n in enumerate(names):
        oof = res["oofs"][n]
        vals = [rmse(oof["target"], oof[f"pred_{seed}"]) for seed in res["seeds"]]
        ax.scatter(vals, [i + 0.25] * len(vals), color="black", s=10, zorder=3,
                   label="OOF RMSE по сидам" if i == 0 else None)
    ax.axvline(s[res["comp"]["reference"]]["fold_rmse_mean"], color="red", ls="--", lw=1, label="baseline")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("RMSE (меньше – лучше)")
    ax.set_title("RMSE по экспериментам")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_forest(res, path):
    pairs = res["pairs"]
    groups = [("Улучшения (референс – предыдущий шаг / baseline)", pairs[pairs["kind"] != "ablation"]),
              ("Абляции (референс – итоговая модель)", pairs[pairs["kind"] == "ablation"])]
    heights = [max(len(g[["model", "ref"]].drop_duplicates()), 1) for _, g in groups]
    fig, axes = plt.subplots(2, 1, figsize=(9, 0.55 * sum(heights) + 3), gridspec_kw={"height_ratios": heights})
    colors = plt.cm.tab10.colors
    for ax, (title, g) in zip(axes, groups):
        labels = g[["model", "ref"]].drop_duplicates().apply(lambda r: f"{r.model} vs {r.ref}", axis=1).tolist()
        for i, label in enumerate(labels):
            sub = g[g.apply(lambda r: f"{r.model} vs {r.ref}", axis=1) == label]
            for j, row in enumerate(sub.itertuples(index=False)):
                y = i + (j - (len(sub) - 1) / 2) * 0.18
                ax.errorbar(row.delta, y, xerr=[[row.delta - row.ci_low], [row.ci_high - row.delta]], fmt="o",
                            color=colors[j], capsize=3, ms=4, label=f"seed {row.seed}" if i == 0 else None)
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Δ RMSE = RMSE(модель) − RMSE(референс), 95% ДИ")
        if labels:
            ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main_path(res):
    comp = res["comp"]
    path = [comp["final"]]
    while "improvement" in res["exps"][path[-1]]:
        path.append(res["exps"][path[-1]]["improvement"]["parent"])
    return path[::-1]


def plot_quintiles(res, path):
    names = main_path(res)
    seed = res["seeds"][0]
    bins = res["exps"][names[0]]["eval"]["target_bins"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for n in names:
        oof = res["oofs"][n]
        y, p = oof["target"].values, oof[f"pred_{seed}"].values
        q = pd.qcut(y, bins, labels=False)
        r = [rmse(y[q == k], p[q == k]) for k in range(bins)]
        b = [float(np.mean(p[q == k] - y[q == k])) for k in range(bins)]
        x = [f"Q{k + 1}" for k in range(bins)]
        axes[0].plot(x, r, marker="o", label=n)
        axes[1].plot(x, b, marker="o", label=n)
    axes[0].set_title(f"RMSE по квинтилям таргета (seed {seed})")
    axes[1].set_title(f"bias = mean(pred − target) по квинтилям (seed {seed})")
    axes[1].axhline(0, color="black", lw=1)
    for ax in axes:
        ax.set_xlabel("квинтиль таргета (Q1 – самые сложные)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_residuals(res, path):
    comp = res["comp"]
    seed = res["seeds"][0]
    bins = res["exps"][comp["reference"]]["eval"]["target_bins"] * 2
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, color in [(comp["reference"], "tab:gray"), (comp["final"], "tab:blue")]:
        oof = res["oofs"][n]
        y, p = oof["target"].values, oof[f"pred_{seed}"].values
        ax.scatter(y, p - y, s=3, alpha=0.15, color=color)
        q = pd.qcut(y, bins, labels=False)
        ax.plot([y[q == k].mean() for k in range(bins)], [np.mean(p[q == k] - y[q == k]) for k in range(bins)],
                marker="o", color=color, label=f"{n}: std ratio {np.std(p) / np.std(y):.2f}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("target")
    ax.set_ylabel("pred − target")
    ax.set_title(f"Остатки: baseline против итоговой модели (seed {seed})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_fold_pairs(res, path):
    comp = res["comp"]
    a = res["fold_tables"][comp["reference"]]
    b = res["fold_tables"][comp["final"]]
    m = a.merge(b, on=["seed", "fold"], suffixes=("_ref", "_final"))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for row in m.itertuples(index=False):
        ax.plot([0, 1], [row.rmse_ref, row.rmse_final], color="tab:green" if row.rmse_final < row.rmse_ref else "tab:red",
                alpha=0.7, marker="o", ms=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([comp["reference"], comp["final"]])
    ax.set_ylabel("RMSE на фолде")
    better = int((m["rmse_final"] < m["rmse_ref"]).sum())
    ax.set_title(f"RMSE по фолдам (лучше на {better}/{len(m)})")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_alpha(res, path):
    tuned = [n for n in res["exps"] if alpha_files(res, n)[1] is not None]
    if not tuned:
        return False
    show = [n for n in main_path(res) if n in tuned] or tuned[:2]
    fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 4), squeeze=False)
    for ax, n in zip(axes[0], show):
        alphas, curve = alpha_files(res, n)
        for (seed, fold), g in curve.groupby(["seed", "fold"]):
            ax.plot(g["alpha"], g["inner_rmse"], color="tab:blue", alpha=0.3, lw=1)
        if alphas is not None:
            sel = curve.merge(alphas, on=["seed", "fold", "alpha"])
            ax.scatter(sel["alpha"], sel["inner_rmse"], color="red", s=15, zorder=3, label="выбранная alpha")
        fixed = res["exps"][n]["model_defaults"]["fixed_alpha"]
        ax.axvline(fixed, color="gray", ls="--", lw=1, label=f"baseline alpha = {fixed}")
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_ylabel("внутренняя LOO RMSE (train-часть фолда)")
        ax.set_title(n)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return True


def plot_feature_coefs(res, path):
    name = res["comp"]["final"]
    coefs = stored_table(res["metrics"].get(name), "feature_coefs")
    if coefs is None:
        return False
    cfg = res["exps"][name]
    group_of = {c: g for g, cols in cfg["features"]["groups"].items() for c in cols}
    order = feature_columns(cfg)
    stats = coefs.groupby("feature")["coef"].agg(["mean", "std"]).reindex(order)
    groups = list(cfg["features"]["groups"])
    colors = {g: plt.cm.tab10.colors[i] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(order) + 1.5))
    y = np.arange(len(order))[::-1]
    ax.barh(y, stats["mean"], xerr=stats["std"], color=[colors[group_of[c]] for c in order], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("коэффициент Ridge (стандартизированный признак), mean ± std по 15 фолдам")
    ax.set_title(f"Readability-признаки в итоговой модели ({name})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in groups]
    ax.legend(handles, groups, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return True


def write_outputs(res, out):
    fig = out / "figures"
    tables = out / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for old in fig.glob("0*.png"):
        old.unlink()
    for old in tables.iterdir():
        if old.is_file():
            old.unlink()
    exp_df, comp_df = build_tables(res)
    exp_df.to_csv(tables / "experiments.csv", index=False)
    comp_df.to_csv(tables / "comparisons.csv", index=False)
    res["pairs"].to_csv(tables / "paired_bootstrap.csv", index=False)
    error_examples(res).to_csv(tables / "error_examples_final.csv", index=False)
    (tables / "checks.json").write_text(json.dumps(build_checks(res), indent=2, ensure_ascii=False))
    plot_rmse_by_experiment(res, fig / "01_rmse_by_experiment.png")
    plot_forest(res, fig / "02_delta_forest.png")
    plot_quintiles(res, fig / "03_quintiles_main_path.png")
    plot_residuals(res, fig / "04_residuals_baseline_vs_final.png")
    plot_fold_pairs(res, fig / "05_fold_rmse_baseline_vs_final.png")
    plot_alpha(res, fig / "06_alpha_selection.png")
    plot_feature_coefs(res, fig / "07_feature_coefs_final.png")
    return exp_df, comp_df


def run(exps_dir=None, require_hash=True):
    exps = load_experiments() if exps_dir is None else load_experiments(exps_dir)
    validate(exps)
    first = next(iter(exps.values()))
    res = compute(exps, resolve(first["paths"]["reports"]), resolve(first["paths"]["folds"]), require_hash)
    out = resolve(first["comparison"]["output"])
    exp_df, comp_df = write_outputs(res, out)
    print(comp_df[["kind", "model", "ref", "delta_mean", "folds_better", "verdict"]].to_string(index=False))
    rep = res["reproduction"]
    if rep is not None:
        print(f"E0 vs HW1 baseline: max |delta RMSE| = {rep['max_abs_rmse']:.1e}, max |delta pred| = {rep['max_abs_pred']:.1e}, "
              f"{'reproduced' if rep['ok'] else 'NOT REPRODUCED'}")
    print("folds hash consistent across experiments" if not res["hash_problems"] else f"[warn] {res['hash_problems']}")
    for b in res["boundary"]:
        print(f"[warn] {b}")
    if not res["boundary"]:
        print("alpha grid boundary rule: not triggered")
    print(f"Saved tables to {out / 'tables'} and figures to {out / 'figures'}")


def selftest():
    from src.registry import load_experiments as load
    from src.tracking import file_sha256

    exps = load()
    first = next(iter(exps.values()))
    seeds = first["cv"]["seeds"]
    folds_path = resolve(first["paths"]["folds"])
    base_oof = load_oof(resolve(first["comparison"]["hw1_baseline_oof"]), seeds)
    y = base_oof["target"].values

    r = paired_bootstrap(y, base_oof["pred_42"], base_oof["pred_42"], 1000, 0.95, 0)
    assert r["delta"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0, r
    r = paired_bootstrap(y, base_oof["pred_42"] + np.random.default_rng(0).normal(0, 0.3, len(y)), base_oof["pred_42"], 1000, 0.95, 0)
    assert r["ci_low"] > 0, r
    closer = base_oof["pred_42"] + 0.3 * (y - base_oof["pred_42"])
    r_better = paired_bootstrap(y, closer, base_oof["pred_42"], 1000, 0.95, 0)
    assert r_better["ci_high"] < 0, r_better
    print("bootstrap checks OK: identical predictions -> Δ = 0, CI [0, 0]; added noise -> CI > 0; "
          "predictions moved toward target -> CI < 0")

    tmp = Path(tempfile.mkdtemp(prefix="compare_selftest_"))
    try:
        rng = np.random.default_rng(1)
        folds_hash = file_sha256(folds_path)
        grid = alpha_grid(first)
        for name in exps:
            d = tmp / name
            d.mkdir()
            oof = base_oof.copy()
            if name != first["comparison"]["reference"]:
                for s in seeds:
                    oof[f"pred_{s}"] = oof[f"pred_{s}"] + rng.normal(0, 0.05, len(oof))
            if name == first["comparison"]["final"]:
                for s in seeds:
                    p = base_oof[f"pred_{s}"]
                    oof[f"pred_{s}"] = p + 0.2 * (y - p)
            oof.to_csv(d / "oof.csv", index=False)
            m = {"run_info": {"folds_sha256": folds_hash}}
            if exps[name]["model"]["tuning"] == "nested_loo":
                m["alpha_curve"] = [{"seed": s, "fold": k, "alpha": float(a), "inner_rmse": 0.7 + 0.01 * (np.log10(a) + 0.5) ** 2}
                                    for s in seeds for k in range(5) for a in grid]
                m["alphas"] = [{"seed": s, "fold": k, "alpha": float(grid[15])} for s in seeds for k in range(5)]
            cols = feature_columns(exps[name])
            if cols:
                m["feature_coefs"] = [{"seed": s, "fold": k, "feature": c, "coef": float(rng.normal())}
                                      for s in seeds for k in range(5) for c in cols]
            (d / "metrics.json").write_text(json.dumps(m))
        res = compute(exps, tmp, folds_path)
        assert res["reproduction"] is not None and res["reproduction"]["ok"], res["reproduction"]
        out = tmp / "comparison"
        exp_df, comp_df = write_outputs(res, out)
        assert len(exp_df) == len(exps)
        assert set(comp_df["kind"]) == {"improvement", "total", "ablation"}
        total = comp_df[comp_df["kind"] == "total"].iloc[0]
        assert total["verdict"] == "значимо лучше", total
        figs = sorted(p.name for p in (out / "figures").glob("*.png"))
        assert len(figs) == 7, figs
        tables = sorted(p.name for p in (out / "tables").iterdir())
        assert tables == ["checks.json", "comparisons.csv", "error_examples_final.csv", "experiments.csv", "paired_bootstrap.csv"], tables
        bad = json.loads((tmp / "e1_ridgecv" / "metrics.json").read_text())
        bad["run_info"]["folds_sha256"] = "0" * 64
        (tmp / "e1_ridgecv" / "metrics.json").write_text(json.dumps(bad))
        try:
            compute(exps, tmp, folds_path)
            raise AssertionError("folds hash mismatch not detected")
        except ValueError:
            pass
        keep = ROOT / "reports" / "_compare_selftest"
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(out, keep)
        print(f"compare selftest OK: {len(exp_df)} experiments, {len(comp_df)} comparisons, "
              f"{len(figs)} figures, tables {tables}")
        print(f"inspect fake-data output in {keep} (delete it afterwards)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--allow-missing-hash", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        run(require_hash=not args.allow_missing_hash)
