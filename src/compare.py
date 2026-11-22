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

from src.registry import comparison_pairs, describe, load_experiments, validate
from src.utils import ROOT, compute_metrics, resolve, rmse

KIND_LABEL = {"improvement": "улучшение", "total": "итог", "ablation": "абляция"}


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


def verdict(rows):
    if all(r["ci_high"] < 0 for r in rows):
        return "значимо лучше"
    if all(r["ci_low"] > 0 for r in rows):
        return "значимо хуже"
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


def alpha_files(reports, name):
    d = reports / name
    alphas = pd.read_csv(d / "alphas.csv") if (d / "alphas.csv").exists() else None
    curve = pd.read_csv(d / "alpha_curve.csv") if (d / "alpha_curve.csv").exists() else None
    return alphas, curve


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
        alphas, _ = alpha_files(reports, name)
        if alphas is not None:
            grid = np.logspace(**{"start": first["model_defaults"]["alpha_grid"]["log10_start"],
                                  "stop": first["model_defaults"]["alpha_grid"]["log10_stop"],
                                  "num": first["model_defaults"]["alpha_grid"]["num"]})
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
        v = verdict(seed_rows)
        for row in seed_rows:
            pair_rows.append({**pair, **row, "verdict": v, "folds_better": better_folds, "n_folds": len(merged)})
    pairs_df = pd.DataFrame(pair_rows)
    return {"exps": exps, "seeds": seeds, "comp": comp, "summaries": summaries, "fold_tables": fold_tables,
            "oofs": oofs, "pairs": pairs_df, "reports": reports, "hash_problems": hash_problems}


def fmt(x, d=4):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


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


def write_markdown(res, exp_df, comp_df, out):
    comp = res["comp"]
    view = pd.DataFrame({
        "ID": exp_df["id"], "Роль": exp_df["role"], "Модель": exp_df["model"], "Сиды": exp_df["seeds"],
        "RMSE (фолды)": [f"{a:.4f} ± {b:.4f}" for a, b in zip(exp_df["fold_rmse_mean"], exp_df["fold_rmse_std"])],
        "OOF RMSE": [f"{a:.4f} ± {b:.4f}" for a, b in zip(exp_df["oof_rmse"], exp_df["oof_rmse_seed_std"])],
        "R²": exp_df["r2"].map(lambda x: fmt(x, 3)), "Spearman": exp_df["spearman"].map(lambda x: fmt(x, 3)),
        "std ratio": exp_df["std_ratio"].map(lambda x: fmt(x, 3)),
        "наклон калибр.": exp_df["calib_slope"].map(lambda x: fmt(x, 3)),
        "RMSE Q1": exp_df["rmse_q1"].map(lambda x: fmt(x, 3)),
        "RMSE Q5": exp_df["rmse_q_last"].map(lambda x: fmt(x, 3)),
        "RMSE URL": exp_df["rmse_url"].map(lambda x: fmt(x, 3)),
        "alpha (медиана)": exp_df["alpha_median"].map(lambda x: "–" if pd.isna(x) else f"{x:.3g}"),
    })
    cview = pd.DataFrame({
        "Тип": comp_df["kind"].map(KIND_LABEL), "Модель": comp_df["model"], "Референс": comp_df["ref"],
        "Δ RMSE (среднее)": comp_df["delta_mean"].map(lambda x: f"{x:+.4f}"),
        "Δ по сидам": comp_df["delta_per_seed"], "95% ДИ по сидам": comp_df["ci_per_seed"],
        "Фолдов лучше": comp_df["folds_better"], "Вердикт": comp_df["verdict"],
    })
    b = comp["bootstrap"]
    lines = [
        "# Таблица экспериментов",
        "",
        "Все эксперименты: одни и те же фолды (`data/folds.csv`, 5-fold x 3 сида), одна метрика, "
        "одинаковый код расчета метрик (метрики пересчитаны в `src/compare.py` из OOF-предсказаний).",
        "",
        "## Эксперименты",
        "",
        md_table(view),
        "",
        "- `RMSE (фолды)` – mean ± std по 15 фолдам; `OOF RMSE` – RMSE по всем OOF-предсказаниям, mean ± std по сидам.",
        "- `std ratio` = std(pred) / std(target); `наклон калибр.` – наклон регрессии target на pred "
        "(1 – откалиброванные предсказания, > 1 – сжатие к среднему).",
        "- `RMSE Q1` / `RMSE Q5` – самые сложные / самые легкие 20% текстов; `RMSE URL` – срез современных текстов.",
        "",
        "## Сравнения",
        "",
        f"Δ = RMSE(модель) − RMSE(референс), отрицательная Δ – модель лучше. Для абляций референс – итоговая модель, "
        f"поэтому положительная Δ означает, что удаленный компонент полезен. ДИ – парный bootstrap по OOF "
        f"({b['iters']} ресемплов, {int(b['ci'] * 100)}%). Правило вердикта (зафиксировано до запуска): эффект значим, "
        f"только если ДИ не включает 0 для всех сидов.",
        "",
        md_table(cview),
        "",
        "## Гипотезы и ожидания",
        "",
    ]
    for r in exp_df.itertuples(index=False):
        lines += [f"**{r.id}** – {r.title}", "", f"- Гипотеза: {r.hypothesis}", f"- Ожидание: {r.expected}", ""]
    if res["hash_problems"]:
        lines += ["## Предупреждения", ""] + [f"- {p}" for p in res["hash_problems"]] + [""]
    lines += ["## Графики", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted((out / "figures").glob("*.png"))]
    (out / "experiments.md").write_text("\n".join(lines))


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
    tuned = [n for n in res["exps"] if alpha_files(res["reports"], n)[1] is not None]
    if not tuned:
        return False
    show = [n for n in main_path(res) if n in tuned] or tuned[:2]
    fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 4), squeeze=False)
    for ax, n in zip(axes[0], show):
        alphas, curve = alpha_files(res["reports"], n)
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


def write_outputs(res, out):
    out.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    for old in fig.glob("*.png"):
        old.unlink()
    exp_df, comp_df = build_tables(res)
    exp_df.to_csv(out / "experiments.csv", index=False)
    comp_df.to_csv(out / "comparisons.csv", index=False)
    res["pairs"].to_csv(out / "paired_bootstrap.csv", index=False)
    plot_rmse_by_experiment(res, fig / "01_rmse_by_experiment.png")
    plot_forest(res, fig / "02_delta_forest.png")
    plot_quintiles(res, fig / "03_quintiles_main_path.png")
    plot_residuals(res, fig / "04_residuals_baseline_vs_final.png")
    plot_fold_pairs(res, fig / "05_fold_rmse_baseline_vs_final.png")
    plot_alpha(res, fig / "06_alpha_selection.png")
    write_markdown(res, exp_df, comp_df, out)
    return exp_df, comp_df


def run(exps_dir=None, require_hash=True):
    exps = load_experiments() if exps_dir is None else load_experiments(exps_dir)
    validate(exps)
    first = next(iter(exps.values()))
    res = compute(exps, resolve(first["paths"]["reports"]), resolve(first["paths"]["folds"]), require_hash)
    out = resolve(first["comparison"]["output"])
    exp_df, comp_df = write_outputs(res, out)
    print(comp_df[["kind", "model", "ref", "delta_mean", "folds_better", "verdict"]].to_string(index=False))
    print(f"Saved {out / 'experiments.md'}")


def selftest():
    from src.registry import load_experiments as load
    from src.tracking import file_sha256

    exps = load()
    first = next(iter(exps.values()))
    seeds = first["cv"]["seeds"]
    folds_path = resolve(first["paths"]["folds"])
    base_oof = load_oof(resolve(first["paths"]["reports"]) / "tfidf_ridge" / "oof.csv", seeds)
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
            (d / "metrics.json").write_text(json.dumps({"run_info": {"folds_sha256": folds_hash}}))
            if exps[name]["model"]["tuning"] == "nested_loo":
                grid = np.logspace(-2, 3, 31)
                curve = pd.DataFrame([{"seed": s, "fold": k, "alpha": a, "inner_rmse": 0.7 + 0.01 * (np.log10(a) + 0.5) ** 2}
                                      for s in seeds for k in range(5) for a in grid])
                curve.to_csv(d / "alpha_curve.csv", index=False)
                pd.DataFrame([{"seed": s, "fold": k, "alpha": grid[9]} for s in seeds for k in range(5)]).to_csv(d / "alphas.csv", index=False)
        res = compute(exps, tmp, folds_path)
        out = tmp / "comparison"
        exp_df, comp_df = write_outputs(res, out)
        assert len(exp_df) == len(exps)
        assert set(comp_df["kind"]) == {"improvement", "total", "ablation"}
        total = comp_df[comp_df["kind"] == "total"].iloc[0]
        assert total["verdict"] == "значимо лучше", total
        figs = sorted(p.name for p in (out / "figures").glob("*.png"))
        assert len(figs) == 6, figs
        bad = dict(json.loads((tmp / "e1_ridgecv" / "metrics.json").read_text()))
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
        print(f"compare selftest OK: {len(exp_df)} experiments, {len(comp_df)} comparisons, figures {figs}")
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
