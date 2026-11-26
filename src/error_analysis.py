import argparse
import json
import re
import zlib
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_folds, load_train
from src.features import WORD_RE, attach_features
from src.registry import load_experiments, validate
from src.tracking import file_sha256
from src.utils import load_config, resolve, rmse

ANALYSIS_CONFIG = "configs/analysis/hw3.yaml"
FEATURES = ["n_chars", "n_words", "n_sentences", "words_per_sentence", "chars_per_word", "syllables_per_word",
            "long_word_share", "polysyllable_share", "type_token_ratio", "flesch_reading_ease",
            "flesch_kincaid_grade", "dale_chall"]


def domain_of(url):
    if not isinstance(url, str):
        return None
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1).lower().removeprefix("www.") if m else None


def rare_word_share(texts, max_df):
    tokens = [[w.lower() for w in WORD_RE.findall(t)] for t in texts]
    df_count = Counter()
    for toks in tokens:
        df_count.update(set(toks))
    return np.array([np.mean([df_count[w] <= max_df for w in toks]) if toks else 0.0 for toks in tokens])


def load_context(config_path=ANALYSIS_CONFIG):
    acfg = load_config(config_path)
    exps = load_experiments()
    validate(exps)
    sel_cfg, ref_cfg = exps[acfg["selected_experiment"]], exps[acfg["reference_experiment"]]
    seeds = sel_cfg["cv"]["seeds"]
    train = load_train(sel_cfg)
    folds = load_folds(sel_cfg)
    assert (folds["id"].values == train["id"].values).all()
    runs = resolve(acfg["hw2_runs"])
    oofs, run_metrics = {}, {}
    for role, name in (("sel", acfg["selected_experiment"]), ("ref", acfg["reference_experiment"])):
        oof = pd.read_csv(runs / name / "oof.csv")
        assert (oof["id"].values == train["id"].values).all(), f"{name}: OOF ids differ from train"
        oofs[role] = oof
        run_metrics[role] = json.loads((runs / name / "metrics.json").read_text())

    feats = attach_features(train, sel_cfg, FEATURES)
    df = train[["id", "excerpt", "target", "standard_error", "url_legal"]].copy()
    df["has_url"] = df["url_legal"].notna()
    df["domain"] = df["url_legal"].map(domain_of)
    for c in FEATURES:
        df[c] = feats[c].values
    df["rare_word_share"] = rare_word_share(df["excerpt"], acfg["errors"]["rare_word_max_df"])
    for role in ("sel", "ref"):
        for s in seeds:
            df[f"pred_{role}_{s}"] = oofs[role][f"pred_{s}"].values
            df[f"res_{role}_{s}"] = df[f"pred_{role}_{s}"] - df["target"]
    a = acfg["analysis_seed"]
    df["res"] = df[f"res_sel_{a}"]
    df["target_q"] = pd.qcut(df["target"], acfg["slices"]["target_bins"], labels=False) + 1
    thr = acfg["errors"]["large_error_threshold"]
    df["large"] = df["res"].abs() >= thr
    df["large_all_seeds"] = np.all([df[f"res_sel_{s}"].abs() >= thr for s in seeds], axis=0)
    return {"acfg": acfg, "exps": exps, "sel_cfg": sel_cfg, "ref_cfg": ref_cfg, "seeds": seeds, "train": train,
            "folds": folds, "df": df, "run_metrics": run_metrics}


def boot_rng(ctx, salt=0):
    b = ctx["acfg"]["bootstrap"]
    return np.random.default_rng([b["seed"], salt])


def ci_bounds(samples, ci):
    a = (1 - ci) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def bootstrap_metric(values, fn, ctx, salt=0):
    b = ctx["acfg"]["bootstrap"]
    values = np.asarray(values)
    rng = boot_rng(ctx, salt)
    idx = rng.integers(0, len(values), size=(b["iters"], len(values)))
    samples = np.array([fn(values[i]) for i in idx])
    return ci_bounds(samples, b["ci"])


def bootstrap_diff(a, b_vals, fn, ctx, salt=0):
    b = ctx["acfg"]["bootstrap"]
    a, b_vals = np.asarray(a), np.asarray(b_vals)
    rng = boot_rng(ctx, salt)
    ia = rng.integers(0, len(a), size=(b["iters"], len(a)))
    ib = rng.integers(0, len(b_vals), size=(b["iters"], len(b_vals)))
    samples = np.array([fn(a[i]) - fn(b_vals[j]) for i, j in zip(ia, ib)])
    return ci_bounds(samples, b["ci"])


def rmse_of_res(r):
    return float(np.sqrt(np.mean(np.square(r))))


def group_stats(d, seeds, role="sel"):
    out = {"n": int(len(d)), "target_mean": float(d["target"].mean())}
    out["rmse"] = float(np.mean([rmse_of_res(d[f"res_{role}_{s}"]) for s in seeds]))
    out["bias"] = float(np.mean([d[f"res_{role}_{s}"].mean() for s in seeds]))
    out["mae"] = float(np.mean([d[f"res_{role}_{s}"].abs().mean() for s in seeds]))
    return out


def error_types(ctx):
    df, seeds, acfg = ctx["df"], ctx["seeds"], ctx["acfg"]
    ecfg = acfg["errors"]
    thr = ecfg["large_error_threshold"]
    median = df["target"].median()
    hard, easy = df["target"] <= median, df["target"] > median
    rare_t = pd.qcut(df.loc[hard, "rare_word_share"].rank(method="first"), 3, labels=False)
    wps_t = pd.qcut(df.loc[easy, "words_per_sentence"].rank(method="first"), 3, labels=False)
    out_cfg = ecfg["types"]["feature_outliers"]
    outlier = (df["words_per_sentence"] > out_cfg["words_per_sentence_gt"]) | (df["flesch_reading_ease"] < out_cfg["flesch_lt"])
    same_sign = np.all([np.sign(df[f"res_sel_{s}"]) == np.sign(df[f"res_{r}_{s2}"])
                        for s in seeds for r in ("sel", "ref") for s2 in seeds], axis=0)
    noise = df["large_all_seeds"] & np.all([df[f"res_ref_{s}"].abs() >= thr for s in seeds], axis=0) & same_sign

    masks = {
        "extremes": (df["target_q"].isin([1, 5]), df["target_q"].isin([2, 3, 4])),
        "rare_terms": (hard & df.index.isin(rare_t[rare_t == 2].index), hard & df.index.isin(rare_t[rare_t == 0].index)),
        "long_sentences": (easy & df.index.isin(wps_t[wps_t == 2].index), easy & df.index.isin(wps_t[wps_t == 0].index)),
        "feature_outliers": (outlier, ~outlier),
        "label_noise": (noise, ~noise),
    }
    fns = {"rmse": rmse_of_res, "bias": lambda r: float(np.mean(r)), "standard_error": lambda v: float(np.mean(v))}
    rows, examples = [], []
    for i, (name, tcfg) in enumerate(ecfg["types"].items()):
        g_mask, c_mask = masks[name]
        g, c = df[g_mask], df[c_mask]
        metric = tcfg["metric"]
        col = "standard_error" if metric == "standard_error" else "res"
        fn = fns[metric]
        g_val, c_val = fn(g[col].values), fn(c[col].values)
        lo, hi = bootstrap_diff(g[col].values, c[col].values, fn, ctx, salt=100 + i) if len(g) > 1 else (np.nan, np.nan)
        sign = 1 if tcfg["expected"] == "higher" else -1
        if np.isnan(lo):
            status = "недостаточно данных"
        elif (lo > 0 and sign > 0) or (hi < 0 and sign < 0):
            status = "подтверждено"
        elif (lo > 0 and sign < 0) or (hi < 0 and sign > 0):
            status = "противоположный эффект"
        else:
            status = "не подтверждено"
        gs, cs = group_stats(g, seeds), group_stats(c, seeds)
        rows.append({
            "type": name, "title": tcfg["title"], "group": tcfg["group"], "comparison": tcfg["comparison"],
            "metric": metric, "expected": tcfg["expected"],
            "n_group": gs["n"], "n_comparison": cs["n"],
            "metric_group": g_val, "metric_comparison": c_val, "diff": g_val - c_val, "diff_ci_low": lo, "diff_ci_high": hi,
            "status": status,
            "rmse_group": gs["rmse"], "rmse_comparison": cs["rmse"], "bias_group": gs["bias"], "bias_comparison": cs["bias"],
            "large_share_group": float(g["large"].mean()) if len(g) else np.nan,
            "large_share_comparison": float(c["large"].mean()) if len(c) else np.nan,
            "large_all_seeds_share_group": float(g["large_all_seeds"].mean()) if len(g) else np.nan,
            "target_mean_group": gs["target_mean"], "target_mean_comparison": cs["target_mean"],
            "se_mean_group": float(g["standard_error"].mean()) if len(g) else np.nan,
            "se_mean_comparison": float(c["standard_error"].mean()) if len(c) else np.nan,
        })
        if tcfg["expected"] == "lower" and metric == "bias":
            pick = g.nsmallest(ecfg["n_examples_per_type"], "res")
        elif metric == "bias":
            pick = g.nlargest(ecfg["n_examples_per_type"], "res")
        else:
            pick = g.reindex(g["res"].abs().sort_values(ascending=False).index).head(ecfg["n_examples_per_type"])
        for r in pick.itertuples(index=False):
            a = acfg["analysis_seed"]
            examples.append({
                "type": name, "id": r.id, "target": r.target, "standard_error": r.standard_error,
                "pred_selected": getattr(r, f"pred_sel_{a}"), "pred_reference": getattr(r, f"pred_ref_{a}"),
                "residual_selected": r.res,
                **{f"residual_selected_seed_{s}": getattr(r, f"res_sel_{s}") for s in seeds},
                "large_all_seeds": bool(r.large_all_seeds), "has_url": bool(r.has_url), "domain": r.domain,
                "words_per_sentence": r.words_per_sentence, "rare_word_share": r.rare_word_share,
                "dale_chall": r.dale_chall, "n_chars": r.n_chars, "flesch_reading_ease": r.flesch_reading_ease,
                "excerpt": " ".join(str(r.excerpt).split())[:400],
            })
    return pd.DataFrame(rows), pd.DataFrame(examples), masks


def near_duplicates(ctx):
    df, a = ctx["df"], ctx["acfg"]["analysis_seed"]
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(df["excerpt"])
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    i, j = np.where(np.triu(sim) >= ctx["acfg"]["errors"]["near_duplicate_min_similarity"])
    rows = []
    for x, y in zip(i, j):
        rows.append({
            "id_a": df.at[x, "id"], "id_b": df.at[y, "id"], "similarity": float(sim[x, y]),
            "target_a": df.at[x, "target"], "target_b": df.at[y, "target"],
            "abs_target_diff": abs(df.at[x, "target"] - df.at[y, "target"]),
            "pred_selected_a": df.at[x, f"pred_sel_{a}"], "pred_selected_b": df.at[y, f"pred_sel_{a}"],
            "se_a": df.at[x, "standard_error"], "se_b": df.at[y, "standard_error"],
            "excerpt_a": " ".join(df.at[x, "excerpt"].split())[:200], "excerpt_b": " ".join(df.at[y, "excerpt"].split())[:200],
        })
    return pd.DataFrame(rows)


def slice_table(ctx, labels, slice_name):
    df, seeds = ctx["df"], ctx["seeds"]
    rows = []
    for k, (label, d) in enumerate(df.groupby(labels, sort=True, observed=True)):
        s, r = group_stats(d, seeds, "sel"), group_stats(d, seeds, "ref")
        lo, hi = bootstrap_metric(d["res"].values, rmse_of_res, ctx, salt=1000 + zlib.crc32(slice_name.encode()) % 1000 + k) if len(d) > 1 else (np.nan, np.nan)
        rows.append({
            "slice": slice_name, "group": str(label), "n": s["n"], "target_mean": s["target_mean"],
            "url_share": float(d["has_url"].mean()), "rmse": s["rmse"], "rmse_ci_low": lo, "rmse_ci_high": hi,
            "bias": s["bias"], "mae": s["mae"], "large_share": float(d["large"].mean()),
            "rmse_reference": r["rmse"], "rmse_gain_vs_reference": r["rmse"] - s["rmse"],
            "small_group": s["n"] < ctx["acfg"]["slices"]["min_domain_size"],
        })
    return pd.DataFrame(rows)


def range_labels(values, bins, prefix):
    codes, edges = pd.qcut(values, bins, labels=False, retbins=True, duplicates="drop")
    names = [f"{prefix}{i + 1} [{edges[i]:.0f}, {edges[i + 1]:.0f}]" if edges[-1] > 50 else
             f"{prefix}{i + 1} [{edges[i]:.2f}, {edges[i + 1]:.2f}]" for i in range(len(edges) - 1)]
    return pd.Series([names[int(c)] for c in codes], index=values.index)


def slices(ctx):
    df, scfg = ctx["df"], ctx["acfg"]["slices"]
    target_labels = range_labels(df["target"], scfg["target_bins"], "Q")
    length_labels = range_labels(df["n_chars"], scfg["length_bins"], "L")
    counts = df["domain"].value_counts()
    big = set(counts[counts >= scfg["min_domain_size"]].index)
    source = np.where(~df["has_url"], "0 без URL",
                      np.where(df["domain"].isin(big), "2 " + df["domain"].fillna(""), "3 другие с URL"))
    url = np.where(df["has_url"], "с URL", "без URL")
    tables = {
        "target": slice_table(ctx, target_labels, "target_quintile"),
        "length": slice_table(ctx, length_labels, "length_tercile"),
        "url": slice_table(ctx, pd.Series(url, index=df.index), "has_url"),
        "source": slice_table(ctx, pd.Series(source, index=df.index), "source_domain"),
    }
    tables["source"]["group"] = tables["source"]["group"].str.slice(2)

    rows = []
    for q, d in df.groupby("target_q"):
        se_hi = d["standard_error"] > d["standard_error"].median()
        lo_d, hi_d = d[~se_hi], d[se_hi]
        diff_lo, diff_hi = bootstrap_diff(hi_d["res"].values, lo_d["res"].values, rmse_of_res, ctx, salt=2000 + int(q))
        for name, g in (("низкий SE", lo_d), ("высокий SE", hi_d)):
            s = group_stats(g, ctx["seeds"])
            rows.append({"target_quintile": f"Q{int(q)}", "se_group": name, "n": s["n"],
                         "se_mean": float(g["standard_error"].mean()), "rmse": s["rmse"], "bias": s["bias"],
                         "rmse_diff_high_minus_low": rmse_of_res(hi_d["res"]) - rmse_of_res(lo_d["res"]),
                         "diff_ci_low": diff_lo, "diff_ci_high": diff_hi})
    tables["se_within_target"] = pd.DataFrame(rows)
    return tables, target_labels


def plot_error_types(types, path):
    n = len(types)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4))
    units = {"rmse": "RMSE", "bias": "bias = mean(pred − target)", "standard_error": "mean standard_error"}
    for ax, r in zip(np.atleast_1d(axes), types.itertuples(index=False)):
        ax.bar(["группа", "сравнение"], [r.metric_group, r.metric_comparison], color=["tab:red", "tab:gray"])
        if r.metric == "bias":
            ax.axhline(0, color="black", lw=1)
        ax.set_title(f"{r.title}\n{r.status}", fontsize=9)
        ax.set_ylabel(units[r.metric], fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"группа\nn={r.n_group}", f"сравнение\nn={r.n_comparison}"], fontsize=8)
        ci = "" if np.isnan(r.diff_ci_low) else f"\n95% ДИ [{r.diff_ci_low:+.3f}, {r.diff_ci_high:+.3f}]"
        ax.text(0.5, 0.97, f"Δ = {r.diff:+.3f}{ci}", transform=ax.transAxes, ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="lightgray"))
    fig.suptitle("Типы ошибок: группа против сравнения (seed анализа), Δ = группа − сравнение", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def binned(ax, x, y, bins, color):
    q = pd.qcut(pd.Series(x).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"x": x, "y": y, "q": q.values}).groupby("q")
    ax.plot(g["x"].mean(), g["y"].mean(), marker="o", color=color, label="среднее по бинам")


def plot_mechanisms(ctx, masks, path):
    df = ctx["df"]
    median = df["target"].median()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    hard = df[df["target"] <= median]
    axes[0].scatter(hard["rare_word_share"], hard["res"], s=4, alpha=0.3)
    binned(axes[0], hard["rare_word_share"].values, hard["res"].values, 8, "tab:red")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("доля редких слов (встречаются не более чем в 2 текстах)")
    axes[0].set_ylabel("pred − target")
    axes[0].set_title("Сложные тексты (target ≤ медианы): больше редких слов – завышена легкость", fontsize=9)
    axes[0].legend(fontsize=8)
    easy = df[df["target"] > median]
    x = easy["words_per_sentence"].clip(upper=60)
    axes[1].scatter(x, easy["res"], s=4, alpha=0.3)
    binned(axes[1], x.values, easy["res"].values, 8, "tab:red")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("слов в предложении (обрезано на 60)")
    axes[1].set_ylabel("pred − target")
    axes[1].set_title("Легкие тексты (target > медианы): длинные предложения – занижена легкость", fontsize=9)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_residuals_by_quintile(ctx, labels, path):
    df, a = ctx["df"], ctx["acfg"]["analysis_seed"]
    order = sorted(labels.unique())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    pos = np.arange(len(order))
    data_ref = [df.loc[labels == g, f"res_ref_{a}"].values for g in order]
    data_sel = [df.loc[labels == g, f"res_sel_{a}"].values for g in order]
    b1 = ax.boxplot(data_ref, positions=pos - 0.18, widths=0.32, patch_artist=True, showfliers=False)
    b2 = ax.boxplot(data_sel, positions=pos + 0.18, widths=0.32, patch_artist=True, showfliers=False)
    for b, c in ((b1, "lightgray"), (b2, "tab:blue")):
        for patch in b["boxes"]:
            patch.set_facecolor(c)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlabel("квинтиль таргета (Q1 – самые сложные)")
    ax.set_ylabel("pred − target")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], [ctx["acfg"]["reference_experiment"], ctx["acfg"]["selected_experiment"]], fontsize=8)
    ax.set_title(f"Распределение остатков по уровню сложности (seed {a})")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_slices(tables, path):
    panels = [("length", "Длина отрывка (символы)"), ("source", "Источник текста"), ("url", "Наличие URL")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1, 2, 0.8]})
    for ax, (key, title) in zip(axes, panels):
        t = tables[key]
        x = np.arange(len(t))
        ax.errorbar(x, t["rmse"], yerr=[t["rmse"] - t["rmse_ci_low"], t["rmse_ci_high"] - t["rmse"]], fmt="o",
                    capsize=4, color="tab:blue", label="итоговая модель (95% ДИ)")
        ax.scatter(x, t["rmse_reference"], marker="x", color="gray", label="бейзлайн")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\nn={n}" for g, n in zip(t["group"], t["n"])], fontsize=7, rotation=0)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("RMSE")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_se_within_target(t, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    qs = t["target_quintile"].unique()
    x = np.arange(len(qs))
    for k, (name, color) in enumerate((("низкий SE", "tab:green"), ("высокий SE", "tab:red"))):
        sub = t[t["se_group"] == name].set_index("target_quintile").loc[qs]
        ax.bar(x + (k - 0.5) * 0.38, sub["rmse"], width=0.38, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(qs)
    ax.set_xlabel("квинтиль таргета (Q1 – самые сложные)")
    ax.set_ylabel("RMSE")
    ax.set_title("Шум разметки отдельно от уровня сложности: SE выше / ниже медианы внутри квинтиля")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def output_dirs(acfg):
    out = resolve(acfg["output"])
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return out, tables, figures


def main(config_path):
    ctx = load_context(config_path)
    acfg = ctx["acfg"]
    out, tdir, fdir = output_dirs(acfg)
    sel = ctx["run_metrics"]["sel"]
    folds_hash = file_sha256(resolve(ctx["sel_cfg"]["paths"]["folds"]))
    assert sel["run_info"]["folds_sha256"] == folds_hash, "folds.csv differs from the one used in HW2"

    types, examples, masks = error_types(ctx)
    types.to_csv(tdir / "error_types.csv", index=False)
    examples.to_csv(tdir / "error_examples.csv", index=False)
    dups = near_duplicates(ctx)
    dups.to_csv(tdir / "near_duplicates.csv", index=False)
    tables, target_labels = slices(ctx)
    for name, t in tables.items():
        t.to_csv(tdir / f"slices_{name}.csv", index=False)

    plot_error_types(types, fdir / "01_error_types.png")
    plot_mechanisms(ctx, masks, fdir / "02_error_mechanisms.png")
    plot_residuals_by_quintile(ctx, target_labels, fdir / "03_residuals_by_quintile.png")
    plot_slices(tables, fdir / "04_slices.png")
    plot_se_within_target(tables["se_within_target"], fdir / "05_se_within_target.png")

    df = ctx["df"]
    summary = {
        "selected_experiment": acfg["selected_experiment"], "reference_experiment": acfg["reference_experiment"],
        "analysis_seed": acfg["analysis_seed"], "folds_sha256": folds_hash, "hw2_git_commit": sel["run_info"].get("git_commit"),
        "n_texts": int(len(df)), "large_error_threshold": acfg["errors"]["large_error_threshold"],
        "large_share": float(df["large"].mean()), "large_all_seeds_share": float(df["large_all_seeds"].mean()),
        "rmse_selected_mean_over_seeds": float(np.mean([rmse_of_res(df[f"res_sel_{s}"]) for s in ctx["seeds"]])),
        "n_near_duplicate_pairs": int(len(dups)),
    }
    (tdir / "error_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 200)
    print(f"Selected: {acfg['selected_experiment']}, RMSE {summary['rmse_selected_mean_over_seeds']:.4f}; "
          f"large errors (|res| >= {summary['large_error_threshold']}): {summary['large_share']:.1%}, "
          f"on all seeds: {summary['large_all_seeds_share']:.1%}")
    print(types[["type", "n_group", "n_comparison", "metric", "metric_group", "metric_comparison", "diff",
                 "diff_ci_low", "diff_ci_high", "status"]].round(4).to_string(index=False))
    print(f"near-duplicate pairs (similarity >= {acfg['errors']['near_duplicate_min_similarity']}): {len(dups)}")
    for name in ("target", "length", "source", "url"):
        print(f"--- slice: {name}")
        print(tables[name][["group", "n", "target_mean", "rmse", "rmse_ci_low", "rmse_ci_high", "bias", "rmse_reference"]].round(3).to_string(index=False))
    print("--- SE within target quintile")
    print(tables["se_within_target"][["target_quintile", "se_group", "n", "se_mean", "rmse", "diff_ci_low", "diff_ci_high"]].round(3).to_string(index=False))
    print(f"Saved error analysis tables to {tdir} and figures to {fdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ANALYSIS_CONFIG)
    main(parser.parse_args().config)
