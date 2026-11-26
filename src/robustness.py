import argparse
import json
import re
import zlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.error_analysis import ANALYSIS_CONFIG, FEATURES, load_context, output_dirs, rmse_of_res
from src.features import build_features
from src.models import build_model, selected_alpha
from src.registry import feature_columns
from src.utils import compute_metrics

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_TOKEN = re.compile(r"[A-Za-z]+")


def make_inputs(cfg, ids, texts, feats=None):
    cols = feature_columns(cfg)
    df = pd.DataFrame({"id": np.asarray(ids), "excerpt": np.asarray(texts, dtype=object)})
    if cols:
        f = feats if feats is not None else build_features(df["excerpt"])
        for c in cols:
            df[c] = np.asarray(f[c])
    return df


def fold_iter(ctx):
    for seed in ctx["seeds"]:
        fold_col = ctx["folds"][f"fold_{seed}"].values
        for k in sorted(np.unique(fold_col)):
            yield seed, int(k), np.where(fold_col != k)[0], np.where(fold_col == k)[0]


def summarize_runs(rows_df, keys):
    out = []
    for key, g in rows_df.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, key))
        rec.update({"n_folds": len(g), "n_train_mean": float(g["n_train"].mean()),
                    "rmse_mean": float(g["rmse"].mean()), "rmse_std": float(g["rmse"].std(ddof=1)),
                    "spearman_mean": float(g["spearman"].mean())})
        if "alpha" in g and g["alpha"].notna().any():
            rec["alpha_median"] = float(g["alpha"].median())
        out.append(rec)
    return pd.DataFrame(out)


def oof_summary(oof_by_seed, y):
    vals = [compute_metrics(y, p) for p in oof_by_seed.values()]
    ratios = [float(np.std(p) / np.std(y)) for p in oof_by_seed.values()]
    return {"oof_rmse": float(np.mean([v["rmse"] for v in vals])), "oof_spearman": float(np.mean([v["spearman"] for v in vals])),
            "std_ratio": float(np.mean(ratios))}


def train_size(ctx):
    rcfg = ctx["acfg"]["robustness"]
    df, y = ctx["df"], ctx["df"]["target"].values
    models = {"selected": ctx["sel_cfg"], "reference": ctx["ref_cfg"]}
    inputs = {role: make_inputs(cfg, df["id"], df["excerpt"], df[FEATURES]) for role, cfg in models.items()}
    rows, oofs = [], {}
    for seed, k, tr, va in fold_iter(ctx):
        perm = np.random.default_rng([rcfg["subsample_seed"], seed, k]).permutation(tr)
        for frac in rcfg["train_fractions"]:
            sub = np.sort(perm[: int(round(frac * len(tr)))])
            for role, cfg in models.items():
                model = build_model(cfg)
                X = inputs[role]
                model.fit(X.iloc[sub], y[sub])
                pred = model.predict(X.iloc[va])
                oofs.setdefault((role, frac), {}).setdefault(seed, np.full(len(df), np.nan))[va] = pred
                m = compute_metrics(y[va], pred)
                rows.append({"check": "train_size", "model": role, "train_fraction": frac, "seed": seed, "fold": k,
                             "n_train": len(sub), "rmse": m["rmse"], "spearman": m["spearman"], "alpha": selected_alpha(model)})
    folds_df = pd.DataFrame(rows)
    summary = summarize_runs(folds_df, ["model", "train_fraction"])
    extra = pd.DataFrame([oof_summary(oofs[(r["model"], r["train_fraction"])], y) for _, r in summary.iterrows()])
    summary = pd.concat([summary, extra], axis=1)
    full = summary[summary["train_fraction"] == 1.0].set_index("model")["rmse_mean"]
    summary["delta_vs_full"] = summary.apply(lambda r: r["rmse_mean"] - full[r["model"]], axis=1)
    return folds_df, summary, oofs


def typo_word(word, rng):
    i = int(rng.integers(0, len(word)))
    op = int(rng.integers(0, 4))
    if op == 0 and len(word) > 1:
        i = min(i, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == 1 and len(word) > 1:
        return word[:i] + word[i + 1:]
    if op == 2:
        c = LETTERS[int(rng.integers(0, 26))]
        return word[:i] + (c.upper() if word[i].isupper() else c) + word[i + 1:]
    c = LETTERS[int(rng.integers(0, 26))]
    return word[:i] + c + word[i:]


def add_typos(text, text_id, rate, rcfg):
    if rate == 0:
        return text, 0, 0
    rng = np.random.default_rng([rcfg["typo_seed"], zlib.crc32(str(text_id).encode()), int(round(rate * 1000))])
    changed = total = 0

    def repl(m):
        nonlocal changed, total
        w = m.group(0)
        if len(w) < rcfg["typo_min_word_length"]:
            return w
        total += 1
        if rng.random() < rate:
            changed += 1
            return typo_word(w, rng)
        return w

    return WORD_TOKEN.sub(repl, text), changed, total


def typos(ctx):
    rcfg = ctx["acfg"]["robustness"]
    df, y = ctx["df"], ctx["df"]["target"].values
    models = {"selected": ctx["sel_cfg"], "reference": ctx["ref_cfg"]}
    clean = {role: make_inputs(cfg, df["id"], df["excerpt"], df[FEATURES]) for role, cfg in models.items()}
    noisy, realized, example = {}, {}, None
    for rate in rcfg["typo_rates"]:
        res = [add_typos(t, i, rate, rcfg) for t, i in zip(df["excerpt"], df["id"])]
        texts = [r[0] for r in res]
        realized[rate] = sum(r[1] for r in res) / max(sum(r[2] for r in res), 1)
        feats = df[FEATURES] if rate == 0 else build_features(texts)
        noisy[rate] = {role: make_inputs(cfg, df["id"], texts, feats) for role, cfg in models.items()}
        if rate == max(rcfg["typo_rates"]):
            example = {"id": df["id"].iloc[0], "rate": rate, "clean": " ".join(df["excerpt"].iloc[0].split())[:300],
                       "noisy": " ".join(texts[0].split())[:300]}
    rows, oofs = [], {}
    for seed, k, tr, va in fold_iter(ctx):
        for role, cfg in models.items():
            model = build_model(cfg)
            model.fit(clean[role].iloc[tr], y[tr])
            for rate in rcfg["typo_rates"]:
                pred = model.predict(noisy[rate][role].iloc[va])
                oofs.setdefault((role, rate), {}).setdefault(seed, np.full(len(df), np.nan))[va] = pred
                m = compute_metrics(y[va], pred)
                rows.append({"check": "typos", "model": role, "typo_rate": rate, "seed": seed, "fold": k, "n_train": len(tr),
                             "rmse": m["rmse"], "spearman": m["spearman"], "alpha": selected_alpha(model)})
    folds_df = pd.DataFrame(rows)
    summary = summarize_runs(folds_df, ["model", "typo_rate"])
    extra = pd.DataFrame([{**oof_summary(oofs[(r["model"], r["typo_rate"])], y),
                           "realized_word_typo_share": realized[r["typo_rate"]]} for _, r in summary.iterrows()])
    summary = pd.concat([summary, extra], axis=1)
    clean_rmse = summary[summary["typo_rate"] == 0].set_index("model")["rmse_mean"]
    summary["delta_vs_clean"] = summary.apply(lambda r: r["rmse_mean"] - clean_rmse[r["model"]], axis=1)
    return folds_df, summary, oofs, example


def domain_holdout(ctx):
    rcfg = ctx["acfg"]["robustness"]
    df, y, a = ctx["df"], ctx["df"]["target"].values, ctx["acfg"]["analysis_seed"]
    counts = df["domain"].value_counts()
    domains = sorted(counts[counts >= rcfg["domain_min_size"]].index)
    rows = []
    for i, d in enumerate(domains):
        test = np.where(df["domain"] == d)[0]
        train = np.where(df["domain"] != d)[0]
        for role, cfg in (("selected", ctx["sel_cfg"]), ("reference", ctx["ref_cfg"])):
            X = make_inputs(cfg, df["id"], df["excerpt"], df[FEATURES])
            model = build_model(cfg)
            model.fit(X.iloc[train], y[train])
            pred = model.predict(X.iloc[test])
            key = "sel" if role == "selected" else "ref"
            cv_pred = df[f"pred_{key}_{a}"].values[test]
            res_hold, res_cv = pred - y[test], cv_pred - y[test]
            b = ctx["acfg"]["bootstrap"]
            rng = np.random.default_rng([b["seed"], 3000 + i])
            idx = rng.integers(0, len(test), size=(b["iters"], len(test)))
            samples = np.sqrt(np.mean(res_hold[idx] ** 2, axis=1)) - np.sqrt(np.mean(res_cv[idx] ** 2, axis=1))
            lo, hi = np.quantile(samples, [(1 - b["ci"]) / 2, 1 - (1 - b["ci"]) / 2])
            rows.append({
                "domain": d, "model": role, "n_test": len(test), "n_train": len(train),
                "target_mean": float(y[test].mean()),
                "rmse_cv": rmse_of_res(res_cv), "rmse_domain_holdout": rmse_of_res(res_hold),
                "delta": rmse_of_res(res_hold) - rmse_of_res(res_cv), "delta_ci_low": float(lo), "delta_ci_high": float(hi),
                "bias_cv": float(res_cv.mean()), "bias_domain_holdout": float(res_hold.mean()),
                "cv_train_size_mean": float(np.mean([np.sum(ctx["folds"][f"fold_{s}"].values != k)
                                                     for s in ctx["seeds"] for k in np.unique(ctx["folds"][f"fold_{s}"])])),
            })
    return pd.DataFrame(rows)


def reproduction_check(ctx, oofs, key):
    df = ctx["df"]
    out = {}
    for role, col in (("selected", "sel"), ("reference", "ref")):
        diffs = [float(np.max(np.abs(oofs[(role, key)][s] - df[f"pred_{col}_{s}"].values))) for s in ctx["seeds"]]
        out[role] = max(diffs)
    return out


def plot_train_size(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for role, color in (("selected", "tab:blue"), ("reference", "gray")):
        s = summary[summary["model"] == role].sort_values("train_fraction")
        x = s["n_train_mean"]
        axes[0].errorbar(x, s["rmse_mean"], yerr=s["rmse_std"], fmt="o-", capsize=3, color=color, label=role)
        axes[1].plot(x, s["std_ratio"], "o-", color=color, label=role)
    axes[0].set_xlabel("размер обучающей выборки (текстов)")
    axes[0].set_ylabel("RMSE (mean ± std по 15 фолдам)")
    axes[0].set_title("Кривая обучения: та же валидация, меньше train")
    axes[1].set_xlabel("размер обучающей выборки (текстов)")
    axes[1].set_ylabel("std(pred) / std(target)")
    axes[1].set_title("Сжатие предсказаний к среднему")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_typos(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for role, color in (("selected", "tab:blue"), ("reference", "gray")):
        s = summary[summary["model"] == role].sort_values("typo_rate")
        x = s["typo_rate"] * 100
        axes[0].errorbar(x, s["rmse_mean"], yerr=s["rmse_std"], fmt="o-", capsize=3, color=color, label=role)
        axes[1].plot(x, s["spearman_mean"], "o-", color=color, label=role)
    axes[0].set_xlabel("доля слов с опечаткой в валидации, %")
    axes[0].set_ylabel("RMSE (mean ± std по 15 фолдам)")
    axes[0].set_title("Устойчивость к опечаткам (обучение на чистом тексте)")
    axes[1].set_xlabel("доля слов с опечаткой в валидации, %")
    axes[1].set_ylabel("Spearman")
    axes[1].set_title("Сохранение порядка текстов по сложности")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_domains(t, path):
    s = t[t["model"] == "selected"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(s) + 1.8))
    y = np.arange(len(s))[::-1]
    for yi, r in zip(y, s.itertuples(index=False)):
        ax.plot([r.rmse_cv, r.rmse_domain_holdout], [yi, yi], color="lightgray", zorder=1)
    ax.scatter(s["rmse_cv"], y, color="tab:blue", label="обычная CV (источник есть в train)", zorder=2)
    ax.scatter(s["rmse_domain_holdout"], y, color="tab:red", label="источник исключен из train", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} (n={n})" for d, n in zip(s["domain"], s["n_test"])], fontsize=8)
    ax.set_xlabel("RMSE итоговой модели на текстах источника")
    ax.set_title(f"Перенос на новый источник (train без источника: ~{int(s['n_train'].mean())} текстов, в CV: ~{int(s['cv_train_size_mean'].iloc[0])})", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(config_path):
    ctx = load_context(config_path)
    acfg = ctx["acfg"]
    out, tdir, fdir = output_dirs(acfg)

    ts_folds, ts_summary, ts_oofs = train_size(ctx)
    ty_folds, ty_summary, ty_oofs, typo_example = typos(ctx)
    dom = domain_holdout(ctx)

    pd.concat([ts_folds, ty_folds], ignore_index=True).to_csv(tdir / "robustness_folds.csv", index=False)
    ts_summary.to_csv(tdir / "robustness_train_size.csv", index=False)
    ty_summary.to_csv(tdir / "robustness_typos.csv", index=False)
    dom.to_csv(tdir / "robustness_domain_holdout.csv", index=False)

    plot_train_size(ts_summary, fdir / "06_train_size.png")
    plot_typos(ty_summary, fdir / "07_typos.png")
    plot_domains(dom, fdir / "08_domain_holdout.png")

    tol = acfg["robustness"]["reproduction_tol"]
    rep_ts = reproduction_check(ctx, ts_oofs, 1.0)
    rep_ty = reproduction_check(ctx, ty_oofs, 0.0)
    checks = {
        "train_fraction_1.0_reproduces_hw2_oof_max_abs_diff": rep_ts,
        "typo_rate_0_reproduces_hw2_oof_max_abs_diff": rep_ty,
        "reproduced": all(v < tol for v in list(rep_ts.values()) + list(rep_ty.values())),
        "tolerance": tol,
        "typo_example": typo_example,
    }
    (tdir / "robustness_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 200)
    print("--- train size")
    print(ts_summary[["model", "train_fraction", "n_train_mean", "rmse_mean", "rmse_std", "delta_vs_full", "spearman_mean", "std_ratio"]].round(4).to_string(index=False))
    print("--- typos")
    print(ty_summary[["model", "typo_rate", "realized_word_typo_share", "rmse_mean", "rmse_std", "delta_vs_clean", "spearman_mean", "std_ratio"]].round(4).to_string(index=False))
    print("--- domain holdout")
    print(dom[["domain", "model", "n_test", "target_mean", "rmse_cv", "rmse_domain_holdout", "delta", "delta_ci_low", "delta_ci_high"]].round(4).to_string(index=False))
    print(f"HW2 reproduction (fraction 1.0 / typo rate 0): {rep_ts} / {rep_ty} -> {'reproduced' if checks['reproduced'] else 'NOT REPRODUCED'}")
    print(f"Saved robustness tables to {tdir} and figures to {fdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ANALYSIS_CONFIG)
    main(parser.parse_args().config)
