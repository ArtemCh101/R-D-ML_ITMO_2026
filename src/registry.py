import argparse
import sys

import numpy as np

from src.utils import ROOT, load_config

EXPERIMENTS_DIR = "configs/experiments"
SHARED_KEYS = ["paths", "cv", "eval", "model_defaults", "features", "comparison", "tracking"]
TUNING_MODES = {"fixed", "nested_loo"}


class RegistryError(Exception):
    pass


def load_experiments(directory=EXPERIMENTS_DIR):
    exps = {}
    for path in sorted((ROOT / directory).glob("*.yaml"), key=lambda p: (not p.stem.startswith("e"), p.stem)):
        if path.stem == "common":
            continue
        cfg = load_config(path)
        if cfg.get("experiment") != path.stem:
            raise RegistryError(f"{path.name}: 'experiment' must equal file name ({cfg.get('experiment')})")
        exps[path.stem] = cfg
    return exps


def alpha_grid(cfg):
    g = cfg["model_defaults"]["alpha_grid"]
    return np.logspace(g["log10_start"], g["log10_stop"], g["num"])


def feature_columns(cfg, groups=None):
    defined = cfg["features"]["groups"]
    groups = cfg["model"]["feature_groups"] if groups is None else groups
    return [col for g in groups for col in defined[g]]


def signature(cfg):
    m = cfg["model"]
    return {"use_tfidf": bool(m["use_tfidf"]), "groups": frozenset(m["feature_groups"]), "tuning": m["tuning"]}


def components_diff(a, b):
    diff = set()
    if a["tuning"] != b["tuning"]:
        diff.add("tuning")
    if a["use_tfidf"] != b["use_tfidf"]:
        diff.add("tfidf")
    diff |= {f"features:{g}" for g in a["groups"] ^ b["groups"]}
    return diff


def expand(components, groups):
    out = set()
    for c in components:
        out |= {f"features:{g}" for g in groups} if c == "features:all" else {c}
    return out


def validate(exps):
    errors = []
    if not exps:
        raise RegistryError("no experiments found")
    first = next(iter(exps.values()))
    groups = first["features"]["groups"]
    comp = first["comparison"]

    for key in SHARED_KEYS:
        for name, cfg in exps.items():
            if cfg.get(key) != first.get(key):
                errors.append(f"{name}: shared section '{key}' differs from other experiments")

    base_cfg = load_config("configs/baseline.yaml")
    base_tfidf = {k: v for k, v in base_cfg["model"]["params"].items() if k != "alpha"}
    if first["model_defaults"]["tfidf"] != base_tfidf:
        errors.append("model_defaults.tfidf differs from configs/baseline.yaml")
    if first["model_defaults"]["fixed_alpha"] != base_cfg["model"]["params"]["alpha"]:
        errors.append("model_defaults.fixed_alpha differs from baseline alpha")

    cols = [c for g in groups.values() for c in g]
    if len(cols) != len(set(cols)):
        errors.append("feature groups overlap")
    try:
        from src.features import text_features
        known = set(text_features("A simple test sentence. Another one here."))
        if set(cols) != known:
            errors.append(f"feature groups do not match src/features.py: {sorted(set(cols) ^ known)}")
    except ImportError as e:
        print(f"[warn] feature names not checked against src/features.py: {e}")

    baselines = []
    for name, cfg in exps.items():
        m = cfg.get("model", {})
        if m.get("name") != "linear_text":
            errors.append(f"{name}: model.name must be 'linear_text'")
            continue
        if m.get("tuning") not in TUNING_MODES:
            errors.append(f"{name}: tuning must be one of {sorted(TUNING_MODES)}")
        unknown = set(m.get("feature_groups", [])) - set(groups)
        if unknown:
            errors.append(f"{name}: unknown feature groups {sorted(unknown)}")
        if not m.get("use_tfidf") and not m.get("feature_groups"):
            errors.append(f"{name}: model has no inputs")
        if not str(cfg.get("hypothesis", "")).strip():
            errors.append(f"{name}: empty hypothesis")
        if "improvement" not in cfg and "ablation" not in cfg:
            baselines.append(name)

    if baselines != [comp["reference"]]:
        errors.append(f"exactly one baseline expected ({comp['reference']}), found {baselines}")
    if comp["final"] not in exps:
        errors.append(f"final experiment {comp['final']} not found")
    if errors:
        raise RegistryError("\n".join(errors))

    ref = signature(exps[comp["reference"]])
    if ref != {"use_tfidf": True, "groups": frozenset(), "tuning": "fixed"}:
        errors.append(f"{comp['reference']} must be TF-IDF only with fixed alpha")

    for name, cfg in exps.items():
        sig = signature(cfg)
        for role, other_key, comp_key in [("improvement", "parent", "adds"), ("ablation", "of", "removes")]:
            if role not in cfg:
                continue
            other = cfg[role][other_key]
            if other not in exps:
                errors.append(f"{name}: {role} refers to unknown experiment {other}")
                continue
            actual = components_diff(sig, signature(exps[other]))
            declared = expand(cfg[role][comp_key], groups)
            if actual != declared:
                errors.append(f"{name}: declared {role} {sorted(declared)} but differs from {other} by {sorted(actual)}")
            if role == "ablation" and other != comp["final"]:
                errors.append(f"{name}: ablations must be relative to final model {comp['final']}")
    if errors:
        raise RegistryError("\n".join(errors))
    return True


def comparison_pairs(exps):
    comp = next(iter(exps.values()))["comparison"]
    pairs = []
    for name, cfg in exps.items():
        if "improvement" in cfg:
            pairs.append({"model": name, "ref": cfg["improvement"]["parent"], "kind": "improvement"})
    pairs.append({"model": comp["final"], "ref": comp["reference"], "kind": "total"})
    for name, cfg in exps.items():
        if "ablation" in cfg:
            pairs.append({"model": name, "ref": cfg["ablation"]["of"], "kind": "ablation"})
    return pairs


def describe(cfg):
    m = cfg["model"]
    parts = []
    if m["use_tfidf"]:
        parts.append("TF-IDF")
    if m["feature_groups"]:
        parts.append("features[" + ",".join(m["feature_groups"]) + "]")
    parts.append("RidgeCV(nested LOO)" if m["tuning"] == "nested_loo" else f"Ridge(alpha={cfg['model_defaults']['fixed_alpha']})")
    return " + ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=EXPERIMENTS_DIR)
    args = parser.parse_args()
    try:
        exps = load_experiments(args.dir)
        validate(exps)
    except RegistryError as e:
        print(f"REGISTRY INVALID:\n{e}")
        sys.exit(1)
    grid = alpha_grid(next(iter(exps.values())))
    print(f"Registry OK: {len(exps)} experiments, alpha grid {grid[0]:g}..{grid[-1]:g} ({len(grid)} points)")
    for name, cfg in exps.items():
        role = "baseline"
        if "improvement" in cfg:
            role = f"improvement of {cfg['improvement']['parent']}"
        elif "ablation" in cfg:
            role = f"ablation of {cfg['ablation']['of']}: -{','.join(cfg['ablation']['removes'])}"
        print(f"  {name:22s} {describe(cfg):70s} {role}")
    print("Comparisons:")
    for p in comparison_pairs(exps):
        print(f"  [{p['kind']:11s}] {p['model']} vs {p['ref']}")


if __name__ == "__main__":
    main()
