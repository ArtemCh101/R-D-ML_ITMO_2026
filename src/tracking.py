import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

from src.utils import ROOT, resolve


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def run_info(cfg):
    import sklearn
    info = {
        **git_info(),
        "folds_sha256": file_sha256(resolve(cfg["paths"]["folds"])),
        "train_sha256": file_sha256(resolve(cfg["paths"]["train"])),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    if info["git_dirty"]:
        print("[warn] git working tree has uncommitted changes – the run is not tied to a clean commit")
    return info


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@contextlib.contextmanager
def in_root():
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(old)


def _ensure_experiment(mlflow, tracking):
    name = tracking["experiment_name"]
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        location = resolve(tracking.get("artifact_location", "mlruns"))
        location.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=location.as_uri())
    mlflow.set_experiment(name)


def _log(mlflow, cfg, metrics, fold_df, out_dir, info):
    _ensure_experiment(mlflow, cfg["tracking"])
    role = "ablation" if "ablation" in cfg else "improvement" if "improvement" in cfg else "baseline"
    with mlflow.start_run(run_name=cfg["experiment"]):
        tags = {
            "experiment_id": cfg["experiment"],
            "role": role,
            "title": cfg.get("title", ""),
            "hypothesis": " ".join(str(cfg.get("hypothesis", "")).split()),
            "parent": cfg.get("improvement", {}).get("parent", ""),
            "ablation_of": cfg.get("ablation", {}).get("of", ""),
            **{k: str(v) for k, v in info.items()},
        }
        mlflow.set_tags(tags)
        params = flatten({"model": cfg["model"], "cv": cfg["cv"], "model_defaults": cfg["model_defaults"]})
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        scalars = {}
        for name, stats in metrics["per_fold"].items():
            scalars[f"{name}_mean"] = stats["mean"]
            scalars[f"{name}_std"] = stats["std"]
        for seed, m in metrics["oof_per_seed"].items():
            scalars[f"oof_rmse_seed_{seed}"] = m["rmse"]
        scalars["oof_rmse_across_seeds_mean"] = metrics["oof_rmse_across_seeds"]["mean"]
        mlflow.log_metrics(scalars)
        if fold_df is not None:
            for step, row in enumerate(fold_df.itertuples(index=False)):
                mlflow.log_metric("rmse_fold", row.rmse, step=step)
        mlflow.log_dict(cfg, "config.json")
        if out_dir is not None:
            mlflow.log_artifacts(str(out_dir), artifact_path="reports")


def log_run(cfg, metrics, fold_df=None, out_dir=None, info=None):
    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False):
        return False
    try:
        import mlflow
    except ImportError:
        print("[warn] mlflow is not installed – run is not tracked (results are still saved to reports/)")
        return False
    info = info or run_info(cfg)
    with in_root():
        mlflow.set_tracking_uri(tracking["uri"])
        _log(mlflow, cfg, metrics, fold_df, out_dir, info)
    return True


def selftest():
    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    tmp = ROOT / ".mlflow_selftest"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        cfg = {
            "experiment": "selftest",
            "title": "selftest",
            "hypothesis": "selftest",
            "tracking": {"enabled": True, "uri": "sqlite:///.mlflow_selftest/selftest.db",
                         "experiment_name": "selftest", "artifact_location": ".mlflow_selftest/artifacts"},
            "model": {"name": "linear_text", "use_tfidf": True, "feature_groups": [], "tuning": "fixed"},
            "cv": {"n_splits": 5, "seeds": [42]},
            "model_defaults": {"fixed_alpha": 1.0},
        }
        metrics = {
            "per_fold": {"rmse": {"mean": 0.7, "std": 0.01}},
            "oof_per_seed": {"42": {"rmse": 0.71}},
            "oof_rmse_across_seeds": {"mean": 0.71, "std": 0.0},
        }
        fold_df = pd.DataFrame({"rmse": [0.69, 0.70, 0.71]})
        art = tmp / "artifacts_src"
        art.mkdir()
        (art / "note.txt").write_text("artifact check")
        info = {"git_commit": git_info()["git_commit"], "folds_sha256": "test"}
        cwd = os.getcwd()
        os.chdir(tempfile.gettempdir())
        try:
            assert log_run(cfg, metrics, fold_df, art, info)
        finally:
            os.chdir(cwd)
        with in_root():
            mlflow.set_tracking_uri(cfg["tracking"]["uri"])
            client = MlflowClient()
            exp = client.get_experiment_by_name("selftest")
            runs = client.search_runs([exp.experiment_id])
            assert len(runs) == 1, runs
            run = runs[0]
            assert abs(run.data.metrics["rmse_mean"] - 0.7) < 1e-12
            assert run.data.tags["experiment_id"] == "selftest"
            history = client.get_metric_history(run.info.run_id, "rmse_fold")
            assert [m.value for m in history] == [0.69, 0.70, 0.71]
            files = {f.path for f in client.list_artifacts(run.info.run_id, "reports")}
            assert "reports/note.txt" in files, files
        print(f"tracking selftest OK: mlflow {mlflow.__version__}, sqlite backend in '{ROOT}' works, "
              f"params/metrics/tags/artifacts logged and read back")
        print(f"git: {json.dumps(git_info())}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    if parser.parse_args().selftest:
        selftest()
