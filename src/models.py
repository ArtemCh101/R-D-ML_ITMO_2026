import inspect

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.registry import alpha_grid, feature_columns

STORE_CV_ARG = "store_cv_results" if "store_cv_results" in inspect.signature(RidgeCV).parameters else "store_cv_values"


def build_tfidf_ridge(params):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=tuple(params["ngram_range"]),
            min_df=params["min_df"],
            max_features=params["max_features"],
            sublinear_tf=params["sublinear_tf"],
            lowercase=params["lowercase"],
        )),
        ("ridge", Ridge(alpha=params["alpha"])),
    ])


def build_linear_text(cfg):
    m = cfg["model"]
    d = cfg["model_defaults"]
    blocks = []
    if m["use_tfidf"]:
        t = d["tfidf"]
        blocks.append(("tfidf", TfidfVectorizer(
            ngram_range=tuple(t["ngram_range"]),
            min_df=t["min_df"],
            max_features=t["max_features"],
            sublinear_tf=t["sublinear_tf"],
            lowercase=t["lowercase"],
        ), "excerpt"))
    cols = feature_columns(cfg)
    if cols:
        if d["feature_scaling"] != "standard":
            raise ValueError(f"unsupported feature_scaling {d['feature_scaling']}")
        blocks.append(("feats", StandardScaler(), cols))
    if m["tuning"] == "nested_loo":
        reg = RidgeCV(alphas=alpha_grid(cfg), **{STORE_CV_ARG: True})
    else:
        reg = RidgeCV(alphas=[d["fixed_alpha"]])
    return Pipeline([("features", ColumnTransformer(blocks)), ("ridge", reg)])


def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "tfidf_ridge":
        return build_tfidf_ridge(cfg["model"]["params"])
    if name == "linear_text":
        return build_linear_text(cfg)
    raise ValueError(f"unknown model {name}")


def model_inputs(cfg, df):
    if cfg["model"]["name"] == "tfidf_ridge":
        return df["excerpt"].values
    from src.features import attach_features
    return attach_features(df, cfg, feature_columns(cfg))


def inner_cv_curve(model):
    reg = model[-1]
    if not isinstance(reg, RidgeCV):
        return None
    values = getattr(reg, "cv_results_", None)
    if values is None:
        values = getattr(reg, "cv_values_")
    mse = values.reshape(values.shape[0], -1).mean(axis=0)
    return list(zip(reg.alphas, mse ** 0.5))


def selected_alpha(model):
    reg = model[-1]
    return float(reg.alpha_) if isinstance(reg, RidgeCV) else float(reg.alpha)


def coefficients(model):
    names = model[:-1].get_feature_names_out()
    return names, model[-1].coef_
