from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


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


MODELS = {"tfidf_ridge": build_tfidf_ridge}


def build_model(cfg):
    return MODELS[cfg["model"]["name"]](cfg["model"]["params"])
