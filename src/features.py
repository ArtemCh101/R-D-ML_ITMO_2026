import argparse
import re

import pandas as pd
import textstat

from src.utils import load_config, resolve

WORD_RE = re.compile(r"[A-Za-z']+")


def text_features(text):
    words = WORD_RE.findall(text)
    n_words = max(len(words), 1)
    n_sent = max(textstat.sentence_count(text), 1)
    syllables = [textstat.syllable_count(w) for w in words]
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_sentences": n_sent,
        "words_per_sentence": n_words / n_sent,
        "chars_per_word": sum(len(w) for w in words) / n_words,
        "syllables_per_word": sum(syllables) / n_words,
        "long_word_share": sum(len(w) >= 7 for w in words) / n_words,
        "polysyllable_share": sum(s >= 3 for s in syllables) / n_words,
        "type_token_ratio": len({w.lower() for w in words}) / n_words,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
    }


def build_features(texts):
    return pd.DataFrame([text_features(t) for t in texts])


def build_cache(cfg):
    frames = [pd.read_csv(resolve(cfg["paths"][k]))[["id", "excerpt"]] for k in ("train", "test")]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    feats = build_features(df["excerpt"])
    feats.insert(0, "id", df["id"].values)
    path = resolve(cfg["features"]["cache"])
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(path, index=False)
    return feats


def attach_features(df, cfg, columns):
    path = resolve(cfg["features"]["cache"])
    cache = pd.read_csv(path) if path.exists() else build_cache(cfg)
    if not set(df["id"]).issubset(cache["id"]):
        cache = build_cache(cfg)
    missing = [c for c in columns if c not in cache.columns]
    if missing:
        raise ValueError(f"feature cache {path} lacks columns {missing}; rebuild with python -m src.features")
    out = df[["id", "excerpt"]].merge(cache[["id", *columns]], on="id", how="left", validate="one_to_one")
    assert (out["id"].values == df["id"].values).all()
    if out[columns].isna().any().any():
        raise ValueError("NaN in readability features")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/e2_ridgecv_features.yaml")
    feats = build_cache(load_config(parser.parse_args().config))
    print(f"feature cache: {len(feats)} texts, {feats.shape[1] - 1} features")
