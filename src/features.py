import re

import pandas as pd
import textstat

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
