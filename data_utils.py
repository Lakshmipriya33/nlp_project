import re
from typing import Set

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


STEMMER = PorterStemmer()


def _ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def ensure_nltk_resources() -> None:
    _ensure_nltk_resource("corpora/stopwords", "stopwords")
    _ensure_nltk_resource("tokenizers/punkt", "punkt")


def get_stop_words() -> Set[str]:
    ensure_nltk_resources()
    return set(stopwords.words("english"))


def add_meta_features(df: pd.DataFrame, stop_words: Set[str]) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out["text"].fillna("").astype(str)
    out["word_count"] = out["text"].apply(lambda x: len(x.split()))
    out["unique_word_count"] = out["text"].apply(lambda x: len(set(x.split())))
    out["stop_word_count"] = out["text"].apply(
        lambda x: len([w for w in x.lower().split() if w in stop_words])
    )
    out["url_count"] = out["text"].apply(lambda x: len(re.findall(r"https?://\\S+", x)))
    out["mean_word_length"] = out["text"].apply(
        lambda x: float(np.mean([len(w) for w in x.split()])) if x.split() else 0.0
    )
    return out


def preprocess_text(text: str, stop_words: Set[str]) -> str:
    text = str(text)
    text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.lower() not in stop_words]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)
