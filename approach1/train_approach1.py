"""Train Approach 1 (KNN with TF-IDF) and save weights/approach1.joblib.

Usage: python train_approach1.py training_dataset_updated.csv [--out weights/approach1.joblib]

Final configuration (5-fold CV on training_dataset_updated.csv, per-question QWK):
  - text: "<br>" tags replaced by spaces, lowercased
  - two TF-IDF views of each answer, fitted on all questions (sublinear TF, min_df=2, L2 norm):
      (a) character 3-6 grams within word boundaries (char_wb)  - robust to typos / word forms
      (b) Porter-stemmed word unigrams + bigrams                 - exact terminology
    concatenated as [char, 0.5 * word] and L2-normalised -> dot product = cosine similarity
  - no stop-word removal (evaluated; it hurts)
  - neighbours are searched within the same question
  - neighbour score = cosine * exp(-0.5 * |log(1+len_a) - log(1+len_b)|)   (length feature)
  - K = max(1, round(sqrt(n) * f)), n = graded answers for the question,
      f = 0.7 for small questions (n < 100), 0.5 for large ones
  - weighted vote with weights score^5 (normalised), restricted to the allowed grades;
      ordinal smoothing: each vote also counts for nearby grades with a Gaussian kernel,
      sigma = 0.5 * (max grade - min grade), so 4.5 and 5.0 votes support each other;
      ties -> lower grade
  - low-similarity rule: best neighbour score < 0.15 -> predict the lowest allowed grade
"""
import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BR = re.compile(r"<br\s*/?>", re.I)
TOK = re.compile(r"(?u)\b\w\w+\b")
CONFIG = dict(char_ngrams=(3, 6), word_ngrams=(1, 2), min_df=2, word_weight=0.5,
              power=5, len_weight=0.5, low_sim=0.15, vote_smooth=0.5,
              k_factor=0.5, k_factor_small=0.7, small_n=100)

_stemmer = PorterStemmer()
_cache = {}


def clean(t):
    return BR.sub(" ", str(t))


def stem_text(t):
    out = []
    for w in TOK.findall(t.lower()):
        s = _cache.get(w)
        if s is None:
            s = _cache[w] = _stemmer.stem(w)
        out.append(s)
    return " ".join(out)


def build(df):
    texts = df["EssayText"].map(clean).values
    stemmed = [stem_text(t) for t in texts]
    c = CONFIG
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=c["char_ngrams"], min_df=c["min_df"],
                         sublinear_tf=True, lowercase=True, dtype=np.float32)
    vw = TfidfVectorizer(analyzer="word", ngram_range=c["word_ngrams"], min_df=c["min_df"],
                         sublinear_tf=True, lowercase=True, dtype=np.float32)
    Xc = vc.fit_transform(texts)
    Xw = vw.fit_transform(stemmed)
    X = normalize(hstack([Xc, Xw * c["word_weight"]]).tocsr()).astype(np.float32)

    lens = np.log1p(np.array([len(s) for s in texts], dtype=np.float32))
    q = df["EssaySet"].astype(str).values
    y = df["Score1"].astype(float).values
    qmax = df.groupby("EssaySet")["Score1"].transform("max").astype(float).values
    ynorm = np.where(qmax > 0, y / np.maximum(qmax, 1e-9), 0.0)   # for the unseen-question fallback
    order = np.argsort(q, kind="stable")
    X, q, y, ynorm, lens = X[order], q[order], y[order], ynorm[order], lens[order]
    groups = {}
    for name in pd.unique(q):
        idx = np.where(q == name)[0]
        groups[str(name)] = (int(idx[0]), int(idx[-1]) + 1)
    return {
        "config": c,
        "vocab_c": {str(t): int(i) for t, i in vc.vocabulary_.items()},
        "idf_c": vc.idf_.astype(np.float32),
        "vocab_w": {str(t): int(i) for t, i in vw.vocabulary_.items()},
        "idf_w": vw.idf_.astype(np.float32),
        "data": X.data.astype(np.float32),
        "indices": X.indices.astype(np.int32),
        "indptr": X.indptr.astype(np.int64),
        "shape": np.array(X.shape),
        "y": y.astype(np.float32),
        "ynorm": ynorm.astype(np.float32),
        "loglen": lens.astype(np.float32),
        "groups": groups,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="weights/approach1.joblib")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    art = build(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, a.out, compress=3)
    print(f"saved {a.out}  ({len(art['groups'])} questions, "
          f"{art['shape'][0]} answers, {art['shape'][1]} features)")