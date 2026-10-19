"""Approach 1: KNN with TF-IDF (cosine), per-question index, pooled fallback.

Config (chosen by 5-fold CV on training_dataset.csv, Score1 target):
Porter stemming, unigrams, sublinear TF, no stop-word removal, K=21,
cosine-similarity-weighted vote. No fitting happens here; artifacts are loaded.
"""
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

_stemmer = PorterStemmer()
_tok = re.compile(r"(?u)\b\w\w+\b")


@lru_cache(maxsize=None)
def _stem_word(w):
    return _stemmer.stem(w)


def _stem_text(t):
    return " ".join(_stem_word(w) for w in _tok.findall(str(t).lower()))


class _Index:
    def __init__(self, d):
        self.X = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
        self.y = d["y"].astype(int)
        self.idf = d["idf"]
        self.cv = CountVectorizer(vocabulary=d["vocab"])

    def vectorize(self, stemmed_texts):
        C = self.cv.transform(stemmed_texts).astype(np.float32)
        C.data = 1.0 + np.log(C.data)               # sublinear tf
        C = C.multiply(self.idf).tocsr()
        return normalize(C)                          # l2 -> dot product == cosine


class TfidfKnnModel:
    def __init__(self, weights_dir, config, assets):
        art = joblib.load(Path(weights_dir) / "approach1.joblib")
        self.K = int(art["K"])
        self.per_q = {q: _Index(d) for q, d in art["per_q"].items()}
        self.pooled = _Index(art["pooled"])

    @staticmethod
    def _snap(score, allowed):
        return int(min(allowed, key=lambda g: (abs(g - score), g)))

    def _predict_group(self, index, texts, allowed):
        stemmed = [_stem_text(t) for t in texts]
        Xv = index.vectorize(stemmed)
        S = (Xv @ index.X.T).toarray()
        k = min(self.K, S.shape[1])
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, top, axis=1)
        labs = index.y[top]
        out = []
        for i in range(len(texts)):
            if sims[i].sum() <= 0:                   # no vocabulary overlap -> unweighted vote
                w = np.ones(k)
            else:
                w = sims[i]
            votes = {g: float(w[labs[i] == g].sum()) for g in allowed}
            best = max(allowed, key=lambda g: (votes[g], -g))
            out.append(self._snap(best, allowed))
        return out

    def predict(self, test_rows, questions):
        by_q = {}
        for pos, r in enumerate(test_rows):
            by_q.setdefault(r["question"], []).append(pos)
        result = [None] * len(test_rows)
        for q, positions in by_q.items():
            allowed = [int(g) for g in questions[q]["allowed_grades"]]
            index = self.per_q.get(str(q), self.pooled)   # unseen question -> pooled fallback
            texts = [test_rows[p]["text"] for p in positions]
            for p, s in zip(positions, self._predict_group(index, texts, allowed)):
                result[p] = {"id": test_rows[p]["id"], "score": int(s)}
        return result


def load_model(weights_dir, config, assets):
    return TfidfKnnModel(weights_dir, config, assets)