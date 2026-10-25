"""Approach 1: KNN with TF-IDF (cosine similarity).

Each answer = [character 3-6 gram TF-IDF, 0.5 * stemmed word 1-2 gram TF-IDF], L2-normalised.
Neighbours are searched within the question. Neighbour score = cosine * exp(-0.5 * |log-length diff|).
K = max(1, round(sqrt(n) * f)) with f = 0.7 for small questions (n < 100) and 0.5 otherwise.
Score^5-weighted vote over the allowed grades, smoothed across nearby grades with a Gaussian kernel
(sigma = 0.5 * grade range) so the vote respects the ordinal scale (ties -> lower grade); if the best neighbour score is
below 0.15 the answer is unlike anything graded -> lowest allowed grade.
Unseen questions: neighbours from all questions, grades rescaled to 0-1, weighted mean mapped onto
the question's allowed grades. Grades may be fractional (e.g. 2.5); outputs are always allowed grades.
No fitting happens here; artifacts are loaded.
"""
import re
from pathlib import Path

import joblib
import numpy as np
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

_BR = re.compile(r"<br\s*/?>", re.I)
_TOK = re.compile(r"(?u)\b\w\w+\b")
_stemmer = PorterStemmer()
_cache = {}


def _clean(t):
    return _BR.sub(" ", str(t))


def _stem_text(t):
    out = []
    for w in _TOK.findall(t.lower()):
        s = _cache.get(w)
        if s is None:
            s = _cache[w] = _stemmer.stem(w)
        out.append(s)
    return " ".join(out)


def _as_grade(g):
    g = float(g)
    return int(g) if g.is_integer() else g


def _tfidf(cv, idf, texts):
    C = cv.transform(texts).astype(np.float32)
    C.data = 1.0 + np.log(C.data)                           # sublinear tf
    return normalize(C.multiply(idf).tocsr())


class TfidfKnnModel:
    def __init__(self, weights_dir, config=None, assets=None):
        a = joblib.load(Path(weights_dir) / "approach1.joblib")
        c = a["config"]
        self.c = c
        self.cv_c = CountVectorizer(analyzer="char_wb", ngram_range=tuple(c["char_ngrams"]),
                                    lowercase=True, vocabulary=a["vocab_c"])
        self.cv_w = CountVectorizer(analyzer="word", ngram_range=tuple(c["word_ngrams"]),
                                    lowercase=True, vocabulary=a["vocab_w"])
        self.idf_c = np.asarray(a["idf_c"], dtype=np.float32)
        self.idf_w = np.asarray(a["idf_w"], dtype=np.float32)
        self.X = csr_matrix((a["data"], a["indices"], a["indptr"]), shape=tuple(a["shape"]))
        self.y = np.asarray(a["y"], dtype=float)
        self.ynorm = np.asarray(a["ynorm"], dtype=float)
        self.loglen = np.asarray(a["loglen"], dtype=float)
        self.groups = a["groups"]

    def _vectorize(self, texts):
        texts = [_clean(t) for t in texts]
        self._q_loglen = np.log1p(np.array([len(t) for t in texts], dtype=float))
        Xc = _tfidf(self.cv_c, self.idf_c, texts)
        Xw = _tfidf(self.cv_w, self.idf_w, [_stem_text(t) for t in texts])
        return normalize(hstack([Xc, Xw * self.c["word_weight"]]).tocsr())   # dot product = cosine

    def _k(self, n):
        c = self.c
        f = c["k_factor_small"] if n < c["small_n"] else c["k_factor"]
        return min(n, max(1, int(round(np.sqrt(n) * f))))

    def _topk(self, Xq, lo, hi):
        S = (Xq @ self.X[lo:hi].T).toarray()
        S = S * np.exp(-self.c["len_weight"] * np.abs(self._q_loglen[:, None] - self.loglen[None, lo:hi]))
        k = self._k(hi - lo)
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.clip(np.take_along_axis(S, top, axis=1), 0, None)
        best = sims.max(1)
        w = sims ** self.c["power"]
        tot = w.sum(1, keepdims=True)
        w = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 1.0)   # normalise; no overlap -> plain vote
        return top + lo, w, best

    def _predict_known(self, texts, lo, hi, allowed):
        top, w, best = self._topk(self._vectorize(texts), lo, hi)
        labs = self.y[top]
        A = np.asarray(allowed, dtype=float)
        # neighbour grades outside the allowed list are snapped to the nearest allowed grade
        snapped = A[np.abs(labs[..., None] - A).argmin(-1)]
        votes = np.stack([(w * (snapped == g)).sum(1) for g in A], 1)
        sig = self.c.get("vote_smooth", 0.0) * (A.max() - A.min())
        if sig > 0:                               # ordinal smoothing: nearby grades share votes
            votes = votes @ np.exp(-(A[:, None] - A[None, :]) ** 2 / (2 * sig ** 2))
        pick = np.argmax(votes >= votes.max(1, keepdims=True) * (1 - 1e-6), axis=1)  # ties -> lower grade
        pred = A[pick]
        pred[best < self.c["low_sim"]] = A[0]     # unlike any graded answer -> lowest grade
        return pred

    def _predict_unseen(self, texts, allowed):
        top, w, _ = self._topk(self._vectorize(texts), 0, self.X.shape[0])
        A = np.asarray(allowed, dtype=float)
        lo, hi = A.min(), A.max()
        v = lo + (w * self.ynorm[top]).sum(1) * (hi - lo)
        return A[np.argmin(np.abs(v[:, None] - A) + 1e-9 * A, axis=1)]

    def predict(self, test_rows, questions):
        by_q = {}
        for pos, r in enumerate(test_rows):
            by_q.setdefault(r["question"], []).append(pos)
        result = [None] * len(test_rows)
        for q, positions in by_q.items():
            qinfo = questions.get(q, questions.get(str(q)))
            allowed = sorted(float(g) for g in qinfo["allowed_grades"])
            texts = [test_rows[p]["text"] for p in positions]
            key = str(q)
            if key in self.groups:
                lo, hi = self.groups[key]
                preds = self._predict_known(texts, lo, hi, allowed)
            else:
                preds = self._predict_unseen(texts, allowed)
            for p, s in zip(positions, preds):
                result[p] = {"id": test_rows[p]["id"], "score": _as_grade(s)}
        return result


def load_model(weights_dir, config=None, assets=None):
    return TfidfKnnModel(weights_dir, config, assets)