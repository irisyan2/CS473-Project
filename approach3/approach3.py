"""Approach 3: Open approach -- per-question stacked ensemble.

Base models (all trained per question, out-of-fold predictions are stacked):
  * Ridge on word TF-IDF (1-2 grams)
  * Ridge on char TF-IDF (2-5 grams, robust to the many misspellings)
  * Logistic regression on word+char TF-IDF (expected grade)
  * For each sentence-embedding model (staff encoders): cosine-KNN regression
    and RBF kernel ridge regression
  * Logistic regression on the concatenated embeddings
A non-negative linear blend combines them, and the grade cut-points are tuned
directly to maximise Quadratic Weighted Kappa on out-of-fold predictions.

Nothing is trained at grading time: load_model() only loads the saved bundle
(weights/approach3_bundle.joblib) and the staff-supplied frozen encoders.
This file is also imported by training_approach3.py so that feature code is
identical in training and inference. No custom classes are pickled.
"""
from pathlib import Path
import re
import time

import numpy as np
import joblib

BUNDLE_NAME = "approach3_bundle.joblib"
DEFAULT_HF = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
    "multiqa_minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "e5_small": "intfloat/e5-small-v2",
}
KNN_K = 15
KNN_TEMP = 10.0
KRR_GAMMA = 2.0
KRR_LAMBDA = 0.5
TIME_BUDGET_SECONDS = 330  # CPU-only runs are limited to 10 minutes in total


# --------------------------------------------------------------------------
# Text + embedding helpers
# --------------------------------------------------------------------------
def clean_text(t):
    t = "" if t is None else str(t)
    t = t.lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)  # sooooo -> soo
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_encoder(path, local_only=True):
    import torch
    from sentence_transformers import SentenceTransformer
    try:
        torch.set_num_threads(4)
    except Exception:
        pass
    enc = SentenceTransformer(str(path), device="cpu", trust_remote_code=False,
                              local_files_only=local_only)
    enc.max_seq_length = 192
    return enc


def encode_texts(enc, texts, batch_size=64):
    import torch
    with torch.inference_mode():
        E = enc.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    return E.astype(np.float32)


# --------------------------------------------------------------------------
# QWK + thresholds
# --------------------------------------------------------------------------
def qwk_idx(a, b, k):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if k < 2 or len(a) == 0:
        return 0.0
    O = np.zeros((k, k))
    np.add.at(O, (a, b), 1)
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / float(k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    return float(1.0 - (w * O).sum() / den) if den > 0 else 0.0


def to_index(z, th):
    return np.searchsorted(np.asarray(th), z, side="left")


def optimize_thresholds(z, y_idx, grades, rounds=4, grid=61):
    g = np.asarray(grades, dtype=float)
    k = len(g)
    th = (g[:-1] + g[1:]) / 2.0
    gmin, gmax = g[0], g[-1]
    best = qwk_idx(to_index(z, th), y_idx, k)
    for _ in range(rounds):
        for j in range(k - 1):
            lo = th[j - 1] + 1e-3 if j > 0 else gmin
            hi = th[j + 1] - 1e-3 if j < k - 2 else gmax
            cands = np.linspace(lo, hi, grid)
            scores = []
            for c in cands:
                t2 = th.copy()
                t2[j] = c
                scores.append(qwk_idx(to_index(z, t2), y_idx, k))
            scores = np.array(scores)
            m = scores.max()
            plateau = cands[scores >= m - 1e-9]
            th[j] = float(np.median(plateau))  # centre of plateau = less overfit
            best = m
    return th, best


# --------------------------------------------------------------------------
# Base models
# --------------------------------------------------------------------------
def _expected(clf, X):
    return clf.predict_proba(X) @ clf.classes_.astype(float)


def _knn_pred(S, y):
    k = min(KNN_K, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, axis=1)
    w = np.exp(KNN_TEMP * (s - s.max(axis=1, keepdims=True)))
    w /= w.sum(axis=1, keepdims=True)
    return (w * y[idx]).sum(axis=1)


def _rbf(S):
    return np.exp(-KRR_GAMMA * (2.0 - 2.0 * S))


def fit_base(texts, embs, y, emb_names):
    """texts: list[str]; embs: {name: (n,d) float32}; y: float array of grades."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, LogisticRegression
    from scipy.sparse import hstack

    y = np.asarray(y, dtype=float)
    yi = np.round(y, 6)
    st = {"y": y.astype(np.float32)}
    wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                         max_features=15000, strip_accents="unicode")
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                         min_df=2, max_features=30000)
    Xw = wv.fit_transform(texts)
    Xc = cv.fit_transform(texts)
    wv.stop_words_ = None
    cv.stop_words_ = None
    Xwc = hstack([Xw, Xc]).tocsr()
    st["wv"], st["cv"] = wv, cv
    st["ridge_word"] = Ridge(alpha=1.0).fit(Xw, y)
    st["ridge_char"] = Ridge(alpha=1.0).fit(Xc, y)
    st["logit_wc"] = LogisticRegression(C=10.0, max_iter=300).fit(Xwc, yi)
    for n in emb_names:
        E = embs[n].astype(np.float32)
        S = E @ E.T
        K = _rbf(S).astype(np.float64)
        ym = float(y.mean())
        alpha = np.linalg.solve(K + KRR_LAMBDA * np.eye(len(y)), y - ym)
        st["E_" + n] = E
        st["krr_" + n] = (alpha.astype(np.float32), ym)
    if emb_names:
        cat = np.hstack([embs[n] for n in emb_names])
        st["logit_emb"] = LogisticRegression(C=3.0, max_iter=300).fit(cat, yi)
    return st


def predict_base(st, texts, embs, emb_names, use_emb=True):
    from scipy.sparse import hstack
    Xw = st["wv"].transform(texts)
    Xc = st["cv"].transform(texts)
    cols = [st["ridge_word"].predict(Xw),
            st["ridge_char"].predict(Xc),
            _expected(st["logit_wc"], hstack([Xw, Xc]).tocsr())]
    if use_emb and emb_names:
        y = st["y"].astype(np.float64)
        for n in emb_names:
            E = embs[n].astype(np.float32)
            Et = st["E_" + n].astype(np.float32)
            S = E @ Et.T
            cols.append(_knn_pred(S, y))
            alpha, ym = st["krr_" + n]
            cols.append(_rbf(S) @ alpha.astype(np.float64) + ym)
        cat = np.hstack([embs[n] for n in emb_names])
        cols.append(_expected(st["logit_emb"], cat))
    return np.column_stack(cols)


def compact_state(st):
    for key in list(st.keys()):
        if key.startswith("E_"):
            st[key] = st[key].astype(np.float16)
    return st


# --------------------------------------------------------------------------
# Blend (non-negative stacking + QWK-optimised cut-points)
# --------------------------------------------------------------------------
def fit_blend(P, y, grades):
    from sklearn.linear_model import LinearRegression
    g = np.asarray(grades, dtype=float)
    lr = LinearRegression(positive=True).fit(P, y)
    z = np.clip(lr.predict(P), g[0], g[-1])
    y_idx = np.searchsorted(g, y)
    th, best = optimize_thresholds(z, y_idx, g)
    return {"coef": lr.coef_.astype(np.float64), "intercept": float(lr.intercept_),
            "th": th, "train_qwk": best}


def apply_blend(P, blend, grades):
    g = np.asarray(grades, dtype=float)
    z = np.clip(P @ blend["coef"] + blend["intercept"], g[0], g[-1])
    return z, to_index(z, blend["th"])


# --------------------------------------------------------------------------
# Mapping predicted class index -> an allowed grade of the evaluated question
# --------------------------------------------------------------------------
def _py(v):
    f = float(v)
    return int(f) if f.is_integer() else f


def map_index(idx, k, allowed):
    allowed = list(allowed)
    if len(allowed) == k:
        return [_py(allowed[int(i)]) for i in idx]
    out = []
    for i in idx:
        frac = i / float(max(k - 1, 1))
        out.append(_py(allowed[int(round(frac * (len(allowed) - 1)))]))
    return out


def map_normalised(frac, allowed):
    allowed = list(allowed)
    frac = np.clip(frac, 0.0, 1.0)
    return [_py(allowed[int(round(f * (len(allowed) - 1)))]) for f in frac]


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
class Approach3Model:
    def __init__(self, weights_dir, config, assets):
        self.t0 = time.time()
        wd = Path(weights_dir) if weights_dir else Path(__file__).resolve().parent / "weights"
        path = wd / BUNDLE_NAME
        if not path.exists():
            path = Path(__file__).resolve().parent / "weights" / BUNDLE_NAME
        self.bundle = joblib.load(path)
        self.assets = assets
        self.emb_names = list(self.bundle.get("emb_names", []))

    def _embed(self, texts):
        if not self.emb_names:
            return None
        embs = {}
        try:
            for i, n in enumerate(self.emb_names):
                if i > 0 and time.time() - self.t0 > TIME_BUDGET_SECONDS:
                    return None
                p = self.assets["models"][n]["path"]
                enc = load_encoder(p, local_only=True)
                embs[n] = encode_texts(enc, texts)
                del enc
            return embs
        except Exception:
            return None

    def predict(self, test_rows, questions):
        rows = list(test_rows)
        texts = [clean_text(r.get("text")) for r in rows]
        embs = self._embed(texts)
        use_emb = embs is not None
        per_q = self.bundle["per_q"]
        glob = self.bundle.get("global")
        by_q = {}
        for i, r in enumerate(rows):
            by_q.setdefault(str(r["question"]), []).append(i)

        preds = [None] * len(rows)
        for q, ids in by_q.items():
            qinfo = (questions or {}).get(q) or (questions or {}).get(
                rows[ids[0]]["question"]) or {}
            entry = per_q.get(q)
            allowed = qinfo.get("allowed_grades") or (entry["grades"] if entry else [0, 1, 2, 3])
            allowed = sorted(allowed)
            sub_t = [texts[i] for i in ids]
            sub_e = {n: embs[n][ids] for n in embs} if use_emb else None
            try:
                if entry is not None and "const" in entry:
                    out = map_normalised(np.full(len(ids), entry["const_frac"]), allowed)
                elif entry is not None:
                    g = entry["grades"]
                    P = predict_base(entry["state"], sub_t, sub_e, self.emb_names, use_emb)
                    blend = entry["blend_full"] if use_emb else entry["blend_lex"]
                    _, idx = apply_blend(P, blend, g)
                    out = map_index(idx, len(g), allowed)
                elif glob is not None and use_emb:
                    cat = np.hstack([sub_e[n] for n in self.emb_names])
                    frac = glob["ridge"].predict(cat)
                    out = map_normalised(frac, allowed)
                else:
                    out = [_py(allowed[len(allowed) // 2])] * len(ids)
            except Exception:
                out = [_py(allowed[len(allowed) // 2])] * len(ids)
            for i, s in zip(ids, out):
                preds[i] = s
        return [{"id": r["id"], "score": preds[i]} for i, r in enumerate(rows)]


def load_model(weights_dir, config, assets):
    return Approach3Model(weights_dir, config, assets)