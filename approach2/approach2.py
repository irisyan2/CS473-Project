"""Approach 2: KNN with sentence embeddings (cosine similarity), with per-question centering.

Inference only. Artifacts are produced by train_approach2.py:
  weights/approach2_knn.npz     (emb[float16], y, qid, mu_qid, mu, mu_pool
                                 [+ W, W_pool when transform == "white"])
  weights/approach2_config.json (encoders, center, k, temp, agg, classes)
"""
from pathlib import Path
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HF_NAMES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "multiqa_minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "e5_small": "intfloat/e5-small-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}
# TF-IDF settings for the optional hybrid similarity (cfg["tfidf"] = {"kind", "lam"});
# must match experiments2_approach2.py
TFIDF_KW = {"word": dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            "char": dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)}
PREFIX = {"e5_small": "query: "}  # e5 expects a prefix; others need none


# ---------------------------------------------------------------- encoder
def get_encoder(name, assets=None):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    path = None
    if assets:
        path = (assets.get("models") or {}).get(name, {}).get("path")
    if path:  # grader / local assets: offline only
        return SentenceTransformer(str(path), device="cpu",
                                   trust_remote_code=False, local_files_only=True)
    if assets:
        raise KeyError(f"assets.json has no models['{name}']['path']; "
                       f"available: {list((assets.get('models') or {}).keys())}")
    # no assets given at all: download from the hub by name (local training only)
    return SentenceTransformer(HF_NAMES[name], device="cpu", trust_remote_code=False)


def encode_texts(encoder, texts, name):
    texts = [PREFIX.get(name, "") + (t if isinstance(t, str) else "") for t in texts]
    emb = encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def whiten_fit(E, n, alpha):
    """PCA-whitening fit on E (rows). Returns (mu, W): project with l2((X - mu) @ W).
    Same math as experiments_approach2.fit_tf (n dims, power alpha on the std-devs)."""
    mu = E.mean(0)
    _, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    s = S[:n] / np.sqrt(len(E) - 1)
    W = Vt[:n].T * (s + 1e-6) ** (-alpha)
    return mu.astype(np.float32), W.astype(np.float32)


def embed(encoders, names, texts):
    """One or several encoders; each is L2-normalised, then concatenated and re-normalised."""
    return l2(np.hstack([encode_texts(e, texts, n) for e, n in zip(encoders, names)]))


# ---------------------------------------------------------------- KNN core
def weights_for(sims, temp):
    """Softmax over cosine similarity; a huge temp (1e6) is plain uniform weighting."""
    return np.exp((sims - sims.max(axis=1, keepdims=True)) / temp)


def aggregate(neigh_labels, w, allowed, agg):
    """neigh_labels, w: (n, K). Returns int array of predictions in `allowed`."""
    allowed = np.asarray(sorted(allowed))
    if agg == "mean":
        m = (w * neigh_labels).sum(1) / w.sum(1)
        return allowed[np.abs(m[:, None] - allowed[None, :]).argmin(1)]
    votes = np.stack([(w * (neigh_labels == g)).sum(1) for g in allowed], axis=1)
    return allowed[votes.argmax(1)]  # ties -> lower grade


def knn_from_sims(sims, y, k, temp, agg, allowed):
    k = min(k, sims.shape[1])
    idx = np.argsort(-sims, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    return aggregate(y[idx], weights_for(s, temp), allowed, agg)


def knn_predict(Xq, E, y, k, temp, agg, allowed):
    return knn_from_sims(Xq @ E.T, y, k, temp, agg, allowed)


def clean_text(t):
    return t if isinstance(t, str) else ""


# ---------------------------------------------------------------- model
class KNNEmbeddingModel:
    def __init__(self, weights_dir, assets):
        wd = Path(weights_dir)
        self.cfg = json.loads((wd / "approach2_config.json").read_text())
        d = np.load(wd / "approach2_knn.npz", allow_pickle=False)
        emb = d["emb"].astype(np.float32)
        y = d["y"].astype(np.int64)
        qid = d["qid"].astype(str)
        self.mu = {q: m.astype(np.float32) for q, m in zip(d["mu_qid"].astype(str), d["mu"])}
        self.mu_pool = d["mu_pool"].astype(np.float32)
        # optional per-question whitening projections (cfg["transform"] == "white")
        if self.cfg.get("transform") == "white":
            self.W = {q: w.astype(np.float32) for q, w in zip(d["mu_qid"].astype(str), d["W"])}
            self.W_pool = d["W_pool"].astype(np.float32)
        else:
            self.W, self.W_pool = {}, None
        # projected (centred [+ whitened]), re-normalised training vectors per question
        # (+ pooled fallback)
        self.index = {q: (self._proj(emb[qid == q], self.mu[q], self.W.get(q)), y[qid == q])
                      for q in np.unique(qid)}
        self.pool = (self._proj(emb, self.mu_pool, self.W_pool), y)
        # optional TF-IDF hybrid: refit per-question vectorizers from the stored training texts
        self.tfidf = self.cfg.get("tfidf")
        self._tf = {}
        if self.tfidf:
            self._texts = np.array(json.loads(bytes(d["texts_json"]).decode("utf-8")), dtype=object)
            self._qid = qid
            for q in np.unique(qid):
                self._tf[q] = self._fit_tf(self._texts[qid == q])
        self.names = self.cfg["encoders"]
        self.encoders = [get_encoder(n, assets) for n in self.names]

    def _fit_tf(self, texts):
        v = TfidfVectorizer(**TFIDF_KW[self.tfidf["kind"]])
        return v, v.fit_transform(list(texts))

    @staticmethod
    def _proj(X, mu, W):
        Z = X - mu
        if W is not None:
            Z = Z @ W
        return l2(Z)

    def _allowed(self, questions, q):
        info = (questions or {}).get(q) or (questions or {}).get(str(q)) or {}
        g = info.get("allowed_grades")
        if g is None:
            g = self.cfg["classes"].get(str(q), sorted(set(self.pool[1].tolist())))
        return sorted(int(x) for x in g)

    def predict(self, test_rows, questions):
        if not test_rows:
            return []
        X = embed(self.encoders, self.names, [r["text"] for r in test_rows])
        qs = np.array([str(r["question"]) for r in test_rows])
        preds = np.zeros(len(test_rows), dtype=np.int64)
        for q in np.unique(qs):
            rows = np.where(qs == q)[0]
            if q in self.index:
                (E, y), mu, W = self.index[q], self.mu[q], self.W.get(q)
            else:  # unseen question -> pooled training set
                (E, y), mu, W = self.pool, self.mu_pool, self.W_pool
            sims = self._proj(X[rows], mu, W) @ E.T
            if self.tfidf:
                if q not in self._tf:  # unseen question -> pooled vectorizer (fit lazily, once)
                    if "__pool__" not in self._tf:
                        self._tf["__pool__"] = self._fit_tf(self._texts)
                    self._tf[q] = self._tf["__pool__"]
                v, Xtr = self._tf[q]
                S_tf = (v.transform([clean_text(test_rows[i]["text"]) for i in rows]) @ Xtr.T).toarray()
                lam = float(self.tfidf["lam"])
                sims = (1 - lam) * sims + lam * S_tf
            preds[rows] = knn_from_sims(sims, y, self.cfg["k"],
                                      self.cfg["temp"], self.cfg["agg"], self._allowed(questions, q))
        return [{"id": r["id"], "score": int(p)} for r, p in zip(test_rows, preds)]


def load_model(weights_dir, config, assets):
    wd = weights_dir if weights_dir else Path(__file__).resolve().parent / "weights"
    return KNNEmbeddingModel(wd, assets)