"""Approach 2: k-nearest-neighbour grading on sentence embeddings (cosine similarity).

Inference only. Artifacts are produced by train_approach2.py:
  weights/approach2_knn.npz     emb[float16], y, qid, mu_qid, mu, mu_pool
                                [+ W, W_pool                when transform == "white"]
                                [+ M_P, M_b                 when a supervised metric is used]
  weights/approach2_config.json encoders, center, transform, pca_dims, alpha, metric,
                                k, temp, agg, classes

Pipeline per essay set: encode (one or more Sentence-BERT models, each L2-normalised, then
concatenated) -> centre, or PCA-whiten, fitted on that set's training answers -> optional
supervised LDA/NCA projection of that space -> cosine KNN -> weighted vote / mean / median.
"""
from pathlib import Path
import json
import numpy as np

HF_NAMES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "multiqa_minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "e5_small": "intfloat/e5-small-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}
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


def embed(encoders, names, texts):
    """One or several encoders; each is L2-normalised, then concatenated and re-normalised."""
    return l2(np.hstack([encode_texts(e, texts, n) for e, n in zip(encoders, names)]))


# ---------------------------------------------------------------- embedding-space transforms
def whiten_fit(E, n, alpha):
    """PCA-whitening fit on E (rows). Returns (mu, W): project with l2((X - mu) @ W)
    (keeps n components, each scaled by its std-dev ** -alpha)."""
    mu = E.mean(0)
    _, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
    s = S[:n] / np.sqrt(len(E) - 1)
    W = Vt[:n].T * (s + 1e-6) ** (-alpha)
    return mu.astype(np.float32), W.astype(np.float32)


def project(X, mu, W=None):
    """Centre (and whiten if W is given), then L2-normalise, so dot product == cosine."""
    Z = X - mu
    if W is not None:
        Z = Z @ W
    return l2(Z)


def fit_metric(Z, y, kind, dims):
    """Supervised linear projection of (whitened, L2-normalised) vectors Z using grade labels y.
    kind: "lda" (shrinkage LDA, <= n_grades-1 dims) or "nca" (neighbourhood components, `dims`).
    Returns (P, b) so the metric-space vector is l2(Z @ P + b). The map is affine, so it is read
    off sklearn's transform() rather than relying on solver-specific attributes."""
    if kind == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        est = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto",
                                         n_components=max(1, min(dims, len(set(y.tolist())) - 1)))
    else:
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        est = NeighborhoodComponentsAnalysis(n_components=dims, init="pca", max_iter=40, random_state=0)
    est.fit(Z, y)
    b = est.transform(np.zeros((1, Z.shape[1])))[0]
    P = est.transform(np.eye(Z.shape[1])) - b
    return P.astype(np.float32), b.astype(np.float32)


# ---------------------------------------------------------------- KNN core
def weights_for(sims, temp):
    """Softmax over cosine similarity; a huge temp (1e6) is plain uniform weighting."""
    return np.exp((sims - sims.max(axis=1, keepdims=True)) / temp)


def aggregate(neigh_labels, w, allowed, agg):
    """neigh_labels, w: (n, K). Returns int array of predictions in `allowed`.
    agg: "vote" (weighted majority), "mean" (weighted mean, snapped to a grade),
    "median" (weighted median grade)."""
    allowed = np.asarray(sorted(allowed))
    if agg == "mean":
        m = (w * neigh_labels).sum(1) / w.sum(1)
        return allowed[np.abs(m[:, None] - allowed[None, :]).argmin(1)]
    votes = np.stack([(w * (neigh_labels == g)).sum(1) for g in allowed], axis=1)
    if agg == "median":
        cum = np.cumsum(votes, axis=1)
        return allowed[(cum >= 0.5 * cum[:, -1:]).argmax(1)]
    return allowed[votes.argmax(1)]  # ties -> lower grade


def knn_from_sims(sims, y, k, temp, agg, allowed):
    """sims: (n_query, n_train) cosine similarities."""
    k = min(k, sims.shape[1])
    idx = np.argsort(-sims, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    return aggregate(y[idx], weights_for(s, temp), allowed, agg)


# ---------------------------------------------------------------- model
class KNNEmbeddingModel:
    def __init__(self, weights_dir, assets):
        wd = Path(weights_dir)
        self.cfg = json.loads((wd / "approach2_config.json").read_text())
        if self.cfg.get("tfidf"):
            raise ValueError("These weights were trained with a TF-IDF component, which this "
                             "embeddings-only code no longer supports. Retrain with train_approach2.py.")
        d = np.load(wd / "approach2_knn.npz", allow_pickle=False)
        emb = d["emb"].astype(np.float32)
        y = d["y"].astype(np.int64)
        qid = d["qid"].astype(str)
        qids = d["mu_qid"].astype(str)
        self.mu = {q: m.astype(np.float32) for q, m in zip(qids, d["mu"])}
        self.mu_pool = d["mu_pool"].astype(np.float32)
        if self.cfg.get("transform") == "white":  # per-question whitening projections
            self.W = {q: w.astype(np.float32) for q, w in zip(qids, d["W"])}
            self.W_pool = d["W_pool"].astype(np.float32)
        else:
            self.W, self.W_pool = {}, None
        # projected, L2-normalised training vectors per question (+ pooled fallback)
        self.index = {q: (project(emb[qid == q], self.mu[q], self.W.get(q)), y[qid == q])
                      for q in np.unique(qid)}
        self.pool = (project(emb, self.mu_pool, self.W_pool), y)
        # optional supervised metric (cfg["metric"] = {"kind", "weight", "dims"})
        self.metric = self.cfg.get("metric")
        if self.metric:
            P, b = dict(zip(qids, d["M_P"])), dict(zip(qids, d["M_b"]))
            self.mP = {q: P[q].astype(np.float32) for q in self.index}
            self.mb = {q: b[q].astype(np.float32) for q in self.index}
            self.mtrain = {q: l2(self.index[q][0] @ self.mP[q] + self.mb[q]) for q in self.index}
        self.names = self.cfg["encoders"]
        self.encoders = [get_encoder(n, assets) for n in self.names]

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
            Z = project(X[rows], mu, W)
            sims = Z @ E.T  # cosine similarity in the embedding space
            if self.metric and q in self.mtrain:  # (unseen questions skip the metric component)
                m = float(self.metric["weight"])
                sims = (1 - m) * sims + m * (l2(Z @ self.mP[q] + self.mb[q]) @ self.mtrain[q].T)
            preds[rows] = knn_from_sims(sims, y, self.cfg["k"], self.cfg["temp"], self.cfg["agg"],
                                        self._allowed(questions, q))
        return [{"id": r["id"], "score": int(p)} for r, p in zip(test_rows, preds)]


def load_model(weights_dir, config, assets):
    wd = weights_dir if weights_dir else Path(__file__).resolve().parent / "weights"
    return KNNEmbeddingModel(wd, assets)