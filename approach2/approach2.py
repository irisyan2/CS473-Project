"""Approach 2: KNN with sentence embeddings (cosine similarity), with per-question centering.

Inference only. Artifacts are produced by train_approach2.py:
  weights/approach2_knn.npz     (emb[float16], y, qid, mu_qid, mu, mu_pool)
  weights/approach2_config.json (encoders, center, k, temp, agg, classes)
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


def knn_predict(Xq, E, y, k, temp, agg, allowed):
    sims = Xq @ E.T
    k = min(k, E.shape[0])
    idx = np.argsort(-sims, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    return aggregate(y[idx], weights_for(s, temp), allowed, agg)


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
        # centred, re-normalised training vectors per question (+ pooled fallback)
        self.index = {q: (l2(emb[qid == q] - self.mu[q]), y[qid == q]) for q in np.unique(qid)}
        self.pool = (l2(emb - self.mu_pool), y)
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
                (E, y), mu = self.index[q], self.mu[q]
            else:  # unseen question -> pooled training set
                (E, y), mu = self.pool, self.mu_pool
            preds[rows] = knn_predict(l2(X[rows] - mu), E, y, self.cfg["k"],
                                      self.cfg["temp"], self.cfg["agg"], self._allowed(questions, q))
        return [{"id": r["id"], "score": int(p)} for r, p in zip(test_rows, preds)]


def load_model(weights_dir, config, assets):
    wd = weights_dir if weights_dir else Path(__file__).resolve().parent / "weights"
    return KNNEmbeddingModel(wd, assets)