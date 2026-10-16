"""Approach 2: KNN with sentence embeddings (cosine similarity).

Inference only. Artifacts are produced by train_approach2.py:
  weights/approach2_knn.npz   (emb, y, qid)
  weights/approach2_config.json (encoder, k, weighting, agg, classes)
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
TEMP = 0.05  # softmax temperature for similarity weighting


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
    # local training fallback: download from the hub by name
    return SentenceTransformer(HF_NAMES[name], device="cpu", trust_remote_code=False)


def encode_texts(encoder, texts, name):
    texts = [PREFIX.get(name, "") + (t if isinstance(t, str) else "") for t in texts]
    emb = encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


# ---------------------------------------------------------------- KNN core
def weights_for(sims, weighting):
    if weighting == "softmax":
        return np.exp((sims - sims.max(axis=1, keepdims=True)) / TEMP)
    return np.ones_like(sims)


def aggregate(neigh_labels, w, allowed, agg):
    """neigh_labels, w: (n, K). Returns int array of predictions in `allowed`."""
    allowed = np.asarray(sorted(allowed))
    if agg == "mean":
        m = (w * neigh_labels).sum(1) / w.sum(1)
        return allowed[np.abs(m[:, None] - allowed[None, :]).argmin(1)]
    votes = np.stack([(w * (neigh_labels == g)).sum(1) for g in allowed], axis=1)
    return allowed[votes.argmax(1)]  # ties -> lower grade


def knn_predict(Xq, E, y, k, weighting, agg, allowed):
    sims = Xq @ E.T
    k = min(k, E.shape[0])
    idx = np.argsort(-sims, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    return aggregate(y[idx], weights_for(s, weighting), allowed, agg)


# ---------------------------------------------------------------- model
class KNNEmbeddingModel:
    def __init__(self, weights_dir, assets):
        wd = Path(weights_dir)
        self.cfg = json.loads((wd / "approach2_config.json").read_text())
        d = np.load(wd / "approach2_knn.npz", allow_pickle=False)
        self.emb, self.y = d["emb"].astype(np.float32), d["y"].astype(np.int64)
        qid = d["qid"].astype(str)
        self.index = {q: (self.emb[qid == q], self.y[qid == q]) for q in np.unique(qid)}
        self.encoder = get_encoder(self.cfg["encoder"], assets)

    def _allowed(self, questions, q):
        info = (questions or {}).get(q) or (questions or {}).get(str(q)) or {}
        g = info.get("allowed_grades")
        if g is None:
            g = self.cfg["classes"].get(str(q), sorted(set(self.y.tolist())))
        return sorted(int(x) for x in g)

    def predict(self, test_rows, questions):
        if not test_rows:
            return []
        X = encode_texts(self.encoder, [r["text"] for r in test_rows], self.cfg["encoder"])
        qs = np.array([str(r["question"]) for r in test_rows])
        preds = np.zeros(len(test_rows), dtype=np.int64)
        for q in np.unique(qs):
            rows = np.where(qs == q)[0]
            E, y = self.index.get(q, (self.emb, self.y))  # unseen question -> pooled
            allowed = self._allowed(questions, q)
            preds[rows] = knn_predict(X[rows], E, y, self.cfg["k"],
                                      self.cfg["weighting"], self.cfg["agg"], allowed)
        return [{"id": r["id"], "score": int(p)} for r, p in zip(test_rows, preds)]


def load_model(weights_dir, config, assets):
    wd = weights_dir if weights_dir else Path(__file__).resolve().parent / "weights"
    return KNNEmbeddingModel(wd, assets)