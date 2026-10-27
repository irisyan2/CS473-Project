"""Approach 2: KNN with sentence embeddings (cosine similarity) -- size-aware version.

Inference only; artifacts come from train_approach2.py:
  weights/approach2_knn.npz      emb[float16], y[float32], yfrac, qid, mu_qid, mu, mu_pool,
                                 [W_qid, W, M_P, M_b  for large questions]
                                 [cal_qid, cal_gamma, cal_ybar  when a group uses agg == "cal"]
  weights/approach2_config.json  encoders, large_n, groups{large,small}, fallback, classes

Per question the stored answers are projected (centred; large questions are also PCA-whitened and
optionally passed through a supervised NCA metric) and the K nearest by cosine are aggregated.
Large questions (>= large_n graded answers) and small ones use different settings.
Grades may be half-points; predictions are snapped to the nearest allowed grade, exact ties upward.

agg == "cal" (calibrated mean): the weighted mean of the neighbours' grades is stretched away from
the question's mean grade by a factor gamma >= 1 before snapping. Neighbour averages regress toward
the middle, which quadratic-weighted kappa penalises; gamma is chosen per question so the spread of
leave-one-out predictions on the stored training answers equals the spread of the real grades.

groups.small.xq = {"weight": b} (only with agg == "cal"): the calibrated within-question prediction is
blended, with weight b, with a prediction from the nearest answers to OTHER questions (their grades
as a fraction of each question's top grade, scaled to this question's top grade). A tiny question has
only ~20 labelled answers; this borrows a little generic "what a good answer looks like" signal.
"""
from pathlib import Path
import json
import re
import numpy as np

HF_NAMES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "multiqa_minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "e5_small": "intfloat/e5-small-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}
PREFIX = {"e5_small": "query: "}  # e5 expects a prefix; the others need none
_BR = re.compile(r"<br\s*/?>", re.I)


# ---------------------------------------------------------------- encoder
def clean(t):
    """Some answers contain HTML line breaks; replace them with spaces."""
    return _BR.sub(" ", t if isinstance(t, str) else "")


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
    texts = [PREFIX.get(name, "") + clean(t) for t in texts]
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
    """PCA-whitening fit on E (rows). Returns (mu, W): project with l2((X - mu) @ W).
    Keeps min(n, rows-1) components, each scaled by its std-dev ** -alpha."""
    n = max(1, min(n, len(E) - 1))
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
    Returns (P, b) so the metric-space vector is l2(Z @ P + b)."""
    y = np.round(np.asarray(y, float) * 2).astype(int)  # class ids (half-point grades -> ints)
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
def snap(m, allowed):
    """Nearest allowed grade; an exact midpoint tie goes to the HIGHER grade."""
    g = np.asarray(sorted(float(x) for x in allowed))
    d = np.abs(np.asarray(m, float)[:, None] - g[None, ::-1])  # reversed grid: argmin picks the higher
    return g[::-1][d.argmin(1)]


def weights_for(sims, temp):
    """Softmax over cosine similarity; a huge temp (1e6) is plain uniform weighting."""
    return np.exp((sims - sims.max(axis=1, keepdims=True)) / temp)


def aggregate(neigh_labels, w, allowed, agg):
    """neigh_labels, w: (n, K). Returns predictions (floats) on the `allowed` grid.
    agg: "mean" (weighted mean), "median" (weighted median), "vote" (weighted majority)."""
    allowed = np.asarray(sorted(float(x) for x in allowed))
    lab = np.asarray(neigh_labels, float)
    if agg == "cal":
        raise ValueError('agg "cal" needs per-question statistics: use knn_from_sims(..., cal=...)')
    if agg == "mean":
        return snap((w * lab).sum(1) / w.sum(1), allowed)
    if agg == "median":
        o = np.argsort(lab, axis=1)
        ls, ws = np.take_along_axis(lab, o, 1), np.take_along_axis(w, o, 1)
        cum = np.cumsum(ws, axis=1)
        pos = (cum >= 0.5 * cum[:, -1:]).argmax(1)
        return snap(ls[np.arange(len(ls)), pos], allowed)
    votes = np.stack([(w * np.isclose(lab, g)).sum(1) for g in allowed], axis=1)
    return allowed[votes.argmax(1)]  # ties -> lower grade


GAMMA_CAP = 4.0  # largest stretch factor for agg == "cal"


def calibrated_cont(lab, w, gamma, ybar):
    """Weighted mean of neighbour grades stretched away from `ybar` by `gamma` (not yet snapped)."""
    m = (w * np.asarray(lab, float)).sum(1) / w.sum(1)
    return ybar + gamma * (m - ybar)


def calibrated(lab, w, gamma, ybar, allowed):
    """Weighted mean of neighbour grades, stretched away from `ybar` by `gamma`, then snapped."""
    return snap(calibrated_cont(lab, w, gamma, ybar), allowed)


def knn_mean(sims, vals, k, temp):
    """Similarity-weighted mean of `vals` over the k most similar rows (continuous)."""
    k = min(k, sims.shape[1])
    idx = np.argsort(-sims, axis=1)[:, :k]
    w = weights_for(np.take_along_axis(sims, idx, axis=1), temp)
    return (w * vals[idx]).sum(1) / w.sum(1)


def train_sims(Z, Zm=None, weight=0.0):
    """Cosine similarities among a question's own training answers, built exactly like the
    query-time similarities (optionally blended with the NCA metric space)."""
    S = Z @ Z.T
    if Zm is not None and weight:
        S = (1 - weight) * S + weight * (Zm @ Zm.T)
    return S


def loo_topk(S, kmax):
    """Leave-one-out neighbours of each training answer (an answer is never its own neighbour)."""
    S = np.array(S, dtype=np.float64)
    np.fill_diagonal(S, -np.inf)
    kmax = max(0, min(kmax, S.shape[1] - 1))
    idx = np.argsort(-S, axis=1)[:, :kmax]
    return idx, np.take_along_axis(S, idx, axis=1)


def spread_from_topk(idx, s, y, k, temp, cap=GAMMA_CAP):
    """(gamma, ybar): stretch that makes the spread of leave-one-out weighted-mean predictions
    equal the spread of the real grades (gamma is clipped to [1, cap])."""
    y = np.asarray(y, float); ybar = float(y.mean()); k = min(k, idx.shape[1])
    if k < 1 or y.std() < 1e-9:
        return 1.0, ybar
    w = weights_for(s[:, :k], temp)
    m = (w * y[idx[:, :k]]).sum(1) / w.sum(1)
    return (1.0 if m.std() < 1e-9 else float(np.clip(y.std() / m.std(), 1.0, cap))), ybar


def fit_spread(S, y, k, temp):
    idx, s = loo_topk(S, k)
    return spread_from_topk(idx, s, y, k, temp)


def knn_from_sims(sims, y, k, temp, agg, allowed, cal=None):
    """sims: (n_query, n_train) cosine similarities. cal = (gamma, ybar) for agg == "cal"."""
    k = min(k, sims.shape[1])
    idx = np.argsort(-sims, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    w = weights_for(s, temp)
    if agg == "cal":
        gamma, ybar = cal if cal is not None else (1.0, float(np.mean(y)))
        return calibrated(y[idx], w, gamma, ybar, allowed)
    return aggregate(y[idx], w, allowed, agg)


def _py(p):
    p = float(p)
    return int(p) if p.is_integer() else p


# ---------------------------------------------------------------- model
class KNNEmbeddingModel:
    def __init__(self, weights_dir, assets):
        wd = Path(weights_dir)
        self.cfg = json.loads((wd / "approach2_config.json").read_text())
        d = np.load(wd / "approach2_knn.npz", allow_pickle=False)
        emb = d["emb"].astype(np.float32)
        y = d["y"].astype(np.float64)
        qid = d["qid"].astype(str)
        mu = {q: m.astype(np.float32) for q, m in zip(d["mu_qid"].astype(str), d["mu"])}
        W = {}
        if "W" in d.files:
            W = {q: w.astype(np.float32) for q, w in zip(d["W_qid"].astype(str), d["W"])}
        met = self.cfg["groups"]["large"].get("metric")
        MP, Mb = {}, {}
        if met and "M_P" in d.files:
            MP = {q: p.astype(np.float32) for q, p in zip(d["W_qid"].astype(str), d["M_P"])}
            Mb = {q: b.astype(np.float32) for q, b in zip(d["W_qid"].astype(str), d["M_b"])}
        cal = {}
        if "cal_gamma" in d.files:
            cal = {q: (float(g), float(b)) for q, g, b in
                   zip(d["cal_qid"].astype(str), d["cal_gamma"], d["cal_ybar"])}
        self.index = {}
        for q in np.unique(qid):
            m = qid == q
            Z = project(emb[m], mu[q], W.get(q))
            info = dict(Z=Z, y=y[m], mu=mu[q], W=W.get(q), cal=cal.get(q),
                        grp=self.cfg["groups"]["large" if m.sum() >= self.cfg["large_n"] else "small"])
            if q in MP:
                info["P"], info["b"] = MP[q], Mb[q]
                info["Zm"] = l2(Z @ MP[q] + Mb[q])
            self.index[q] = info
        self.mu_pool = d["mu_pool"].astype(np.float32)  # pooled fallback for unseen questions
        self.pool_Z = project(emb, self.mu_pool)
        self.pool_frac = d["yfrac"].astype(np.float64)
        self.pool_q = qid  # question of each pooled answer (to exclude the question's own answers)
        self.names = self.cfg["encoders"]
        self.encoders = [get_encoder(n, assets) for n in self.names]

    def _allowed(self, questions, q):
        info = (questions or {}).get(q) or (questions or {}).get(str(q)) or {}
        g = info.get("allowed_grades")
        if g is None:
            g = self.cfg["classes"].get(str(q), [0, 1, 2, 3])
        return sorted(float(x) for x in g)

    def predict(self, test_rows, questions):
        if not test_rows:
            return []
        X = embed(self.encoders, self.names, [r["text"] for r in test_rows])
        qs = np.array([str(r["question"]) for r in test_rows])
        preds = np.zeros(len(test_rows), dtype=np.float64)
        for q in np.unique(qs):
            rows = np.where(qs == q)[0]
            allowed = self._allowed(questions, q)
            if q in self.index:
                info = self.index[q]
                g = info["grp"]
                Zq = project(X[rows], info["mu"], info["W"])
                sims = Zq @ info["Z"].T
                if "Zm" in info and g.get("metric"):
                    m = float(g["metric"]["weight"])
                    sims = (1 - m) * sims + m * (l2(Zq @ info["P"] + info["b"]) @ info["Zm"].T)
                if g["agg"] == "cal" and g.get("xq"):
                    # calibrated within-question prediction, blended with other questions' neighbours
                    gam, yb = info["cal"] if info["cal"] is not None else (1.0, float(np.mean(info["y"])))
                    k_ = min(g["k"], sims.shape[1]); ix = np.argsort(-sims, axis=1)[:, :k_]
                    w_ = weights_for(np.take_along_axis(sims, ix, axis=1), g["temp"])
                    cont = calibrated_cont(info["y"][ix], w_, gam, yb)
                    fb = self.cfg["fallback"]; other = self.pool_q != q
                    cx = knn_mean(project(X[rows], self.mu_pool) @ self.pool_Z[other].T,
                                  self.pool_frac[other], fb["k"], fb["temp"]) * max(allowed)
                    b_ = float(g["xq"]["weight"])
                    preds[rows] = snap((1 - b_) * cont + b_ * cx, allowed)
                else:
                    preds[rows] = knn_from_sims(sims, info["y"], g["k"], g["temp"], g["agg"], allowed,
                                                cal=info["cal"])
            else:
                # unseen question: nearest answers over ALL questions, scores as a fraction of each
                # question's maximum, scaled to this question's top grade
                fb = self.cfg["fallback"]
                sims = project(X[rows], self.mu_pool) @ self.pool_Z.T
                frac = knn_from_sims(sims, self.pool_frac, fb["k"], fb["temp"], "mean",
                                     np.linspace(0, 1, 201))
                preds[rows] = snap(frac * max(allowed), allowed)
        return [{"id": r["id"], "score": _py(p)} for r, p in zip(test_rows, preds)]


def load_model(weights_dir, config, assets):
    wd = weights_dir if weights_dir else Path(__file__).resolve().parent / "weights"
    return KNNEmbeddingModel(wd, assets)