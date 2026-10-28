"""Train Approach 3: TF-IDF + Ridge + question-specific grade-centroid features."""

import argparse
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize


SMALL_N = 100
A_BIG = [0.3, 1.0, 3.0]
A_SMALL = [0.1, 0.3, 1.0, 3.0]
A_GLOBAL = [0.3, 1.0, 3.0]
M_GRID = [0, 10, 20, 40, 80]

# Change: include 1.6, which performed well in the previous experiment.
S_GRID = [1.0, 1.3, 1.5, 1.6, 1.7, 1.9]
B_GRID = [-0.05, 0.0, 0.05]

DEFAULT = dict(
    a_big=1.0,
    a_small=0.3,
    a_g=1.0,
    m=20,
    s=1.6,
    b=-0.05,
)

_QWK_WEIGHTS = {}


def clean(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    return str(t).lower().strip()


def load_data(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()

    sep = "\t" if header.count("\t") > header.count(",") else ","

    df = pd.read_csv(
        path,
        sep=sep,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )

    rename = {}

    for col in df.columns:
        name = col.strip().lower()

        if name == "id":
            rename[col] = "id"
        elif name in (
            "essayset", "question", "question number", "question_number"
        ):
            rename[col] = "question"
        elif name in ("essaytext", "text", "student submission"):
            rename[col] = "text"
        elif name in ("score1", "score", "manual score"):
            rename[col] = "score"

    df = df.rename(columns=rename)

    required = {"id", "question", "text", "score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["question", "score"]).copy()
    df["question"] = df["question"].astype(str)
    df["text"] = df["text"].map(clean)
    df["score"] = df["score"].astype(int)

    return df.reset_index(drop=True)


def get_score_info(df, questions_file):
    questions = {}

    if questions_file:
        with open(questions_file, encoding="utf-8-sig") as f:
            questions = json.load(f)

    ranges = {}
    allowed_by_q = {}

    for q, group in df.groupby("question"):
        info = questions.get(q, questions.get(str(q), {}))

        if "allowed_grades" in info and info["allowed_grades"]:
            allowed = sorted(set(float(v) for v in info["allowed_grades"]))
        else:
            allowed = [
                float(v)
                for v in range(
                    int(group.score.min()),
                    int(group.score.max()) + 1,
                )
            ]

        allowed_by_q[q] = allowed
        ranges[q] = (min(allowed), max(allowed))

    return ranges, allowed_by_q


def qwk(actual, predicted, lo, hi):
    k = int(round(hi - lo)) + 1

    if k < 2:
        return 0.0

    actual = np.clip(np.asarray(actual, dtype=int) - int(lo), 0, k - 1)
    predicted = np.clip(
        np.asarray(predicted, dtype=int) - int(lo), 0, k - 1
    )

    observed = np.bincount(
        actual * k + predicted,
        minlength=k * k,
    ).reshape(k, k).astype(float)

    total = observed.sum()

    if total <= 0:
        return 0.0

    expected = np.outer(
        observed.sum(axis=1),
        observed.sum(axis=0),
    ) / total

    if k not in _QWK_WEIGHTS:
        idx = np.arange(k)
        _QWK_WEIGHTS[k] = (
            (idx[:, None] - idx[None, :]) ** 2
        ) / (k - 1) ** 2

    denom = (_QWK_WEIGHTS[k] * expected).sum()

    if denom <= 0:
        return 0.0

    return float(
        1.0 - (_QWK_WEIGHTS[k] * observed).sum() / denom
    )


def make_folds(df, n_splits, repeats, seed, folds_csv=None):
    if folds_csv:
        fdf = pd.read_csv(folds_csv)

        id_col = next(
            (c for c in fdf.columns if c.lower() == "id"),
            fdf.columns[0],
        )
        fold_col = next(
            (c for c in fdf.columns if c.lower() == "fold"),
            None,
        )

        if fold_col is None:
            raise ValueError(
                "Folds CSV must contain columns named Id and fold."
            )

        mapping = dict(
            zip(fdf[id_col].astype(str), fdf[fold_col].astype(int))
        )

        mapped = df["id"].astype(str).map(mapping)

        if mapped.isna().any():
            raise ValueError(
                "The supplied folds file is missing some training IDs."
            )

        return [mapped.to_numpy(dtype=int)]

    sizes = df["question"].value_counts()
    score_strata = df["question"] + "_" + df["score"].astype(str)

    if score_strata.value_counts().min() >= n_splits:
        strata = score_strata
        print("Stratifying by question and score.", flush=True)

    elif sizes.min() >= n_splits:
        strata = df["question"]
        print("Stratifying by question only.", flush=True)

    else:
        print(
            "Some questions are too small to stratify; "
            "using shuffled KFold.",
            flush=True,
        )

        out = []

        for r in range(repeats):
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed + r,
            )

            fold = np.zeros(len(df), dtype=int)

            for k, (_, va) in enumerate(splitter.split(df)):
                fold[va] = k

            out.append(fold)

        return out

    out = []

    for r in range(repeats):
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + r,
        )

        fold = np.zeros(len(df), dtype=int)

        for k, (_, va) in enumerate(
            splitter.split(np.zeros(len(df)), strata)
        ):
            fold[va] = k

        out.append(fold)

    return out


def fit_vecs(texts):
    wv = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    cv = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_features=60000,
        sublinear_tf=True,
    )

    Xw = wv.fit_transform(texts).tocsr()
    Xc = cv.fit_transform(texts).tocsr()

    for vec in (wv, cv):
        if hasattr(vec, "stop_words_"):
            del vec.stop_words_

    return wv, cv, Xw, Xc


def centroid_features(Xw_train, y_train, Xw_query, allowed_scores):
    """Build centroid similarities using training rows only."""
    Xn = normalize(Xw_train, norm="l2")
    Qn = normalize(Xw_query, norm="l2")
    centroids = []
    y_train = np.asarray(y_train)

    for score in allowed_scores:
        mask = y_train == score

        if mask.any():
            centroids.append(
                np.asarray(Xn[mask].mean(axis=0)).ravel()
            )
        else:
            centroids.append(
                np.zeros(Xw_train.shape[1], dtype=float)
            )

    C = np.vstack(centroids)
    return np.asarray(Qn @ C.T)


def centroid_features_loo(Xw_train, y_train, allowed_scores):
    """Training centroid features excluding each essay itself."""
    Xn = normalize(Xw_train, norm="l2").tocsr()
    y_train = np.asarray(y_train)

    class_sums = {}
    class_counts = {}

    for score in allowed_scores:
        mask = y_train == score
        class_counts[score] = int(mask.sum())

        if mask.any():
            class_sums[score] = sparse.csr_matrix(
                Xn[mask].sum(axis=0)
            )
        else:
            class_sums[score] = sparse.csr_matrix(
                (1, Xn.shape[1])
            )

    out = np.zeros(
        (Xn.shape[0], len(allowed_scores)),
        dtype=float,
    )

    for j, score in enumerate(allowed_scores):
        count = class_counts[score]

        if count <= 1:
            continue

        centroid = class_sums[score] / count
        sims = (Xn @ centroid.T).toarray().ravel()

        own = y_train == score

        if own.any():
            own_rows = Xn[own]

            own_dot_sum = (
                own_rows @ class_sums[score].T
            ).toarray().ravel()

            row_norm_sq = np.asarray(
                own_rows.multiply(own_rows).sum(axis=1)
            ).ravel()

            sims[own] = (
                own_dot_sum - row_norm_sq
            ) / (count - 1)

        out[:, j] = sims

    return out


def oof_components(df, z, y, folds, small_q, allowed_by_q):
    N, R = len(df), len(folds)
    qids = df["question"].to_numpy()
    texts = df["text"].tolist()

    alphas = sorted(set(A_BIG + A_SMALL))
    rq = {a: np.zeros((R, N)) for a in alphas}
    rg = {a: np.zeros((R, N)) for a in A_GLOBAL}

    center_all = np.zeros((R, N))
    ntrain_all = np.zeros((R, N))

    for r, fold in enumerate(folds):
        fold_ids = np.unique(fold)

        for fold_num, fold_id in enumerate(fold_ids, start=1):
            tr = np.where(fold != fold_id)[0]
            va = np.where(fold == fold_id)[0]

            # NEW: progress output so the script doesn't appear frozen.
            print(
                f"[CV] Repeat {r + 1}/{R}, "
                f"fold {fold_num}/{len(fold_ids)} starting",
                flush=True,
            )
            print(
                f"     Train={len(tr)}, validation={len(va)}",
                flush=True,
            )
            print("     Building TF-IDF features...", flush=True)

            wv, cv, Xw_tr, Xc_tr = fit_vecs(
                [texts[i] for i in tr]
            )

            Xw_va = wv.transform(
                [texts[i] for i in va]
            ).tocsr()

            Xc_va = cv.transform(
                [texts[i] for i in va]
            ).tocsr()

            X_tr = sparse.hstack([Xw_tr, Xc_tr]).tocsr()
            X_va = sparse.hstack([Xw_va, Xc_va]).tocsr()

            qtr, qva = qids[tr], qids[va]

            global_mean = float(z[tr].mean())
            center_tr = np.full(len(tr), global_mean)
            center_va = np.full(len(va), global_mean)

            qinfo = {}

            for q in np.unique(qtr):
                mask = qtr == q
                qcenter = float(z[tr][mask].mean())
                qinfo[q] = (qcenter, int(mask.sum()))
                center_tr[mask] = qcenter

            ntr_va = np.zeros(len(va))

            for q, (qcenter, count) in qinfo.items():
                mask = qva == q
                center_va[mask] = qcenter
                ntr_va[mask] = count

            center_all[r, va] = center_va
            ntrain_all[r, va] = ntr_va

            for alpha in A_GLOBAL:
                model = Ridge(alpha=alpha).fit(
                    X_tr, z[tr] - center_tr
                )
                rg[alpha][r, va] = (
                    center_va + model.predict(X_va)
                )

            for q in qinfo:
                itr = np.where(qtr == q)[0]
                iva = np.where(qva == q)[0]

                if not len(itr) or not len(iva):
                    continue

                yq = y[tr][itr]
                allowed = allowed_by_q[q]

                csim_tr = centroid_features_loo(
                    Xw_tr[itr], yq, allowed
                )

                csim_va = centroid_features(
                    Xw_tr[itr], yq, Xw_va[iva], allowed
                )

                Xq_tr = sparse.hstack([
                    X_tr[itr],
                    sparse.csr_matrix(csim_tr),
                ]).tocsr()

                Xq_va = sparse.hstack([
                    X_va[iva],
                    sparse.csr_matrix(csim_va),
                ]).tocsr()

                zq = z[tr][itr]
                alphas_for_q = A_SMALL if q in small_q else A_BIG

                for alpha in alphas_for_q:
                    model = Ridge(alpha=alpha).fit(Xq_tr, zq)
                    rq[alpha][r, va[iva]] = model.predict(Xq_va)

            # NEW: confirmation after each complete fold.
            print(
                f"[CV] Repeat {r + 1}, fold {fold_num} complete.",
                flush=True,
            )

        print(f"Repeat {r + 1}/{R} done.", flush=True)

    return {
        "rq": rq,
        "rg": rg,
        "cen": center_all,
        "ntr": ntrain_all,
    }


def evaluate(comp, params, y, lo, hi, df, small_q):
    qids = df["question"].to_numpy()

    small_mask = np.isin(
        qids, list(small_q)
    )[None, :]

    local = np.where(
        small_mask,
        comp["rq"][params["a_small"]],
        comp["rq"][params["a_big"]],
    )

    n = comp["ntr"]
    m = params["m"]

    w = n / (n + m + 1e-9) if m > 0 else np.ones_like(n)

    zpred = (
        w * local
        + (1 - w) * comp["rg"][params["a_g"]]
    )

    center = comp["cen"]
    zpred = center + params["s"] * (zpred - center) + params["b"]

    raw = lo[None, :] + zpred * (hi - lo)[None, :]
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    pred = np.clip(
        np.rint(raw),
        lo[None, :],
        hi[None, :],
    ).astype(int)

    scores = {}

    for q in np.unique(qids):
        idx = np.where(qids == q)[0]
        qlo, qhi = lo[idx[0]], hi[idx[0]]

        scores[q] = float(np.mean([
            qwk(y[idx], pred[r, idx], qlo, qhi)
            for r in range(pred.shape[0])
        ]))

    return scores


def search(comp, y, lo, hi, df, small_q):
    keys = ["a_big", "a_small", "a_g", "m", "s", "b"]
    grids = [A_BIG, A_SMALL, A_GLOBAL, M_GRID, S_GRID, B_GRID]

    results = []
    start = time.time()
    total = np.prod([len(g) for g in grids])

    print(f"Testing {total} parameter combinations...", flush=True)

    for i, values in enumerate(itertools.product(*grids), start=1):
        params = dict(zip(keys, values))
        scores = evaluate(comp, params, y, lo, hi, df, small_q)

        results.append((
            float(np.mean(list(scores.values()))),
            params,
            scores,
        ))

        if i % 100 == 0 or i == total:
            print(
                f"  Tested {i}/{total} combinations "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

    results.sort(key=lambda item: -item[0])
    return results


def fit_final(df, z, y, params, ranges, allowed_by_q, small_q):
    qids = df["question"].to_numpy()

    wv, cv, Xw, Xc = fit_vecs(df["text"].tolist())
    X = sparse.hstack([Xw, Xc]).tocsr()

    global_center = float(z.mean())
    centers = np.full(len(df), global_center)
    per_q = {}

    for q in np.unique(qids):
        idx = np.where(qids == q)[0]
        center = float(z[idx].mean())
        centers[idx] = center
        allowed = allowed_by_q[q]

        csim = centroid_features_loo(
            Xw[idx], y[idx], allowed
        )

        Xq = sparse.hstack([
            X[idx],
            sparse.csr_matrix(csim),
        ]).tocsr()

        alpha = (
            params["a_small"]
            if q in small_q
            else params["a_big"]
        )

        ridge = Ridge(alpha=alpha).fit(Xq, z[idx])

        Xn = normalize(Xw[idx], norm="l2")
        cents = []

        for score in allowed:
            mask = y[idx] == score

            if mask.any():
                cents.append(
                    np.asarray(Xn[mask].mean(axis=0)).ravel()
                )
            else:
                cents.append(
                    np.zeros(Xw.shape[1], dtype=float)
                )

        per_q[q] = {
            "ridge": ridge,
            "center": center,
            "n": int(len(idx)),
            "centroids": np.vstack(cents),
            "allowed": allowed,
        }

    glob = Ridge(alpha=params["a_g"]).fit(X, z - centers)

    return {
        "word_vec": wv,
        "char_vec": cv,
        "per_q": per_q,
        "glob": glob,
        "global_center": global_center,
        "params": dict(params),
        "ranges": ranges,
        "small_q": sorted(small_q),
        "small_n": SMALL_N,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training_dataset_updated.csv")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    print("Loading dataset...", flush=True)
    df = load_data(args.csv)

    ranges, allowed_by_q = get_score_info(df, args.questions)

    lo = df["question"].map(lambda q: ranges[q][0]).to_numpy(float)
    hi = df["question"].map(lambda q: ranges[q][1]).to_numpy(float)
    y = df["score"].to_numpy(int)

    z = np.divide(
        y - lo,
        hi - lo,
        out=np.zeros_like(lo, dtype=float),
        where=(hi - lo) != 0,
    )

    if not np.isfinite(z).all():
        raise ValueError("Normalized scores contain NaN or infinity.")

    sizes = df["question"].value_counts()
    small_q = set(sizes[sizes < SMALL_N].index)

    print(
        f"Loaded {len(df)} essays across {len(sizes)} questions.",
        flush=True,
    )
    print(f"Small questions: {len(small_q)}", flush=True)

    params = dict(DEFAULT)

    if not args.skip_cv:
        folds = make_folds(
            df, args.n_splits, args.repeats,
            args.seed, args.folds,
        )

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "Id": df["id"],
            **{f"fold_r{r}": f for r, f in enumerate(folds)},
        }).to_csv(out_dir / "folds_approach3.csv", index=False)

        print("Computing out-of-fold predictions...", flush=True)

        comp = oof_components(
            df, z, y, folds, small_q, allowed_by_q
        )

        print("Searching hyperparameters...", flush=True)

        results = search(comp, y, lo, hi, df, small_q)
        best_score, params, best_scores = results[0]

        print(f"\nBEST mean validation QWK = {best_score:.4f}")
        print("Best parameters:", params)

        big = [
            v for q, v in best_scores.items()
            if q not in small_q
        ]
        small = [
            v for q, v in best_scores.items()
            if q in small_q
        ]

        if big:
            print(f"Big-question mean QWK: {np.mean(big):.4f}")

        if small:
            print(f"Small-question mean QWK: {np.mean(small):.4f}")

    print("\nFitting final model on all data...", flush=True)

    bundle = fit_final(
        df, z, y, params, ranges, allowed_by_q, small_q
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "approach3_bundle.joblib"
    joblib.dump(bundle, path, compress=3)

    print(f"Saved model to {path}", flush=True)


if __name__ == "__main__":
    main()
