"""Training script for Approach 3 (run locally; NOT submitted to the autograder logic).

Usage:
  python training_approach3.py --csv training_dataset_updated.csv --out weights \
      --assets local_assets/assets.json --emb minilm,bge_small

  --assets is optional: if omitted, encoders are loaded by Hugging Face name
  (needs internet / cached download). assets.json must look like
  {"models": {"minilm": {"path": "..."}, "bge_small": {"path": "..."}}}.
  --emb none  trains the TF-IDF-only version (fast smoke test).

"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

import approach3 as A


def load_df(path):
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    if "EssaySet" not in df.columns and "question" not in df.columns:
        df = pd.read_csv(path, sep="\t", encoding="utf-8", encoding_errors="replace")
    ren = {"Id": "id", "EssaySet": "question", "EssayText": "text", "Score1": "y"}
    df = df.rename(columns=ren)
    df["question"] = df["question"].astype(str)
    df["text"] = df["text"].fillna("").map(A.clean_text)
    df["y"] = df["y"].astype(float)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--assets", default=None)
    ap.add_argument("--emb", default="minilm,bge_small")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="emb_cache")
    args = ap.parse_args()

    t0 = time.time()
    df = load_df(args.csv)
    emb_names = [] if args.emb.lower() == "none" else [e.strip() for e in args.emb.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)

    paths = {}
    if args.assets:
        a = json.load(open(args.assets))
        for n in emb_names:
            paths[n] = a["models"][n]["path"]
    else:
        for n in emb_names:
            paths[n] = A.DEFAULT_HF[n]

    # ---- embeddings for every training row (cached) ----
    embs_all = {}
    for n in emb_names:
        f = cache / f"{n}_{len(df)}.npy"
        if f.exists():
            embs_all[n] = np.load(f)
        else:
            enc = A.load_encoder(paths[n], local_only=bool(args.assets))
            embs_all[n] = A.encode_texts(enc, df["text"].tolist())
            np.save(f, embs_all[n])
            del enc
        print(f"embeddings {n}: {embs_all[n].shape}  ({time.time()-t0:.0f}s)")

    per_q, report, fold_rows = {}, [], []
    gl_X, gl_y = [], []
    n_lex = 3

    for q, sub in df.groupby("question"):
        ids = sub.index.values
        texts = sub["text"].tolist()
        y = sub["y"].values
        grades = sorted(np.unique(y).tolist())
        gmin, gmax = grades[0], grades[-1]
        k = len(grades)
        if emb_names:
            gl_X.append(np.hstack([embs_all[n][ids] for n in emb_names]))
            gl_y.append((y - gmin) / max(gmax - gmin, 1e-9))
        if k < 2 or len(ids) < 30:
            per_q[q] = {"const": True, "const_frac": 0.5, "grades": grades}
            continue

        qe = {n: embs_all[n][ids] for n in emb_names}
        y_idx = np.searchsorted(grades, y)
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        fold_id = np.zeros(len(ids), dtype=int)
        oof = None
        for f, (tr, va) in enumerate(skf.split(texts, y_idx)):
            fold_id[va] = f
            st = A.fit_base([texts[i] for i in tr], {n: qe[n][tr] for n in emb_names},
                            y[tr], emb_names)
            P = A.predict_base(st, [texts[i] for i in va], {n: qe[n][va] for n in emb_names},
                               emb_names, use_emb=True)
            if oof is None:
                oof = np.zeros((len(ids), P.shape[1]))
            oof[va] = P
        for i, rid in enumerate(sub["id"].values):
            fold_rows.append((rid, q, int(fold_id[i])))

        names = ["ridge_word", "ridge_char", "logit_wc"]
        for n in emb_names:
            names += [f"knn_{n}", f"krr_{n}"]
        if emb_names:
            names.append("logit_emb")
        row = {"question": q, "n": len(ids)}
        for j, nm in enumerate(names):  # simple rounding, for comparison
            idx = np.clip(np.searchsorted(grades, oof[:, j]), 0, k - 1)
            near = np.abs(oof[:, j][:, None] - np.array(grades)[None, :]).argmin(1)
            row[nm] = A.qwk_idx(near, y_idx, k)
        # honest estimate: blend + thresholds fit on other folds' OOF rows
        hon = np.zeros(len(ids), dtype=int)
        for f in range(args.folds):
            m = fold_id != f
            bl = A.fit_blend(oof[m], y[m], grades)
            _, ix = A.apply_blend(oof[~m], bl, grades)
            hon[~m] = ix
        row["BLEND_cv_qwk"] = A.qwk_idx(hon, y_idx, k)
        report.append(row)
        print(f"Q{q}: n={len(ids)} blend CV QWK={row['BLEND_cv_qwk']:.4f}  "
              f"best single={max(row[nm] for nm in names):.4f}")

        # final fit on all rows of this question
        blend_full = A.fit_blend(oof, y, grades)
        blend_lex = A.fit_blend(oof[:, :n_lex], y, grades)
        st_final = A.fit_base(texts, qe, y, emb_names)
        per_q[q] = {"grades": grades, "state": A.compact_state(st_final),
                    "blend_full": blend_full, "blend_lex": blend_lex}

    rep = pd.DataFrame(report)
    rep.to_csv("cv_report_approach3.csv", index=False)
    pd.DataFrame(fold_rows, columns=["id", "question", "fold"]).to_csv(
        "folds_approach3.csv", index=False)
    print("\nMean over questions:\n", rep.drop(columns=["question", "n"]).mean().round(4))

    bundle = {"emb_names": emb_names, "per_q": per_q, "global": None}
    if emb_names and gl_X:  # fallback for questions unseen in training
        bundle["global"] = {"ridge": Ridge(alpha=3.0).fit(np.vstack(gl_X), np.concatenate(gl_y))}
    path = out / A.BUNDLE_NAME
    joblib.dump(bundle, path, compress=3)
    print(f"saved {path}  {path.stat().st_size/1e6:.1f} MB   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()