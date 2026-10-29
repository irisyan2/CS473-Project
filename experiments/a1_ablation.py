"""Approach 1 tuning path and ablations (report Section 3.1, "Tuning").

    python experiments/a1_ablation.py --csv training_dataset_updated.csv

Re-implements the final Approach 1 pipeline with switches, so each setting can be turned
off or changed one at a time, and scores every variant on the shared 5 folds.
Writes experiments/results/a1_tuning_path.csv and a1_ablations.csv. Takes ~10 minutes.
"""
import argparse
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from common import OUT, ROOT, fmt, load_data, question_info, score, shared_folds

BR = re.compile(r"<br\s*/?>", re.I)
TOK = re.compile(r"(?u)\b\w\w+\b")
_stem = lru_cache(None)(PorterStemmer().stem)

FINAL = dict(char=(3, 6), word=True, stem=True, stop=False, word_weight=0.5, len_weight=0.5,
             k_factor=0.5, k_factor_small=0.7, power=5, low_sim=0.15, smooth=0.5, fixed_k=None)


def words(t, stem, stop):
    w = TOK.findall(t.lower())
    if stop:
        w = [x for x in w if x not in ENGLISH_STOP_WORDS]
    return " ".join(_stem(x) if stem else x for x in w)


def run(df, fold, allowed, cfg):
    c = {**FINAL, **cfg}
    T = df.EssayText.astype(str).map(lambda t: BR.sub(" ", t)).to_numpy()
    Q, Y = df.EssaySet.to_numpy(), df.Score1.to_numpy()
    WT = np.array([words(t, c["stem"], c["stop"]) for t in T]) if c["word"] else None
    CT = np.array([words(t, False, True) for t in T]) if c["stop"] else T
    L = np.log1p(np.array([len(t) for t in T]))
    pred = np.empty(len(df))
    for f in range(5):
        tr = fold != f
        parts = []
        if c["char"]:
            parts.append(TfidfVectorizer(analyzer="char_wb", ngram_range=c["char"], min_df=2, sublinear_tf=True,
                                         dtype=np.float32).fit(CT[tr]).transform(CT))
        if c["word"]:
            Xw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 dtype=np.float32).fit(WT[tr]).transform(WT)
            parts.append(Xw * (c["word_weight"] if c["char"] else 1.0))
        X = normalize(hstack(parts).tocsr())
        for q in np.unique(Q):
            idx = np.where(Q == q)[0]
            itr, ite = idx[tr[idx]], idx[~tr[idx]]
            n = len(itr)
            S = (X[ite] @ X[itr].T).toarray()
            if c["len_weight"]:
                S *= np.exp(-c["len_weight"] * np.abs(L[ite][:, None] - L[itr][None, :]))
            k = c["fixed_k"] or max(1, round(np.sqrt(n) * (c["k_factor_small"] if n < 100 else c["k_factor"])))
            k = min(k, n)
            top = np.argsort(-S, 1)[:, :k]
            s = np.clip(np.take_along_axis(S, top, 1), 0, None)
            w = s ** c["power"]
            tot = w.sum(1, keepdims=True)
            w = np.where(tot > 0, w / np.where(tot > 0, tot, 1), 1.0)
            A = np.asarray(allowed[q])
            lab = Y[itr][top]
            v = np.stack([(w * (lab == g)).sum(1) for g in A], 1)
            sig = c["smooth"] * (A.max() - A.min())
            if sig > 0:
                v = v @ np.exp(-(A[:, None] - A[None, :]) ** 2 / (2 * sig ** 2))
            p = A[np.argmax(v >= v.max(1, keepdims=True) * (1 - 1e-6), 1)]
            if c["low_sim"]:
                p[s.max(1) < c["low_sim"]] = A[0]
            pred[ite] = p
    return score(df, pred)[0]


STEPS = [
    ("1. Word TF-IDF (stemmed), fixed K = 25", dict(char=None, lenw=0, len_weight=0, low_sim=0, smooth=0, fixed_k=25)),
    ("2. Character 3-5 grams instead of words", dict(char=(3, 5), word=False, len_weight=0, low_sim=0, smooth=0, fixed_k=25)),
    ("3. K scaled to question size", dict(char=(3, 5), word=False, len_weight=0, low_sim=0, smooth=0, k_factor_small=0.5)),
    ("4. Character 3-6 grams + length feature", dict(word=False, low_sim=0, smooth=0, k_factor_small=0.5)),
    ("5. Low-similarity rule", dict(word=False, smooth=0, k_factor_small=0.5)),
    ("6. Add stemmed word view, larger K on small Qs", dict(smooth=0)),
    ("7. Ordinal vote smoothing (final)", dict()),
]
ABLATIONS = [
    ("Final model", dict()),
    ("Remove low-similarity rule", dict(low_sim=0)),
    ("Remove length feature", dict(len_weight=0)),
    ("Add stop-word removal", dict(stop=True)),
    ("Remove vote smoothing", dict(smooth=0)),
    ("Unweighted vote (no similarity^5)", dict(power=0)),
    ("Remove word view (character only)", dict(word=False)),
    ("Remove stemming", dict(stem=False)),
    ("Fixed K = 25 instead of scaled K", dict(fixed_k=25)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "training_dataset_updated.csv"))
    args = ap.parse_args()
    df = load_data(args.csv)
    fold = shared_folds(df)
    allowed, _ = question_info(df)
    for name, table, outfile in [("tuning path", STEPS, "a1_tuning_path.csv"), ("ablations", ABLATIONS, "a1_ablations.csv")]:
        print(f"== Approach 1 {name}")
        rows = []
        for label, cfg in table:
            cfg = {k: v for k, v in cfg.items() if k in FINAL}
            s = run(df, fold, allowed, cfg)
            rows.append(dict(config=label, **s))
            print(f"  {label:48s} {fmt(s)}", flush=True)
        pd.DataFrame(rows).round(4).to_csv(OUT / outfile, index=False)
    print("saved experiments/results/a1_tuning_path.csv and a1_ablations.csv")


if __name__ == "__main__":
    main()
