"""Tables and figures for Experiments 2 and 3 from the .npz files written by exp2_exp3.py.

    python experiments/make_figures.py --csv training_dataset_updated.csv

Needs experiments/results/random.npz (Experiment 2) and, for Experiment 3, any
experiments/results/cluster_N*.npz. Writes exp2_results.csv, exp3_heatmap.csv,
fig_exp2.png and fig_exp3.png (heatmap + clustered-vs-random curves) to experiments/results/.
"""
import argparse
import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, ROOT, load_data, score  # noqa: E402
from exp2_exp3 import split_parts  # noqa: E402

INK, INK2, GREY, GRID, BLUE = "#0b0b0b", "#52514e", "#a8a69f", "#e4e2dc", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "training_dataset_updated.csv"))
    args = ap.parse_args()
    df = load_data(args.csv)
    part = split_parts(df)

    # ---------------- Experiment 2
    z = np.load(OUT / "random.npz")
    sizes, preds, ntr = z["sizes"], z["preds"], z["ntrain"]
    rows_all = np.isin(part, z["splits"])
    e2 = pd.DataFrame([dict(pct=int(round(p * 100)), n_train=int(n), **score(df, preds[k], rows=rows_all)[0])
                       for k, (p, n) in enumerate(zip(sizes, ntr))])
    e2.round(4).to_csv(OUT / "exp2_results.csv", index=False)
    print("Experiment 2:\n", e2.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for col, c, lw, ls in [("original", GREY, 1.6, "-"), ("new", GREY, 1.6, "--"), ("all", BLUE, 2.4, "-")]:
        ax.plot(e2.n_train, e2[col], color=c, lw=lw, ls=ls, marker="o", ms=4)
        ax.annotate({"original": "Original Qs (1-10)", "new": "New Qs (11-91)", "all": "All questions"}[col],
                    (e2.n_train.iloc[-1], e2[col].iloc[-1]), xytext=(8, {"original": 0, "new": -9, "all": 5}[col]),
                    textcoords="offset points", va="center", fontsize=9,
                    color=BLUE if col == "all" else INK2, fontweight="bold" if col == "all" else None)
    ax.set_xlabel("Number of graded training examples (all questions)")
    ax.set_ylabel("QWK (per question, averaged)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlim(0, e2.n_train.max() * 1.22)
    ax.set_title(f"QWK rises from {e2['all'].iloc[0]:.3f} to {e2['all'].iloc[-1]:.3f} as the graded set grows",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp2.png", dpi=200)
    plt.close(fig)

    # ---------------- Experiment 3
    files = sorted(glob.glob(str(OUT / "cluster_N*.npz")), key=lambda f: int(f.split("_N")[1].split(".")[0]))
    if not files:
        print("no cluster_N*.npz yet - skipping Experiment 3")
        return
    h0 = np.load(files[0])
    hs, rows = h0["sizes"], np.isin(part, h0["splits"])      # compare on the SAME splits and answers
    ridx = [int(np.argmin(np.abs(sizes - p))) for p in hs]
    table = {"random": [score(df, preds[k], rows=rows)[0]["all"] for k in ridx]}
    for f in files:
        hz = np.load(f)
        table[f"N = {f.split('_N')[1].split('.')[0]}"] = [score(df, hz["preds"][k], rows=rows)[0]["all"] for k in range(len(hs))]
    H = pd.DataFrame(table, index=[f"{int(round(p * 100))}%" for p in hs]).T
    H.round(4).to_csv(OUT / "exp3_heatmap.csv")
    print("\nExperiment 3 (QWK, all questions):\n", H.round(3).to_string())
    Hn = H.drop(index="random")
    best = Hn.mean(axis=1).idxmax()
    nts = [ntr[k] for k in ridx]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    a1.imshow(Hn.values, cmap="Blues", aspect="auto", vmin=np.nanmin(H.values) - 0.02, vmax=np.nanmax(H.values) + 0.01)
    a1.set_xticks(range(len(hs)), list(H.columns))
    a1.set_yticks(range(len(Hn)), Hn.index)
    for i in range(Hn.shape[0]):
        for j in range(Hn.shape[1]):
            v = Hn.values[i, j]
            a1.text(j, i, f"{v:.2f}\n({v - H.loc['random'].values[j]:+.2f})", ha="center", va="center", fontsize=7.5,
                    color="white" if v > np.nanpercentile(H.values, 70) else INK)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_xlabel("Training-set size (share of T)")
    a1.set_ylabel("Clusters per question (N)")
    a1.set_title("(a) QWK by N; ( ) = vs. random selection", loc="left", color=INK, fontsize=10)
    a2.plot(nts, H.loc["random"], color=GREY, lw=2, marker="o", ms=4)
    a2.plot(nts, H.loc[best], color=BLUE, lw=2.4, marker="o", ms=4)
    a2.annotate("Random", (nts[-1], H.loc["random"].iloc[-1]), xytext=(6, -10), textcoords="offset points", color=INK2, fontsize=9)
    a2.annotate(f"Clustered\n({best})", (nts[-1], H.loc[best].iloc[-1]), xytext=(6, 2), textcoords="offset points",
                color=BLUE, fontsize=9, fontweight="bold")
    a2.set_xlim(0, max(nts) * 1.28)
    a2.grid(axis="y", color=GRID, lw=0.8)
    a2.set_xlabel("Graded training examples (all questions)")
    a2.set_ylabel("QWK")
    a2.set_title(f"(b) Clustered ({best}) vs. random selection", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp3.png", dpi=200)
    plt.close(fig)
    print("\nsaved exp2_results.csv, exp3_heatmap.csv, fig_exp2.png, fig_exp3.png in experiments/results/")


if __name__ == "__main__":
    main()
