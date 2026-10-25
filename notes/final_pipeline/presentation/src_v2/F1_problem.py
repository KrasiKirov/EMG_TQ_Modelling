"""F1_problem: last-semester problem figures.

OLD pipeline (E01_old) per-angle R² scored against its OWN target
(fast_seed<k>.json 'per_angle' r2), learned from session 1 -> tested on session 2.
Main subjects (YES, JM, EG): mean over seeds 0,1,2 from screen study.
Extra subjects (IES, CH, HM): seed 0 from confirm study.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = os.environ.get(
    "WT",
    "C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137",
)
R = os.path.join(WT, "ML", "research", "results")
OUT = os.path.join(WT, "notes", "final_pipeline", "presentation", "figures_v2")
os.makedirs(OUT, exist_ok=True)

MAIN = ["YES", "JM", "EG"]
EXTRA = ["IES", "CH", "HM"]

GREY = "#b9b8b3"
EDGE = "#4a4945"
BAD = "#c0392b"
GRID = "#e6e5e1"
SEC = "#52514e"
MINUS = "\u2212"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": True,
})

YMIN, YMAX = -1.3, 1.12
FLOOR_DRAW = -1.2


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if s in (MINUS + "0.00",):
        s = "0.00"
    return s


def load_r2(subj):
    if subj in MAIN:
        study, seeds = "screen", [0, 1, 2]
    else:
        study, seeds = "confirm", [0]
    per_seed = []
    for k in seeds:
        p = os.path.join(R, study, subj, "S1toS2", "E01_old", f"fast_seed{k}.json")
        with open(p) as f:
            d = json.load(f)
        per_seed.append({key: v["r2"] for key, v in d["per_angle"].items() if key.startswith("S2:")})
    keys = sorted(per_seed[0].keys(), key=lambda s: float(s.split(":")[1]))
    for ps in per_seed:
        assert sorted(ps.keys()) == sorted(keys), subj
    angles = np.array([float(k.split(":")[1]) for k in keys])
    r2 = np.array([np.mean([ps[k] for ps in per_seed]) for k in keys])
    return angles, r2, len(seeds)


def draw_panel(ax, subj, angles, r2, left):
    x = np.arange(len(angles))
    for xi, v in zip(x, r2):
        bad = v < 0.5
        h = max(v, FLOOR_DRAW)
        ax.bar(xi, h, width=0.72, color=BAD if bad else GREY,
               edgecolor=BAD if bad else EDGE, linewidth=1.0, zorder=3)
        if h >= 0:
            y, va = h + 0.03, "bottom"
        else:
            y, va = h - 0.03, "top"
        txt = fmt(v)
        ax.text(xi, y, txt, ha="center", va=va, fontsize=10.5,
                fontweight="bold" if bad else "normal",
                color=BAD if bad else "#1b1b1b", zorder=4)
    ax.axhline(0, color="#1b1b1b", lw=1.0, zorder=2)
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels([fmt(t, 1) if t != 0 else "0" for t in [-1, -0.5, 0, 0.5, 1]])
    ax.set_xlim(-0.6, len(angles) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([fmt(a) for a in angles], rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(subj)
    if left:
        ax.set_ylabel("R\u00b2 (1 = perfect)")
    ax.set_xlabel("ankle angle (rad)   \u2190 toes down     toes up \u2192",
                  fontsize=11.5, color=SEC)


def make(subjs_rows, fname, figsize, data):
    nr = len(subjs_rows)
    nc = len(subjs_rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)
    for r, row in enumerate(subjs_rows):
        for c, s in enumerate(row):
            a, v, _ = data[s]
            draw_panel(axes[r, c], s, a, v, left=(c == 0))
    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main():
    data = {s: load_r2(s) for s in MAIN + EXTRA}
    out = []
    out.append(make([MAIN, EXTRA], "problem_all6.png", (13.3, 6.6), data))
    out.append(make([MAIN], "problem_main3.png", (13.3, 3.6), data))
    out.append(make([EXTRA], "problem_extra3.png", (13.3, 3.6), data))
    for p in out:
        print("saved", p)
    print("NUMBERS")
    for s in MAIN + EXTRA:
        a, v, n = data[s]
        for ai, vi in zip(a, v):
            print(f"{s}\t{ai:.3f}\t{vi:.4f}\t(n_seeds={n})")
        print(f"{s} mean(floor -1) = {np.mean(np.maximum(v, -1)):.3f}; n<0.5 = {(v < 0.5).sum()}")


if __name__ == "__main__":
    main()
