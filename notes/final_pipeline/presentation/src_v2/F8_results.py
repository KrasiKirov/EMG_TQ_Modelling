"""F8_results: old vs new, every subject, scored against the SHARED answer key.

Source: fast_seed<k>_truth.json 'per_angle' (r2, rmse) = metrics of each model's
predictions against the shared truth (measured torque - in-trial passive B(t)).
  OLD  = E01_old                      (screen for YES/JM/EG seeds 0-2, confirm seed 0 for IES/CH/HM)
  NEW  = E22_restanchor_final         (screen seeds 0-2 for YES/JM/EG; confirm seed 0 for IES, HM)
         E32_restanchor_noagc_moe     (confirm seed 0 for CH; = FINAL for CH)
Averaging: mean over seeds first; R2 then floored at -1 before any mean over angles.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch

WT = os.environ.get(
    "WT",
    "C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137",
)
R = os.path.join(WT, "ML", "research", "results")
OUT = os.path.join(WT, "notes", "final_pipeline", "presentation", "figures_v2")
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

MAIN = ["YES", "JM", "EG"]
EXTRA = ["IES", "CH", "HM"]
SUBJ = MAIN + EXTRA
PROTS = ["S1toS2", "S2toS1"]
PROT_TITLE = {
    "S1toS2": "learned from session 1 \u2192 tested on session 2",
    "S2toS1": "learned from session 2 \u2192 tested on session 1",
}

OLD = "#2a78d6"
NEW = "#eb6834"
OLD_D = "#1d5aa3"   # darker text versions for printed values
NEW_D = "#b8471c"
TRUTH = "#1b1b1b"
GREYC = "#8a8984"
BAD = "#c0392b"
GRID = "#e6e5e1"
SEC = "#52514e"
MINUS = "\u2212"
DEG = 57.29578

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11.5,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": True,
    "hatch.linewidth": 1.2,
})

FLOOR = -1.0
FLOOR_DRAW = -1.2


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if s == MINUS + "0." + "0" * nd:
        s = "0." + "0" * nd
    return s


def src(subj, which):
    if subj in MAIN:
        study, seeds = "screen", [0, 1, 2]
        cfg = "E01_old" if which == "old" else "E22_restanchor_final"
    else:
        study, seeds = "confirm", [0]
        if which == "old":
            cfg = "E01_old"
        else:
            cfg = "E32_restanchor_noagc_moe" if subj == "CH" else "E22_restanchor_final"
    return study, cfg, seeds


def load(subj, prot, which):
    study, cfg, seeds = src(subj, which)
    tested = "S2:" if prot == "S1toS2" else "S1:"
    per_seed = []
    for k in seeds:
        p = os.path.join(R, study, subj, prot, cfg, f"fast_seed{k}_truth.json")
        with open(p) as f:
            d = json.load(f)
        per_seed.append({key: v for key, v in d["per_angle"].items() if key.startswith(tested)})
    keys = sorted(per_seed[0].keys(), key=lambda s: float(s.split(":")[1]))
    for ps in per_seed:
        assert sorted(ps.keys()) == sorted(keys), (subj, prot, which)
    angles = np.array([float(k.split(":")[1]) for k in keys])
    r2 = np.array([np.mean([ps[k]["r2"] for ps in per_seed]) for k in keys])
    rmse = np.array([np.mean([ps[k]["rmse"] for ps in per_seed]) for k in keys])
    return dict(keys=keys, angles=angles, r2=r2, rmse=rmse, n=len(seeds), cfg=cfg, study=study)


def collect():
    D = {}
    for s in SUBJ:
        for p in PROTS:
            o = load(s, p, "old")
            n = load(s, p, "new")
            assert o["keys"] == n["keys"], (s, p)
            D[(s, p)] = dict(angles=o["angles"], old_r2=o["r2"], new_r2=n["r2"],
                             old_rmse=o["rmse"], new_rmse=n["rmse"], n=o["n"],
                             new_cfg=n["cfg"], study=o["study"])
    return D


def tick_labels(angles):
    return [f"{fmt(a)} ({fmt(a * DEG, 0)}\u00b0)" for a in angles]


def style_x(ax, angles):
    x = np.arange(len(angles))
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels(angles), fontsize=11, rotation=38, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.95, len(angles) - 0.4)
    ax.tick_params(axis="x", length=3, pad=2)


XLABEL = "angle, rad (deg)"


def bottom_xlabel(ax):
    ax.set_xlabel(XLABEL, fontsize=11.5, color=SEC)
    for xf, ha, txt in ((0.0, "left", "\u2190 toes down"), (1.0, "right", "toes up \u2192")):
        ax.annotate(txt, xy=(xf, 0.5), xycoords=("axes fraction", ax.xaxis.label),
                    ha=ha, va="center", fontsize=11.5, color=SEC)


def fmtv(v):
    return fmt(v, 1) if abs(v) >= 10 else fmt(v)


def layout(fig):
    fig.subplots_adjust(left=0.055, right=0.995, top=0.835, bottom=0.17, hspace=0.78, wspace=0.17)


def panel_title(s):
    return s if s in MAIN else f"{s} (1 run)"


# ---------------------------------------------------------------- R2 per angle
R2_YMIN, R2_YMAX = -1.55, 1.62


def draw_r2(ax, s, d, left, bottom):
    a = d["angles"]
    x = np.arange(len(a))
    w = 0.4
    for xi, vo, vn in zip(x, d["old_r2"], d["new_r2"]):
        for off, v, col, tcol in ((-w / 2, vo, OLD, OLD_D), (w / 2, vn, NEW, NEW_D)):
            floored = v < FLOOR
            h = FLOOR_DRAW if floored else v
            ax.bar(xi + off, h, width=w, color=col, edgecolor="white" if not floored else col,
                   linewidth=0.6, zorder=3,
                   hatch="///" if floored else None,
                   alpha=1.0)
            if floored:
                # white hatch so the "cut" bar looks different
                ax.bar(xi + off, h, width=w, facecolor="none", edgecolor="white",
                       linewidth=0, hatch="///", zorder=3.5)
        if vo < 0.5:
            for off, v, tcol in ((-w / 2, vo, OLD_D), (w / 2, vn, NEW_D)):
                if v < FLOOR:
                    # cut bar: true value written flat just under the cut end
                    ax.text(xi + off, FLOOR_DRAW - 0.04, fmtv(v), ha="center", va="top",
                            fontsize=10.5, color=tcol, fontweight="bold", zorder=5)
                elif v >= 0:
                    ax.text(xi + off, v + 0.03, fmt(v), rotation=90, ha="center", va="bottom",
                            fontsize=10.5, color=tcol, fontweight="bold", zorder=5)
                elif v - 0.8 > R2_YMIN:
                    ax.text(xi + off, v - 0.03, fmt(v), rotation=90, ha="center", va="top",
                            fontsize=10.5, color=tcol, fontweight="bold", zorder=5)
                else:
                    # long negative bar: write the value inside the bar
                    ax.text(xi + off, -0.04, fmt(v), rotation=90, ha="center", va="top",
                            fontsize=10.5, color="white", fontweight="bold", zorder=5)
    ax.axhline(0, color=TRUTH, lw=1.0, zorder=2)
    ax.axhline(FLOOR, color=GREYC, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.set_ylim(R2_YMIN, R2_YMAX)
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([fmt(t, 1) if t != 0 else "0" for t in ticks])
    style_x(ax, a)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(panel_title(s))
    if left:
        ax.set_ylabel("R\u00b2 (1 = perfect)")


def make_r2(D, prot):
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.0), squeeze=False)
    for r, row in enumerate([MAIN, EXTRA]):
        for c, s in enumerate(row):
            draw_r2(axes[r, c], s, D[(s, prot)], left=(c == 0), bottom=(r == 1))
    for ax in axes[1, :]:
        bottom_xlabel(ax)
    handles = [Patch(color=OLD, label="old"), Patch(color=NEW, label="new"),
               Patch(facecolor=GREYC, edgecolor="white", hatch="///",
                     label=f"below {MINUS}1 (bar cut, true value printed)")]
    fig.suptitle(PROT_TITLE[prot], fontsize=16, fontweight="bold", y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.95), fontsize=12, handlelength=1.6)
    fig.text(0.5, -0.01,
             "Both scored against the same answer key (measured torque \u2212 passive torque found inside the trial).  "
             "Values printed where old R\u00b2 < 0.5.  YES, JM, EG: average of 3 runs.",
             ha="center", fontsize=11, color=SEC)
    layout(fig)
    p = os.path.join(OUT, f"results_r2_{prot}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------- error per angle
def draw_err(ax, s, d, left):
    a = d["angles"]
    x = np.arange(len(a))
    w = 0.4
    allv = np.concatenate([d["old_rmse"], d["new_rmse"]])
    med = np.median(allv)
    base = allv[allv <= 4 * med].max()

    def is_out(v):  # only truly off-scale bars are cut
        return v > 3 * base

    top = max(v for v in allv if not is_out(v))
    ytop = top * 1.45
    for off, vals, col in ((-w / 2, d["old_rmse"], OLD), (w / 2, d["new_rmse"], NEW)):
        for xi, v in zip(x, vals):
            if is_out(v):
                # off-scale bar: drawn past the top of the axis (clipped there), hatched,
                # with a white break band and the true value written in the break
                ax.bar(xi + off, ytop * 1.5, width=w, color=col, edgecolor=col, linewidth=0.6, zorder=3)
                ax.bar(xi + off, ytop * 1.5, width=w, facecolor="none", edgecolor="white",
                       linewidth=0, hatch="///", zorder=3.5)
            else:
                ax.bar(xi + off, v, width=w, color=col, edgecolor="white", linewidth=0.6, zorder=3)
    # print values at the angles where old R2 < 0.5 (same angles as the R2 figure)
    for xi, vo, vn, ro in zip(x, d["old_rmse"], d["new_rmse"], d["old_r2"]):
        if ro < 0.5 or is_out(vo) or is_out(vn):
            for off, v, tcol in ((-w / 2, vo, OLD_D), (w / 2, vn, NEW_D)):
                if is_out(v):
                    ax.text(xi + off, ytop * 0.80, fmtv(v), ha="center", va="center",
                            fontsize=10.5, color=tcol, fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                                      edgecolor="none"))
                else:
                    ax.text(xi + off, v + 0.015 * top, fmt(v), rotation=90, ha="center", va="bottom",
                            fontsize=10.5, color=tcol, fontweight="bold", zorder=5)
    ax.set_ylim(0, ytop)
    style_x(ax, a)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(panel_title(s))
    if left:
        ax.set_ylabel("error (Nm)")


def make_err(D, prot):
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.0), squeeze=False)
    for r, row in enumerate([MAIN, EXTRA]):
        for c, s in enumerate(row):
            draw_err(axes[r, c], s, D[(s, prot)], left=(c == 0))
    for ax in axes[1, :]:
        bottom_xlabel(ax)
    handles = [Patch(color=OLD, label="old"), Patch(color=NEW, label="new"),
               Patch(facecolor=GREYC, edgecolor="white", hatch="///",
                     label="very large (bar cut, true value printed)")]
    fig.suptitle(PROT_TITLE[prot] + "  \u2014  typical error (lower is better)",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.95), fontsize=12, handlelength=1.6)
    fig.text(0.5, -0.01,
             "Error = root-mean-square difference from the same answer key, in Nm.  Each subject has its own scale.  "
             "Values printed where old R\u00b2 < 0.5.",
             ha="center", fontsize=11, color=SEC)
    layout(fig)
    p = os.path.join(OUT, f"results_error_{prot}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------- summary
def summarise(D):
    S = {}
    for s in SUBJ:
        row = {}
        for which in ("old", "new"):
            mean8, r2td, ertd = [], [], []
            for p in PROTS:
                d = D[(s, p)]
                r2 = np.maximum(d[f"{which}_r2"], FLOOR)
                er = d[f"{which}_rmse"]
                td = np.argsort(d["angles"])[:2]
                mean8.append(r2.mean())
                r2td.append(r2[td].mean())
                ertd.append(er[td].mean())
            row[which] = dict(mean8=np.mean(mean8), r2td=np.mean(r2td), ertd=np.mean(ertd),
                              mean8_dir=mean8, r2td_dir=r2td, ertd_dir=ertd)
        S[s] = row
    return S


def make_summary(S):
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 5.0))
    x = np.arange(len(SUBJ))
    w = 0.4
    specs = [
        ("mean8", "Average R\u00b2\nover all 8 angles", "R\u00b2 (1 = perfect)", 2),
        ("r2td", "R\u00b2 at each subject's\n2 most toes-down angles", "R\u00b2 (1 = perfect)", 2),
        ("ertd", "Error at each subject's\n2 most toes-down angles", "error (Nm), lower is better", 2),
    ]
    for ax, (key, title, ylab, nd) in zip(axes, specs):
        vo = np.array([S[s]["old"][key] for s in SUBJ])
        vn = np.array([S[s]["new"][key] for s in SUBJ])
        ax.bar(x - w / 2, vo, width=w, color=OLD, edgecolor="white", linewidth=0.6, zorder=3, label="old")
        ax.bar(x + w / 2, vn, width=w, color=NEW, edgecolor="white", linewidth=0.6, zorder=3, label="new")
        if key == "ertd":
            top = max(vo.max(), vn.max())
            span = top
            ax.set_ylim(0, top * 1.32)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        else:
            lo = min(0, vo.min(), vn.min())
            ax.set_ylim(lo - 0.42 if lo < 0 else 0, 1.25)
            span = 1.25 - ax.get_ylim()[0]
        for off, vals, tcol in ((-w / 2, vo, OLD_D), (w / 2, vn, NEW_D)):
            for xi, v in zip(x, vals):
                if v >= 0:
                    y, va = v + 0.012 * span, "bottom"
                else:
                    y, va = v - 0.012 * span, "top"
                ax.text(xi + off, y, fmt(v, nd), rotation=90, ha="center", va=va,
                        fontsize=10.5, color=tcol, fontweight="bold", zorder=5)
        ax.axhline(0, color=TRUTH, lw=1.0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(SUBJ, fontsize=11.5)
        ax.tick_params(axis="x", length=0)
        ax.axvline(2.5, color=GREYC, lw=0.9, ls=(0, (3, 3)), zorder=1)
        # group labels under the subject names: which subjects have 3 runs vs 1 run
        for xc, txt in ((1.0, "3 runs each"), (4.0, "1 run each")):
            ax.annotate(txt, xy=(xc, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -24), textcoords="offset points",
                        ha="center", va="top", fontsize=11, color=SEC)
        ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=13.5, linespacing=1.15)
        ax.set_ylabel(ylab)
        if key != "ertd":
            yt = [t for t in [-1, -0.5, 0, 0.5, 1] if t >= ax.get_ylim()[0] - 1e-9]
            ax.set_yticks(yt)
            ax.set_yticklabels([fmt(t, 1) if t != 0 else "0" for t in yt])
    handles = [Patch(color=OLD, label="old"), Patch(color=NEW, label="new")]
    fig.suptitle("Old vs new, every subject (both directions averaged)", fontsize=16,
                 fontweight="bold", y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.925), fontsize=12, handlelength=1.6)
    fig.text(0.5, -0.035,
             f"Same answer key for both.  R\u00b2 of each angle floored at {MINUS}1 before averaging.  "
             "YES, JM, EG: average of 3 runs;  IES, CH, HM: 1 run.",
             ha="center", fontsize=11, color=SEC)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.72, bottom=0.12, wspace=0.27)
    p = os.path.join(OUT, "results_summary.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main():
    D = collect()
    S = summarise(D)
    outs = []
    for p in PROTS:
        outs.append(make_r2(D, p))
        outs.append(make_err(D, p))
    outs.append(make_summary(S))
    for o in outs:
        print("saved", o)

    print("\nTABLE subject|dir|angle(rad)|deg|old_r2|new_r2|old_err|new_err")
    win_r2 = win_er = tot = 0
    rows = []
    for s in SUBJ:
        for p in PROTS:
            d = D[(s, p)]
            for i, a in enumerate(d["angles"]):
                o2, n2, oe, ne = d["old_r2"][i], d["new_r2"][i], d["old_rmse"][i], d["new_rmse"][i]
                tot += 1
                win_r2 += int(n2 > o2)
                win_er += int(ne < oe)
                line = f"{s}|{p}|{a:+.3f}|{a * DEG:+.0f}|{o2:.3f}|{n2:.3f}|{oe:.3f}|{ne:.3f}"
                rows.append(line)
                print(line)
    print(f"\nnew beats old: R2 {win_r2}/{tot}, error {win_er}/{tot}")
    print("\nCLOSE CALLS / NON-WINS (5 decimals): subj dir angle | old_r2 new_r2 | old_err new_err | r2_win err_win")
    nonwins = []
    for s in SUBJ:
        for p in PROTS:
            d = D[(s, p)]
            for i, a in enumerate(d["angles"]):
                o2, n2, oe, ne = d["old_r2"][i], d["new_r2"][i], d["old_rmse"][i], d["new_rmse"][i]
                if not (n2 > o2) or not (ne < oe) or abs(n2 - o2) < 0.002:
                    line = (f"{s} {p} {a:+.3f} | {o2:.5f} {n2:.5f} | {oe:.5f} {ne:.5f} | "
                            f"{bool(n2 > o2)} {bool(ne < oe)}")
                    nonwins.append(line)
                    print(line)
    # per subject per direction
    print("\nPER SUBJECT (dir-level and both-dir mean)")
    summ = {}
    for s in SUBJ:
        so, sn = S[s]["old"], S[s]["new"]
        line = (f"{s}: mean8 old {so['mean8']:.3f} new {sn['mean8']:.3f} | "
                f"R2 2-toes-down old {so['r2td']:.3f} new {sn['r2td']:.3f} | "
                f"err 2-toes-down old {so['ertd']:.3f} new {sn['ertd']:.3f} | "
                f"per dir S1toS2/S2toS1 mean8 old {so['mean8_dir'][0]:.3f}/{so['mean8_dir'][1]:.3f} "
                f"new {sn['mean8_dir'][0]:.3f}/{sn['mean8_dir'][1]:.3f}; "
                f"r2td old {so['r2td_dir'][0]:.3f}/{so['r2td_dir'][1]:.3f} new {sn['r2td_dir'][0]:.3f}/{sn['r2td_dir'][1]:.3f}; "
                f"errtd old {so['ertd_dir'][0]:.3f}/{so['ertd_dir'][1]:.3f} new {sn['ertd_dir'][0]:.3f}/{sn['ertd_dir'][1]:.3f}")
        print(line)
        summ[s] = line
    # per subject win counts
    print("\nWINS per subject (R2, error) out of 16")
    for s in SUBJ:
        wr = we = 0
        for p in PROTS:
            d = D[(s, p)]
            wr += int((d["new_r2"] > d["old_r2"]).sum())
            we += int((d["new_rmse"] < d["old_rmse"]).sum())
        print(f"{s}: R2 {wr} of 16, error {we} of 16")
    # all-subject means
    for which in ("old", "new"):
        print(which, "mean over 6 subjects: mean8 %.3f r2td %.3f" % (
            np.mean([S[s][which]["mean8"] for s in SUBJ]),
            np.mean([S[s][which]["r2td"] for s in SUBJ])))
    with open(os.path.join(HERE, "F8_results_numbers.json"), "w") as f:
        json.dump({"rows": rows, "summary": summ, "win_r2": win_r2, "win_err": win_er, "total": tot, "close_calls": nonwins},
                  f, indent=1)


if __name__ == "__main__":
    main()
