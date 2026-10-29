"""y06: results of the TECHNIQUES (chapters c13, c16, c17) in the dashboard's format.

Reads  : EXPERIMENTS/dashboard_react/src/data/results.json   (one row per experiment x subject x direction x seed x position)
         DWT/EXPERIMENTS/results/presentation_runs.csv        (full-precision copy of the same scores, read only for a cross-check)
         EXPERIMENTS/pipeline_video/data2/facts_y01.json      (read only for a cross-check of the D01 / X00 group numbers)
Writes : EXPERIMENTS/pipeline_video/data2/y06.npz
         EXPERIMENTS/pipeline_video/data2/facts_y06.json      (one top-level key: "y06")
Run    : .venv311/Scripts/python.exe EXPERIMENTS/pipeline_video/extract2/y06_techniques.py     (numpy / pandas / json only)

Dashboard format (memory 'results summary format'): direction S1toS2 only; per run R2 floored at 0 (R2 < 0 shown as 0);
per cell (person x position) the MEDIAN over seeds; then the MEDIAN over cells, separately for toes-down cells
(each person's 2 most negative positions, flag toes_down=1 in results.json) and for the other 6 positions.
Whenever two experiments are compared, only the seeds present in BOTH are used (per person).
Angle convention: negative = plantarflexion = toes down.

Comparisons (key in facts / prefix in the npz)
  group_numbers : X00_baseline (10 seeds), X00f (3 seeds), X00 restricted to seeds 0-2, D01 (C01 for EG and YES; 10 seeds, and 0-2)
  x05           : X05_autogain          vs X00f_baseline_framework  (6 people, seeds 0-2)
  x04           : X04_equal_angle_weight vs X00f                    (6 people, seeds 0-2)
  x02_*         : dropout 0 / 0.1 / 0.5  vs X00f  (YES, EG; seeds 0-1)       -> npz drop_*
  x06           : X06_window_0.4s        vs X00f  (YES, EG; seeds 0-1)
  d02           : D02_D01+window_0.5s    vs D01 (C01 for EG, YES)  (6 people, seeds 0-2)
  d01_vs_x00    : D01 (C01 for EG, YES)  vs X00_baseline (6 people, seeds 0-9), incl. per-cell R2 range over seeds
  yes_minus0p486: YES at -0.486 rad, per-seed R2 UNFLOORED (X05 vs X00f seeds 0-2; X00, C01 over 10 seeds)
Every comparison is also re-computed from the full-precision presentation_runs.csv and the largest difference is reported (checks).
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
VWT = HERE.parents[2]
RESULTS = VWT / "EXPERIMENTS/dashboard_react/src/data/results.json"
DWT = pathlib.Path("C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137")
PRES = DWT / "EXPERIMENTS/results/presentation_runs.csv"
Y01 = VWT / "EXPERIMENTS/pipeline_video/data2/facts_y01.json"
OUT_DIR = VWT / "EXPERIMENTS/pipeline_video/data2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJ = ["CH", "EG", "HM", "IES", "JM", "YES"]
DIR = "S1toS2"
X00, X00F = "X00_baseline", "X00f_baseline_framework"
X05, X04 = "X05_autogain", "X04_equal_angle_weight"
X02 = {0.0: "X02a_dropout_0.0", 0.1: "X02b_dropout_0.1", 0.5: "X02c_dropout_0.5"}
X06 = "X06_window_0.4s"
D01, C01, D02 = "D01_cleanpassive+autogain+equalweight", "C01_autogain+equalweight", "D02_D01+window_0.5s"

checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("OK   " if ok else "FAIL ") + name + (" | " + str(detail) if detail != "" else ""))


def med(x):
    return float(np.median(np.asarray(x, float)))


# ------------------------------------------------------------------ data
raw = json.load(open(RESULTS))
df = pd.DataFrame(raw["rows"], columns=raw["cols"])
df["pos"] = df.position_rad.round(3)
df["r2f"] = df.r2.clip(lower=0)

pr = pd.read_csv(PRES, encoding="utf-8-sig")
pr = pr.rename(columns={"model": "experiment"})
pr["experiment"] = pr.experiment.replace({"original": X00})
pr = pr[~pr.experiment.isin(["original_fw", "ours"])].copy()      # aliases of X00f / D01+C01
pr["pos"] = pr.position_rad.round(3)
pr["r2f"] = pr.r2.clip(lower=0)


def sel_all(name):
    return {s: name for s in SUBJ}


D01_SEL = {s: (C01 if s in ("EG", "YES") else D01) for s in SUBJ}      # dashboard convention for the final candidate


def build(d, sel, subjects):
    parts = [d[(d.experiment == sel[s]) & (d.subject == s) & (d.direction == DIR)] for s in subjects]
    return pd.concat(parts) if parts else d.iloc[:0]


def cell_frame(v):
    rows = []
    for (s, p), g in v.groupby(["subject", "pos"]):
        r2 = g.r2.values
        rf = np.clip(r2, 0, None)
        rows.append(dict(subject=s, pos=float(p), toes_down=int(g.toes_down.iloc[0]), n_seeds=int(len(g)),
                         r2f=med(rf), r2_raw=med(r2), mae=med(g.mae), r2_min=float(r2.min()), r2_max=float(r2.max()),
                         r2_range=float(r2.max() - r2.min()), r2f_range=float(rf.max() - rf.min()),
                         mae_range=float(g.mae.max() - g.mae.min())))
    c = pd.DataFrame(rows)
    c["si"] = c.subject.map({s: i for i, s in enumerate(SUBJ)})
    return c.sort_values(["si", "pos"]).reset_index(drop=True)


def group(c):
    out = {}
    for name, m in [("toes_down", c.toes_down == 1), ("other", c.toes_down == 0)]:
        x = c[m]
        out[name] = dict(n_cells=int(len(x)), r2=med(x.r2f), mae=med(x.mae), r2_raw_median=med(x.r2_raw),
                         r2_range_median=med(x.r2_range), r2f_range_median=med(x.r2f_range), mae_range_median=med(x.mae_range))
    return out


def per_subject(c):
    out = {}
    for s in SUBJ:
        x = c[c.subject == s]
        if len(x) == 0:
            continue
        t, o = x[x.toes_down == 1], x[x.toes_down == 0]
        out[s] = dict(td_r2=med(t.r2f), td_mae=med(t.mae), td_r2_raw=med(t.r2_raw), other_r2=med(o.r2f), other_mae=med(o.mae),
                      n_td_cells=int(len(t)), n_other_cells=int(len(o)))
    return out


def compare(d, selA, selB, subjects=None, seeds=None):
    """A = 'before', B = 'after'. Per person, only seeds that exist in both (and in `seeds` if given)."""
    subjects = subjects or SUBJ
    a, b = build(d, selA, subjects), build(d, selB, subjects)
    subs = [s for s in subjects if (a.subject == s).any() and (b.subject == s).any()]
    ka, kb = [], []
    used = {}
    for s in subs:
        common = sorted(set(a[a.subject == s].seed) & set(b[b.subject == s].seed))
        if seeds is not None:
            common = [x for x in common if x in seeds]
        used[s] = [int(x) for x in common]
        ka.append(a[(a.subject == s) & a.seed.isin(common)])
        kb.append(b[(b.subject == s) & b.seed.isin(common)])
    a, b = pd.concat(ka), pd.concat(kb)
    ca, cb = cell_frame(a), cell_frame(b)
    assert list(zip(ca.subject, ca.pos)) == list(zip(cb.subject, cb.pos)), "cell grids differ between the two experiments"
    assert (ca.toes_down.values == cb.toes_down.values).all()
    return dict(subjects=subs, seeds_used=used, ca=ca, cb=cb, ga=group(ca), gb=group(cb), psa=per_subject(ca), psb=per_subject(cb))


def pub(r):
    """JSON-safe view of a compare() result (no frames)."""
    return dict(subjects=r["subjects"], seeds_used=r["seeds_used"], before=r["ga"], after=r["gb"],
                per_subject_before=r["psa"], per_subject_after=r["psb"], n_cells=int(len(r["ca"])))


def flat(r):
    """numbers of a comparison, for the precision cross-check"""
    o = {}
    for tag, g in [("a", r["ga"]), ("b", r["gb"])]:
        for grp, vals in g.items():
            for k in ("r2", "mae"):
                o[f"{tag}.{grp}.{k}"] = vals[k]
    for tag, p in [("a", r["psa"]), ("b", r["psb"])]:
        for s, vals in p.items():
            for k in ("td_r2", "td_mae", "other_r2", "other_mae"):
                o[f"{tag}.{s}.{k}"] = vals[k]
    return o


# ------------------------------------------------------------------ data sanity
check("results.json loaded", len(df) > 0, f"{len(df)} rows, built {raw.get('built')}")
grid_ok, flag_ok = True, True
ref = {}
for (e, s), g in df[df.direction == DIR].groupby(["experiment", "subject"]):
    pos = tuple(sorted(g.pos.unique()))
    ref.setdefault(s, pos)
    grid_ok &= (len(pos) == 8 and bool(np.allclose(pos, ref[s], atol=0.006)))
    fl = g.groupby("pos").toes_down.first()
    flag_ok &= [int(fl[p]) for p in pos] == [1, 1, 0, 0, 0, 0, 0, 0]
check("8 positions per person in every experiment, same grid to within 0.006 rad", grid_ok)
check("toes_down flag = the 2 most negative positions in every experiment x person", flag_ok)

# ------------------------------------------------------------------ group numbers
x00_all = cell_frame(build(df, sel_all(X00), SUBJ))
x00f_all = cell_frame(build(df, sel_all(X00F), SUBJ))
x00_s02 = cell_frame(build(df, sel_all(X00), SUBJ).query("seed < 3"))
d01_all = cell_frame(build(df, D01_SEL, SUBJ))
d01_s02 = cell_frame(build(df, D01_SEL, SUBJ).query("seed < 3"))
GRP = {"X00_baseline_seeds0-9": group(x00_all), "X00f_baseline_framework_seeds0-2": group(x00f_all),
       "X00_baseline_seeds0-2": group(x00_s02), "D01_with_C01_for_EG_YES_seeds0-9": group(d01_all),
       "D01_with_C01_for_EG_YES_seeds0-2": group(d01_s02)}
for k, v in GRP.items():
    check(f"group cell counts {k} (12 toes-down + 36 other)", v["toes_down"]["n_cells"] == 12 and v["other"]["n_cells"] == 36)

# ------------------------------------------------------------------ comparisons
R = {}
R["x05"] = compare(df, sel_all(X00F), sel_all(X05))
R["x04"] = compare(df, sel_all(X00F), sel_all(X04))
for rate, e in X02.items():
    R[f"x02_{rate}"] = compare(df, sel_all(X00F), sel_all(e), subjects=["YES", "EG"])
R["x06"] = compare(df, sel_all(X00F), sel_all(X06), subjects=["YES", "EG"])
R["d02"] = compare(df, D01_SEL, sel_all(D02))
R["d01_vs_x00"] = compare(df, sel_all(X00), D01_SEL)
R["x00_vs_x00f_seeds0-2"] = compare(df, sel_all(X00), sel_all(X00F))
R["d01_vs_x05_seeds0-2"] = compare(df, sel_all(X05), D01_SEL, seeds=[0, 1, 2])    # context only

check("x05/x04/d02 use 6 people x seeds 0-2",
      all(R[k]["subjects"] == SUBJ and all(v == [0, 1, 2] for v in R[k]["seeds_used"].values()) for k in ("x05", "x04", "d02")))
check("dropout and X06 comparisons use YES+EG x seeds 0-1",
      all(sorted(R[k]["subjects"]) == ["EG", "YES"] and all(v == [0, 1] for v in R[k]["seeds_used"].values()) for k in ("x02_0.0", "x02_0.1", "x02_0.5", "x06")))
check("d01_vs_x00 uses 6 people x seeds 0-9", R["d01_vs_x00"]["subjects"] == SUBJ and all(v == list(range(10)) for v in R["d01_vs_x00"]["seeds_used"].values()))

# ------------------------------------------------------------------ cross-check with the full-precision CSV
RP = {}
RP["x05"] = compare(pr, sel_all(X00F), sel_all(X05))
RP["x04"] = compare(pr, sel_all(X00F), sel_all(X04))
for rate, e in X02.items():
    RP[f"x02_{rate}"] = compare(pr, sel_all(X00F), sel_all(e), subjects=["YES", "EG"])
RP["x06"] = compare(pr, sel_all(X00F), sel_all(X06), subjects=["YES", "EG"])
RP["d02"] = compare(pr, D01_SEL, sel_all(D02))
RP["d01_vs_x00"] = compare(pr, sel_all(X00), D01_SEL)
prec = {}
for k in RP:
    f1, f2 = flat(R[k]), flat(RP[k])
    prec[k] = max(abs(f1[n] - f2[n]) for n in f1)
check("results.json (4 decimals) vs presentation_runs.csv (full precision): all group and per-person numbers within 0.001",
      max(prec.values()) < 1e-3, {k: round(v, 6) for k, v in prec.items()})

y01 = json.load(open(Y01))["B_group_summaries"] if Y01.exists() else None
if y01:
    for key, ref_key in [("X00_baseline_seeds0-9", "X00_baseline"), ("D01_with_C01_for_EG_YES_seeds0-9", "D01_with_C01_for_EG_YES"),
                         ("X00f_baseline_framework_seeds0-2", "X00f_baseline_framework")]:
        g, yy = GRP[key], y01[ref_key]
        diffs = [abs(g["toes_down"]["r2"] - yy["toes_down_12"]["r2_floor0"]), abs(g["toes_down"]["mae"] - yy["toes_down_12"]["mae_nm"]),
                 abs(g["other"]["r2"] - yy["other_36"]["r2_floor0"]), abs(g["other"]["mae"] - yy["other_36"]["mae_nm"])]
        check(f"group numbers {key} agree with facts_y01 B (4-decimal rounding)", max(diffs) < 1.5e-4, f"max diff {max(diffs):.2e}")

# ------------------------------------------------------------------ D01 vs X00 cells, YES -0.486
cells_x00, cells_d01 = R["d01_vs_x00"]["ca"], R["d01_vs_x00"]["cb"]
yes_pos = sorted(df[(df.subject == "YES") & (df.direction == DIR)].pos.unique())
YES_M486 = [p for p in yes_pos if abs(p + 0.486) < 0.002]
check("YES has exactly one position near -0.486 rad", len(YES_M486) == 1, YES_M486)
P486 = YES_M486[0]
yes_td = sorted(df[(df.subject == "YES") & (df.toes_down == 1)].pos.unique())
check("YES -0.486 is one of YES's two toes-down positions", P486 in yes_td, yes_td)


def runs(exp, subject, pos, seeds):
    g = df[(df.experiment == exp) & (df.subject == subject) & (df.direction == DIR) & (df.pos == pos) & df.seed.isin(seeds)].sort_values("seed")
    assert list(g.seed) == list(seeds), (exp, list(g.seed))
    return g


yes486 = {}
for lab, e, seeds in [("X05", X05, [0, 1, 2]), ("X00f", X00F, [0, 1, 2]), ("X04", X04, [0, 1, 2]),
                      ("X00", X00, list(range(10))), ("C01", C01, list(range(10))), ("D02", D02, [0, 1, 2])]:
    g = runs(e, "YES", P486, seeds)
    v = [float(x) for x in g.r2]
    yes486[lab] = dict(seeds=seeds, r2_per_seed=v, median=med(v), median_floor0=med(np.clip(v, 0, None)), mae_per_seed=[float(x) for x in g.mae])
yes486["C01_seeds0-2_median"] = med(yes486["C01"]["r2_per_seed"][:3])
yes486["X00_seeds0-2_median"] = med(yes486["X00"]["r2_per_seed"][:3])
yes486["position_rad"] = P486


# ------------------------------------------------------------------ JM per position (X04 and X05 vs X00f)
def jm_table(r):
    a, b = r["ca"][r["ca"].subject == "JM"], r["cb"][r["cb"].subject == "JM"]
    return [dict(position_rad=float(x.pos), toes_down=int(x.toes_down), r2_floor0_before=float(x.r2f), r2_floor0_after=float(y.r2f),
                 r2_raw_before=float(x.r2_raw), r2_raw_after=float(y.r2_raw), mae_before=float(x.mae), mae_after=float(y.mae))
            for x, y in zip(a.itertuples(), b.itertuples())]


JM = {"x04": jm_table(R["x04"]), "x05": jm_table(R["x05"])}
jm_m36 = [t for t in JM["x04"] if abs(t["position_rad"] + 0.36) < 0.01]
check("JM has exactly one position near -0.36 rad", len(jm_m36) == 1, [t["position_rad"] for t in jm_m36])
JM_P36 = jm_m36[0]["position_rad"]

ch_x05 = R["x05"]["psa"]["CH"], R["x05"]["psb"]["CH"]
hm_x05 = R["x05"]["psa"]["HM"], R["x05"]["psb"]["HM"]

# ------------------------------------------------------------------ storyboard claims
claims = []


def claim(label, computed, stated, dp):
    ok = bool(np.isfinite(computed) and abs(round(computed, dp) - stated) < 1e-9)
    claims.append(dict(label=label, computed=float(computed), computed_rounded=round(float(computed), dp), storyboard=stated, ok=ok))


g05a, g05b = R["x05"]["ga"], R["x05"]["gb"]
claim("X05 toes-down R2 before (X00f)", g05a["toes_down"]["r2"], 0.81, 2)
claim("X05 toes-down R2 after", g05b["toes_down"]["r2"], 0.90, 2)
claim("X05 toes-down MAE before", g05a["toes_down"]["mae"], 0.79, 2)
claim("X05 toes-down MAE after", g05b["toes_down"]["mae"], 0.58, 2)
claim("X05 other R2 before", g05a["other"]["r2"], 0.971, 3)
claim("X05 other R2 after", g05b["other"]["r2"], 0.977, 3)
claim("X05 other MAE before", g05a["other"]["mae"], 1.01, 2)
claim("X05 other MAE after", g05b["other"]["mae"], 0.74, 2)
for s, (b0, b1) in {"EG": (0.42, 0.93), "JM": (0.60, 0.84), "IES": (0.68, 0.90), "YES": (0.37, 0.43)}.items():
    claim(f"X05 {s} toes-down R2 before", R["x05"]["psa"][s]["td_r2"], b0, 2)
    claim(f"X05 {s} toes-down R2 after", R["x05"]["psb"][s]["td_r2"], b1, 2)
g04a, g04b = R["x04"]["ga"], R["x04"]["gb"]
claim("X04 toes-down R2 before", g04a["toes_down"]["r2"], 0.81, 2)
claim("X04 toes-down R2 after", g04b["toes_down"]["r2"], 0.87, 2)
claim("X04 toes-down MAE before", g04a["toes_down"]["mae"], 0.79, 2)
claim("X04 toes-down MAE after", g04b["toes_down"]["mae"], 0.54, 2)
claim("X04 JM toes-down R2 before", R["x04"]["psa"]["JM"]["td_r2"], 0.60, 2)
claim("X04 JM toes-down R2 after", R["x04"]["psb"]["JM"]["td_r2"], 0.47, 2)
claim(f"X04 JM {JM_P36} rad R2 before", jm_m36[0]["r2_floor0_before"], 0.40, 2)
claim(f"X04 JM {JM_P36} rad R2 after", jm_m36[0]["r2_floor0_after"], 0.03, 2)
drop_x = {0.0: R["x02_0.0"], 0.1: R["x02_0.1"], 0.5: R["x02_0.5"]}
claim("dropout 0.0 toes-down MAE", drop_x[0.0]["gb"]["toes_down"]["mae"], 1.12, 2)
claim("dropout 0.1 toes-down MAE", drop_x[0.1]["gb"]["toes_down"]["mae"], 1.35, 2)
claim("dropout 0.3 (X00f) toes-down MAE", drop_x[0.0]["ga"]["toes_down"]["mae"], 0.93, 2)
claim("dropout 0.5 toes-down MAE", drop_x[0.5]["gb"]["toes_down"]["mae"], 1.25, 2)
claim("X06 toes-down R2 before (X00f YES+EG seeds 0-1)", R["x06"]["ga"]["toes_down"]["r2"], 0.429, 3)
claim("X06 toes-down R2 after", R["x06"]["gb"]["toes_down"]["r2"], 0.497, 3)
claim("X06 toes-down MAE before", R["x06"]["ga"]["toes_down"]["mae"], 0.931, 3)
claim("X06 toes-down MAE after", R["x06"]["gb"]["toes_down"]["mae"], 1.151, 3)
claim("D02 vs D01 toes-down MAE before (D01)", R["d02"]["ga"]["toes_down"]["mae"], 0.455, 3)
claim("D02 vs D01 toes-down MAE after (D02)", R["d02"]["gb"]["toes_down"]["mae"], 0.396, 3)
claim("D02 vs D01 toes-down R2 before", R["d02"]["ga"]["toes_down"]["r2"], 0.939, 3)
claim("D02 vs D01 toes-down R2 after", R["d02"]["gb"]["toes_down"]["r2"], 0.944, 3)
gx, gd = R["d01_vs_x00"]["ga"], R["d01_vs_x00"]["gb"]
claim("D01 vs X00 toes-down R2 before", gx["toes_down"]["r2"], 0.769, 3)
claim("D01 vs X00 toes-down R2 after", gd["toes_down"]["r2"], 0.937, 3)
claim("D01 vs X00 toes-down MAE before", gx["toes_down"]["mae"], 0.833, 3)
claim("D01 vs X00 toes-down MAE after", gd["toes_down"]["mae"], 0.444, 3)
claim("D01 vs X00 other R2 before", gx["other"]["r2"], 0.972, 3)
claim("D01 vs X00 other R2 after", gd["other"]["r2"], 0.981, 3)
claim("D01 vs X00 other MAE before", gx["other"]["mae"], 1.072, 3)
claim("D01 vs X00 other MAE after", gd["other"]["mae"], 0.667, 3)
claim("X00 toes-down seed range (raw R2 max-min, median over cells)", gx["toes_down"]["r2_range_median"], 0.423, 3)
claim("D01 toes-down seed range (raw R2 max-min, median over cells)", gd["toes_down"]["r2_range_median"], 0.026, 3)
claim("X00 other seed range (raw R2)", gx["other"]["r2_range_median"], 0.019, 3)
for sd, st in zip([0, 1, 2], [0.16, -1.44, -3.16]):
    claim(f"YES {P486} X05 seed {sd} R2", yes486["X05"]["r2_per_seed"][sd], st, 2)
for sd, st in zip([0, 1, 2], [-0.52, 0.12, -1.36]):
    claim(f"YES {P486} X00f seed {sd} R2", yes486["X00f"]["r2_per_seed"][sd], st, 2)
claim("C01 YES -0.486 median R2 over 10 seeds", yes486["C01"]["median"], -0.97, 2)
for c in claims:
    check("storyboard: " + c["label"], c["ok"], f"computed {c['computed']:.4f} -> {c['computed_rounded']} vs storyboard {c['storyboard']}")
mismatches = [f"{c['label']}: computed {c['computed']:.4f} (rounds to {c['computed_rounded']}), storyboard says {c['storyboard']}" for c in claims if not c["ok"]]

# ------------------------------------------------------------------ arrays for bar charts
npz = {"subjects": np.array(SUBJ)}


def put_comp(prefix, r):
    subs = r["subjects"]
    npz[f"{prefix}_subjects"] = np.array(subs)
    for tag, ps in [("before", r["psa"]), ("after", r["psb"])]:
        for k in ("td_r2", "td_mae", "other_r2", "other_mae", "td_r2_raw"):
            npz[f"{prefix}_{k}_{tag}"] = np.array([ps[s][k] for s in subs])
    # group table: [toes_down, other] x [R2, MAE] x [before, after]
    npz[f"{prefix}_group"] = np.array([[[r["ga"][g][m], r["gb"][g][m]] for m in ("r2", "mae")] for g in ("toes_down", "other")])
    ca, cb = r["ca"], r["cb"]
    npz[f"{prefix}_cell_subject_idx"] = ca.subject.map({s: i for i, s in enumerate(SUBJ)}).values.astype(int)
    npz[f"{prefix}_cell_pos"] = ca.pos.values
    npz[f"{prefix}_cell_toes_down"] = ca.toes_down.values.astype(int)
    npz[f"{prefix}_cell_r2_floor0_before"], npz[f"{prefix}_cell_r2_floor0_after"] = ca.r2f.values, cb.r2f.values
    npz[f"{prefix}_cell_r2_raw_before"], npz[f"{prefix}_cell_r2_raw_after"] = ca.r2_raw.values, cb.r2_raw.values
    npz[f"{prefix}_cell_mae_before"], npz[f"{prefix}_cell_mae_after"] = ca.mae.values, cb.mae.values
    npz[f"{prefix}_cell_r2_range_before"], npz[f"{prefix}_cell_r2_range_after"] = ca.r2_range.values, cb.r2_range.values


put_comp("x05", R["x05"])
put_comp("x04", R["x04"])
put_comp("x06", R["x06"])
put_comp("d02", R["d02"])
put_comp("d01_vs_x00", R["d01_vs_x00"])
put_comp("x00_vs_x00f", R["x00_vs_x00f_seeds0-2"])
for k in ("x04", "x05"):
    t = JM[k]
    npz[f"jm_{k}_pos"] = np.array([x["position_rad"] for x in t])
    for key in ("r2_floor0_before", "r2_floor0_after", "r2_raw_before", "r2_raw_after", "mae_before", "mae_after"):
        npz[f"jm_{k}_{key}"] = np.array([x[key] for x in t])
# dropout bars: order 0, 0.1, 0.3 (baseline X00f), 0.5; YES+EG, seeds 0-1
order = [(drop_x[0.0], "gb"), (drop_x[0.1], "gb"), (drop_x[0.0], "ga"), (drop_x[0.5], "gb")]
npz["drop_rates"] = np.array([0.0, 0.1, 0.3, 0.5])
for grp, nm in [("toes_down", "td"), ("other", "other")]:
    for m in ("mae", "r2"):
        npz[f"drop_{nm}_{m}"] = np.array([r[tag][grp][m] for r, tag in order])
set_names = list(GRP)
npz["grp_set_names"] = np.array(set_names)
npz["grp_table_cols"] = np.array(["td_r2", "td_mae", "other_r2", "other_mae", "td_r2_range_raw", "other_r2_range_raw"])
npz["grp_table"] = np.array([[GRP[n]["toes_down"]["r2"], GRP[n]["toes_down"]["mae"], GRP[n]["other"]["r2"], GRP[n]["other"]["mae"],
                              GRP[n]["toes_down"]["r2_range_median"], GRP[n]["other"]["r2_range_median"]] for n in set_names])
npz["yes_m486_pos"] = np.array(P486)
for lab in ("X05", "X00f", "X04", "X00", "C01", "D02"):
    npz[f"yes_m486_r2_{lab}"] = np.array(yes486[lab]["r2_per_seed"])
    npz[f"yes_m486_mae_{lab}"] = np.array(yes486[lab]["mae_per_seed"])
npz["d01x00_cell_r2_range_x00"], npz["d01x00_cell_r2_range_d01"] = cells_x00.r2_range.values, cells_d01.r2_range.values
npz["d01x00_cell_r2f_range_x00"], npz["d01x00_cell_r2f_range_d01"] = cells_x00.r2f_range.values, cells_d01.r2f_range.values
npz["d01x00_cell_r2_min_x00"], npz["d01x00_cell_r2_max_x00"] = cells_x00.r2_min.values, cells_x00.r2_max.values
npz["d01x00_cell_r2_min_d01"], npz["d01x00_cell_r2_max_d01"] = cells_d01.r2_min.values, cells_d01.r2_max.values
np.savez(OUT_DIR / "y06.npz", **npz)

# ------------------------------------------------------------------ extras
extra = {}
extra["x05_CH_unchanged"] = dict(before=ch_x05[0], after=ch_x05[1], td_r2_diff=ch_x05[1]["td_r2"] - ch_x05[0]["td_r2"])
extra["x05_HM"] = dict(before=hm_x05[0], after=hm_x05[1], td_r2_diff=hm_x05[1]["td_r2"] - hm_x05[0]["td_r2"])
extra["x04_per_person_td_r2"] = {s: [R["x04"]["psa"][s]["td_r2"], R["x04"]["psb"][s]["td_r2"]] for s in SUBJ}
extra["x05_per_person_td_r2"] = {s: [R["x05"]["psa"][s]["td_r2"], R["x05"]["psb"][s]["td_r2"]] for s in SUBJ}
extra["x05_per_person_td_mae"] = {s: [R["x05"]["psa"][s]["td_mae"], R["x05"]["psb"][s]["td_mae"]] for s in SUBJ}
extra["x04_per_person_td_mae"] = {s: [R["x04"]["psa"][s]["td_mae"], R["x04"]["psb"][s]["td_mae"]] for s in SUBJ}


def yes_cells(r):
    a, b = r["ca"][r["ca"].subject == "YES"], r["cb"][r["cb"].subject == "YES"]
    return [dict(position_rad=float(x.pos), toes_down=int(x.toes_down), r2_floor0_before=float(x.r2f), r2_floor0_after=float(y.r2f),
                 r2_raw_before=float(x.r2_raw), r2_raw_after=float(y.r2_raw), mae_before=float(x.mae), mae_after=float(y.mae))
            for x, y in zip(a.itertuples(), b.itertuples())]


extra["YES_cells_x05"], extra["YES_cells_x04"] = yes_cells(R["x05"]), yes_cells(R["x04"])
extra["seed_range_d01_vs_x00"] = {
    g: dict(x00_raw=gx[g]["r2_range_median"], d01_raw=gd[g]["r2_range_median"], x00_floor0=gx[g]["r2f_range_median"], d01_floor0=gd[g]["r2f_range_median"])
    for g in ("toes_down", "other")}
extra["ratio_td_range_d01_over_x00_raw"] = gd["toes_down"]["r2_range_median"] / gx["toes_down"]["r2_range_median"]
extra["x04_vs_x00f_seed_spread_toes_down_raw"] = dict(
    x00f=g04a["toes_down"]["r2_range_median"], x04=g04b["toes_down"]["r2_range_median"],
    ratio=g04b["toes_down"]["r2_range_median"] / g04a["toes_down"]["r2_range_median"],
    note="3 seeds only; storyboard c10 quotes ~0.6x from the weekly update")

disp = {c["label"]: c["computed_rounded"] for c in claims}
disp.update({
    "yes_m486_x05_seeds": [round(v, 2) for v in yes486["X05"]["r2_per_seed"]],
    "yes_m486_x00f_seeds": [round(v, 2) for v in yes486["X00f"]["r2_per_seed"]],
    "yes_m486_c01_median_10": round(yes486["C01"]["median"], 3),
    "yes_m486_x00_median_10": round(yes486["X00"]["median"], 3),
})

facts = {"y06": dict(
    source=str(RESULTS), results_json_built=raw.get("built"),
    note=("Dashboard format: S1toS2; per run R2 floored at 0; median over seeds per cell (person x position); median over cells, separately for "
          "toes-down (2 most negative positions per person, 12 cells) and other (36 cells). Seeds matched between compared experiments. "
          "r2_range = max - min of the raw (unfloored) R2 over seeds, median over cells (what facts_y01 F calls r2_range). "
          "D01 = D01_cleanpassive+autogain+equalweight for CH, HM, IES, JM and C01_autogain+equalweight for EG, YES. "
          "Dropout 0.3 bar = X00f on YES+EG seeds 0-1."),
    group_numbers=GRP,
    comparisons={k: pub(v) for k, v in R.items()},
    jm_per_position=JM, jm_minus0p36_position=JM_P36,
    yes_minus0p486=yes486,
    extra=extra,
    storyboard_claims=claims,
    precision_crosscheck_max_abs_diff=prec,
    display=disp,
    checks=checks,
    storyboard_mismatches=mismatches,
    npz_arrays={k: list(v.shape) for k, v in npz.items()},
)}
json.dump(facts, open(OUT_DIR / "facts_y06.json", "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))

# ------------------------------------------------------------------ print
print("=" * 90)
for k, v in GRP.items():
    print(f"{k:38s} TD R2 {v['toes_down']['r2']:.4f} MAE {v['toes_down']['mae']:.4f} range {v['toes_down']['r2_range_median']:.4f} | other R2 {v['other']['r2']:.4f} MAE {v['other']['mae']:.4f} range {v['other']['r2_range_median']:.4f}")
for k, r in R.items():
    print(f"\n[{k}] subjects {r['subjects']}")
    for g in ("toes_down", "other"):
        print(f"   {g:10s} R2 {r['ga'][g]['r2']:.4f} -> {r['gb'][g]['r2']:.4f}   MAE {r['ga'][g]['mae']:.4f} -> {r['gb'][g]['mae']:.4f}   raw-range {r['ga'][g]['r2_range_median']:.4f} -> {r['gb'][g]['r2_range_median']:.4f}")
    if k in ("x05", "x04"):
        print("   per-person TD R2:", {s: (round(r['psa'][s]['td_r2'], 3), round(r['psb'][s]['td_r2'], 3)) for s in r['subjects']})
print("\nJM per position (X04):")
print(pd.DataFrame(JM["x04"]).round(3).to_string(index=False))
print("\nYES -0.486:", json.dumps({k: (v["r2_per_seed"] if isinstance(v, dict) else v) for k, v in yes486.items()}))
print("\nMISMATCHES:", *mismatches, sep="\n  ")
print("\nchecks failed:", [c["check"] for c in checks if not c["ok"]])
print("wrote", OUT_DIR / "y06.npz", "and", OUT_DIR / "facts_y06.json")
