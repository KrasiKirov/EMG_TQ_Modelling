"""y01: how well does the LAST-SEMESTER BASELINE (X00_baseline, original LSTM-8 code) actually score?

Reads  : EXPERIMENTS/dashboard_react/src/data/results.json   (rows: one per experiment x subject x direction x seed x position)
         DWT/EXPERIMENTS/results/learning_curves/<exp>/<subject>/S1toS2/seed0.json
Writes : EXPERIMENTS/pipeline_video/data2/facts_y01.json
Run    : .venv311/Scripts/python.exe y01_baseline_scores.py     (numpy / pandas / json only)

Sections
  A  baseline per subject x position (10 seeds)
  B  group summaries exactly like the dashboard (median over seeds per cell, then median over cells)
  C  cells with median R2 < 0.5 ("shape right, offset wrong") + best single-run examples
  D  offset-only reference: R2 of a perfect-shape prediction shifted by the run's bias
  E  learning curves (fit / val / test MAE at the kept epoch)
  F  seed-to-seed spread
"""
import json
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
VWT = HERE.parents[2]                                   # .../emg-torque-explainer-video-cbc2c4
RESULTS = VWT / "EXPERIMENTS/dashboard_react/src/data/results.json"
DWT = pathlib.Path("C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137")
LC = DWT / "EXPERIMENTS/results/learning_curves"
OUT = VWT / "EXPERIMENTS/pipeline_video/data2/facts_y01.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

SUBJ = ["CH", "EG", "HM", "IES", "JM", "YES"]
BASE = "X00_baseline"
FRAMEWORK = "X00f_baseline_framework"
D01 = "D01_cleanpassive+autogain+equalweight"
C01 = "C01_autogain+equalweight"
DIR = "S1toS2"


def r4(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def med(x):
    return float(np.median(x))


def pct(x, q):
    return float(np.percentile(x, q))  # numpy default = linear interpolation


raw = json.load(open(RESULTS))
df = pd.DataFrame(raw["rows"], columns=raw["cols"])
df["pos"] = df.position_rad.round(3)
df["r2f"] = df.r2.clip(lower=0)             # dashboard display rule: R2 < 0 shown as 0
facts = {"source": str(RESULTS), "results_json_built": raw.get("built"),
         "note": "results.json values are rounded to 4 decimals; position_rad = ankle angle in rad, negative = plantarflexion = toes down"}


def pick(exp_by_subject):
    """exp_by_subject: dict subject -> experiment id. S1toS2 only."""
    parts = [df[(df.experiment == e) & (df.subject == s) & (df.direction == DIR)] for s, e in exp_by_subject.items()]
    return pd.concat(parts)


sets = {
    "X00_baseline": pick({s: BASE for s in SUBJ}),
    "X00f_baseline_framework": pick({s: FRAMEWORK for s in SUBJ}),
    # D01 was run on CH, HM, IES, JM; for EG and YES the dashboard uses C01 (same recipe except the passive table)
    "D01_with_C01_for_EG_YES": pick({s: (C01 if s in ("EG", "YES") else D01) for s in SUBJ}),
}
for k, v in sets.items():
    assert v.subject.nunique() == 6, k
    assert (v.groupby(["subject", "pos"]).seed.nunique() > 0).all()
    # toes_down flag must be constant per cell and equal to "2 most negative positions of the person"
    tg = v.groupby(["subject", "pos"]).toes_down.agg(["min", "max"])
    assert (tg["min"] == tg["max"]).all(), k
    for s, g in v.groupby("subject"):
        order = sorted(g.pos.unique())
        flag = g.groupby("pos").toes_down.first()
        assert [int(flag[p]) for p in order] == [1, 1, 0, 0, 0, 0, 0, 0], (k, s)
facts["toes_down_rule_check"] = "OK: in every experiment/subject the 2 most negative positions have toes_down=1 and the other 6 have 0"
facts["seeds_per_set"] = {k: sorted(v.groupby("subject").seed.nunique().to_dict().items()) for k, v in sets.items()}

# ---------------------------------------------------------------- cell table
def cell_table(v):
    rows = []
    for (s, p), g in v.groupby(["subject", "pos"]):
        r2 = g.r2.values
        rows.append(dict(
            subject=s, position_rad=float(p), toes_down=int(g.toes_down.iloc[0]), n_seeds=int(len(g)),
            r2_med=med(r2), r2_floor0_med=med(np.clip(r2, 0, None)), r2_demeaned_med=med(g.r2_demeaned),
            mae_med=med(g.mae), bias_med=med(g.bias), mae_demeaned_med=med(g.mae_demeaned),
            peak_torque=med(g.peak_torque), peak_torque_min=float(g.peak_torque.min()), peak_torque_max=float(g.peak_torque.max()),
            n_seeds_r2_negative=int((r2 < 0).sum()),
            r2_min=float(r2.min()), r2_max=float(r2.max()), r2_p5=pct(r2, 5), r2_p95=pct(r2, 95),
            mae_sd=float(np.std(g.mae.values, ddof=1)) if len(g) > 1 else np.nan,
            r2_range=float(r2.max() - r2.min()), r2f_range=float(np.clip(r2, 0, None).max() - np.clip(r2, 0, None).min()),
            mae_range=float(g.mae.max() - g.mae.min()),
        ))
    c = pd.DataFrame(rows)
    c["subject"] = pd.Categorical(c.subject, SUBJ)
    return c.sort_values(["subject", "position_rad"]).reset_index(drop=True)


cells = {k: cell_table(v) for k, v in sets.items()}
cb = cells["X00_baseline"]

# ---------------------------------------------------------------- A
A = []
for r in cb.itertuples():
    A.append({k: (r4(getattr(r, k), 4) if isinstance(getattr(r, k), float) else getattr(r, k)) for k in
              ["subject", "position_rad", "toes_down", "n_seeds", "r2_med", "r2_floor0_med", "r2_demeaned_med", "mae_med",
               "bias_med", "mae_demeaned_med", "peak_torque", "peak_torque_min", "peak_torque_max", "n_seeds_r2_negative",
               "r2_min", "r2_max", "r2_p5", "r2_p95"]} | {"subject": str(r.subject)})
facts["A_baseline_cells"] = A

# ---------------------------------------------------------------- B
def group_summary(c):
    """Dashboard convention: R2 floored at 0 per run -> median over seeds per cell -> median over cells of the group."""
    out = {}
    for name, m in [("all_48", np.ones(len(c), bool)), ("toes_down_12", (c.toes_down == 1).values), ("other_36", (c.toes_down == 0).values)]:
        x = c[m]
        out[name] = dict(
            n_cells=int(m.sum()),
            r2_floor0=r4(med(x.r2_floor0_med)), r2_raw_median=r4(med(x.r2_med)), r2_demeaned=r4(med(x.r2_demeaned_med)),
            mae_nm=r4(med(x.mae_med)), mae_demeaned_nm=r4(med(x.mae_demeaned_med)), abs_bias_nm=r4(med(x.bias_med.abs())),
            peak_torque_nm=r4(med(x.peak_torque), 3),
            mae_over_peak=r4(med(x.mae_med / x.peak_torque), 4),
            cells_r2_floor0_below_0p5=int((x.r2_floor0_med < 0.5).sum()),
        )
    return out


facts["B_group_summaries"] = {k: group_summary(c) for k, c in cells.items()}

# per-run version of group value (median over cells within each run seed) for interval spread
def per_run_group(v, toes):
    x = v[v.toes_down == toes]
    return x.groupby("seed").r2f.median()


facts["B_per_seed_group_median_r2_floor0"] = {
    k: {n: dict(p5=r4(pct(per_run_group(v, t), 5)), p50=r4(med(per_run_group(v, t))), p95=r4(pct(per_run_group(v, t), 95)))
        for n, t in [("toes_down", 1), ("other", 0)]} for k, v in sets.items()}

# ---------------------------------------------------------------- C
bad = cb[cb.r2_med < 0.5].sort_values("r2_med")
facts["C_cells_median_r2_below_0p5"] = [
    dict(subject=str(r.subject), position_rad=r4(r.position_rad, 3), toes_down=int(r.toes_down), r2_med=r4(r.r2_med),
         r2_demeaned_med=r4(r.r2_demeaned_med), bias_med_nm=r4(r.bias_med), mae_med_nm=r4(r.mae_med),
         mae_demeaned_med_nm=r4(r.mae_demeaned_med), peak_torque_nm=r4(r.peak_torque, 3), n_seeds_r2_negative=int(r.n_seeds_r2_negative))
    for r in bad.itertuples()]
facts["C_n_cells_below_0p5"] = int(len(bad))
facts["C_n_below_0p5_that_are_toes_down"] = int((bad.toes_down == 1).sum())
facts["C_below_0p5_demeaned_above_0p7"] = int((bad.r2_demeaned_med > 0.7).sum())
facts["C_below_0p5_median_raw_r2"] = r4(med(bad.r2_med))
facts["C_below_0p5_median_demeaned_r2"] = r4(med(bad.r2_demeaned_med))

xb = sets["X00_baseline"]
cand = xb[(xb.r2 < 0) & (xb.r2_demeaned > 0.7)].copy()
cand["cell"] = cand.subject + "@" + cand.pos.astype(str)
cand = cand.sort_values("r2_demeaned", ascending=False)
facts["C_n_single_runs_r2_neg_demeaned_gt_0p7"] = int(len(cand))
facts["C_n_single_runs_r2_negative"] = int((xb.r2 < 0).sum())
facts["C_n_single_runs_total"] = int(len(xb))


def ex(r):
    return dict(subject=r.subject, position_rad=r4(r.pos, 3), seed=int(r.seed), r2=r4(r.r2), r2_demeaned=r4(r.r2_demeaned),
                bias_nm=r4(r.bias), mae_nm=r4(r.mae), mae_demeaned_nm=r4(r.mae_demeaned), peak_torque_nm=r4(r.peak_torque, 3),
                toes_down=int(r.toes_down))


# "best 3": highest demeaned R2 among R2<0 runs, one per cell, and the R2 must be a plain negative (not a -57 outlier) -> report both
top_cells = cand.drop_duplicates("cell").head(3)
facts["C_best3_examples_by_demeaned_r2"] = [ex(r) for r in top_cells.itertuples()]
facts["C_top10_candidates"] = [ex(r) for r in cand.head(10).itertuples()]
# strict spec yields only JM -0.078 (a non-toes-down cell with a 12 Nm offset), so also give a relaxed list: R2 < 0 and demeaned R2 > 0.4,
# one run per cell, toes-down cells first
relaxed = xb[(xb.r2 < 0) & (xb.r2_demeaned > 0.4)].copy()
relaxed["cell"] = relaxed.subject + "@" + relaxed.pos.astype(str)
relaxed = relaxed.sort_values("r2_demeaned", ascending=False)
facts["C_relaxed_examples_r2_neg_demeaned_gt_0p4_one_per_cell"] = [ex(r) for r in relaxed.drop_duplicates("cell").itertuples()]
facts["C_toes_down_negative_runs_all"] = [ex(r) for r in xb[(xb.r2 < 0) & (xb.toes_down == 1)].sort_values("r2_demeaned", ascending=False).itertuples()]
facts["C_n_negative_runs_by_toes_down"] = {"toes_down": int(((xb.r2 < 0) & (xb.toes_down == 1)).sum()), "other": int(((xb.r2 < 0) & (xb.toes_down == 0)).sum())}
facts["C_other_sets_cells_median_r2_below_0p5"] = {
    k: [dict(subject=str(r.subject), position_rad=r4(r.position_rad, 3), toes_down=int(r.toes_down), r2_med=r4(r.r2_med), r2_demeaned_med=r4(r.r2_demeaned_med),
             bias_med_nm=r4(r.bias_med), n_seeds_r2_negative=int(r.n_seeds_r2_negative))
        for r in c[c.r2_med < 0.5].sort_values("r2_med").itertuples()] for k, c in cells.items() if k != "X00_baseline"}
# JM -0.078 evidence: offset larger than the peak torque
jm = xb[(xb.subject == "JM") & (xb.pos == -0.078)]
facts["C_JM_minus0p078_evidence"] = dict(n_seeds=int(len(jm)), n_negative=int((jm.r2 < 0).sum()), bias_range_nm=[r4(jm.bias.min()), r4(jm.bias.max())],
    mae_range_nm=[r4(jm.mae.min()), r4(jm.mae.max())], peak_torque_nm=r4(jm.peak_torque.median(), 3), mae_over_peak=r4(jm.mae.median() / jm.peak_torque.median(), 2),
    mae_demeaned_median_nm=r4(jm.mae_demeaned.median()), r2_demeaned_median=r4(jm.r2_demeaned.median()))
jd = df[(df.experiment == D01) & (df.subject == "JM") & (df.direction == DIR) & (df.pos == -0.078)]
facts["C_JM_minus0p078_in_D01"] = dict(median_r2=r4(jd.r2.median()), median_bias_nm=r4(jd.bias.median()), median_mae_nm=r4(jd.mae.median()), n=int(len(jd)))
# also the best, one per subject (more varied for the video)
facts["C_best_per_subject"] = [ex(g.iloc[0]) for _, g in cand.groupby("subject", observed=True)] if len(cand) else []

# ---------------------------------------------------------------- D  offset-only reference
def add_offset(v):
    v = v.copy()
    ok = v.r2 < 1
    v["var_y"] = np.where(ok, v.rmse ** 2 / (1 - v.r2).where(ok, np.nan), np.nan)      # SS_tot / N
    v["r2_offset_only"] = 1 - v.bias ** 2 / v["var_y"]
    v["r2_dem_plus_offset"] = v.r2_demeaned - v.bias ** 2 / v["var_y"]                    # identity check (should equal r2)
    v["offset_share_of_drop"] = (v.r2_demeaned - v.r2).replace(0, np.nan)
    return v


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


xo = add_offset(xb)
# identity check: r2 = r2_demeaned - bias^2/var (only if r2_demeaned is 1 - (MSE - bias^2)/var)
idres = (xo.r2 - xo.r2_dem_plus_offset).abs()
facts["D_identity_check"] = dict(
    statement="r2 == r2_demeaned - bias^2/var_y  (var_y = rmse^2/(1-r2)); residual comes from 4-decimal rounding",
    median_abs_residual=r4(idres.median(), 5), p95_abs_residual=r4(idres.quantile(.95), 5),
    max_abs_residual_runs_with_r2_gt_minus5=r4(idres[xo.r2 > -5].max(), 5), n_runs=int(idres.notna().sum()))

D = []
for (s, p), g in xo.groupby(["subject", "pos"]):
    D.append(dict(subject=s, position_rad=float(p), toes_down=int(g.toes_down.iloc[0]),
                  r2_med=med(g.r2), r2_offset_only_med=float(np.nanmedian(g.r2_offset_only)),
                  r2_demeaned_med=med(g.r2_demeaned), bias_med=med(g.bias), peak=med(g.peak_torque),
                  std_y_nm=float(np.nanmedian(np.sqrt(g.var_y)))))
Dd = pd.DataFrame(D)
Dd["subject"] = pd.Categorical(Dd.subject, SUBJ)
Dd = Dd.sort_values(["subject", "position_rad"]).reset_index(drop=True)
Dd["gap_offset_vs_actual"] = Dd.r2_offset_only_med - Dd.r2_med
facts["D_cells"] = [{k: (r4(v, 4) if isinstance(v, float) else (str(v) if k == "subject" else v)) for k, v in r._asdict().items() if k != "Index"}
                    for r in Dd.itertuples()]

for name, sel_c, sel_r in [("toes_down_12", Dd.toes_down == 1, xo.toes_down == 1), ("other_36", Dd.toes_down == 0, xo.toes_down == 0),
                           ("all_48", Dd.toes_down >= 0, xo.toes_down >= 0)]:
    c = Dd[sel_c]
    r = xo[sel_r]
    pc, n = corr(c.r2_med, c.r2_offset_only_med)
    pr, nr = corr(r.r2, r.r2_offset_only)
    facts.setdefault("D_tracking", {})[name] = dict(
        n_cells=n, pearson_cell_medians=r4(pc), spearman_cell_medians=r4(spearman(c.r2_med, c.r2_offset_only_med)),
        median_abs_gap_cell=r4((c.r2_offset_only_med - c.r2_med).abs().median()),
        median_actual_r2=r4(med(c.r2_med)), median_offset_only_r2=r4(med(c.r2_offset_only_med)),
        n_runs=nr, pearson_per_run=r4(pr), median_abs_gap_per_run=r4((r.r2_offset_only - r.r2).abs().median()),
        frac_runs_offset_only_within_0p1_of_actual=r4(((r.r2_offset_only - r.r2).abs() < 0.1).mean(), 3),
        spearman_per_run=r4(spearman(r.r2, r.r2_offset_only)),
    )
# honest reading of the toes-down cells where actual r2 < 0.5
t = Dd[(Dd.toes_down == 1)]
facts["D_toes_down_table"] = [dict(subject=str(r.subject), position_rad=r4(r.position_rad, 3), r2_med=r4(r.r2_med),
                                   r2_offset_only_med=r4(r.r2_offset_only_med), r2_demeaned_med=r4(r.r2_demeaned_med),
                                   bias_med_nm=r4(r.bias_med), std_of_true_torque_nm=r4(r.std_y_nm), peak_nm=r4(r.peak))
                              for r in t.itertuples()]
# how much of the baseline's loss of R2 (1 - r2) is the offset: 1 - r2 = (1 - r2_dem) + bias^2/var
xo["loss_total"] = 1 - xo.r2
xo["loss_shape"] = 1 - xo.r2_demeaned
xo["loss_offset"] = xo.bias ** 2 / xo.var_y
for name, m in [("toes_down", xo.toes_down == 1), ("other", xo.toes_down == 0)]:
    r = xo[m & np.isfinite(xo.var_y) & (xo.r2 > -5)]
    facts.setdefault("D_loss_decomposition_median_per_run", {})[name] = dict(
        one_minus_r2=r4(med(r.loss_total)), from_shape_errors=r4(med(r.loss_shape)), from_constant_offset=r4(med(r.loss_offset)),
        offset_share=r4(med(r.loss_offset / r.loss_total)))

# ---------------------------------------------------------------- E  learning curves
def lc_summary(exp, s):
    f = LC / exp / s / DIR / "seed0.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    h = d["history"]
    by_ep = {e["epoch"]: e for e in h}
    b = by_ep[d["best_epoch"]]
    kept_by_val_mse = int(min(h, key=lambda e: e["mse_val"])["epoch"])
    kept_by_keras_val = int(min(h, key=lambda e: e["keras_val_loss"])["epoch"])
    fin = d["final_per_trial"]
    ang = {t["trial"]: t["angle"] for t in fin["test"]}
    order = sorted(ang, key=lambda t: ang[t])[:2]   # 2 most negative test angles = toes-down cells

    def tr(sp, f_):
        return float(np.median([t[f_] for t in fin[sp]]))

    def tr_toes(sp, f_):
        # fit/val trial numbers are the S1 ones; match toes-down by angle rank inside each split
        a = sorted(fin[sp], key=lambda t: t["angle"])[:2]
        return float(np.median([t[f_] for t in a]))

    return dict(
        epochs_run=int(d["epochs"]), best_epoch=int(d["best_epoch"]), n_history=len(h), kept_epoch_matches_min_val_mse=(kept_by_val_mse == d["best_epoch"]),
        kept_epoch_matches_min_keras_val_loss=(kept_by_keras_val == d["best_epoch"]),
        epoch_with_min_keras_val_loss=kept_by_keras_val, epoch_with_min_mse_val=kept_by_val_mse,
        mae_fit=r4(b["mae_fit"]), mae_val=r4(b["mae_val"]), mae_test=r4(b["mae_test"]),
        ratio_test_over_val=r4(b["mae_test"] / b["mae_val"], 3), ratio_val_over_fit=r4(b["mae_val"] / b["mae_fit"], 3),
        ratio_test_over_fit=r4(b["mae_test"] / b["mae_fit"], 3),
        mse_fit=r4(b["mse_fit"]), mse_val=r4(b["mse_val"]), mse_test=r4(b["mse_test"]),
        rmse_fit=r4(np.sqrt(b["mse_fit"])), rmse_val=r4(np.sqrt(b["mse_val"])), rmse_test=r4(np.sqrt(b["mse_test"])),
        # curve shape: first epoch vs best vs last
        mae_test_first_epoch=r4(h[0]["mae_test"]), mae_test_last_epoch=r4(h[-1]["mae_test"]), min_mae_test_any_epoch=r4(min(e["mae_test"] for e in h)),
        epoch_of_min_mae_test=int(min(h, key=lambda e: e["mae_test"])["epoch"]),
        lr_first=r4(h[0]["lr"], 6), lr_last=r4(h[-1]["lr"], 6),
        final_median_trial_r2=dict(fit=r4(tr("fit", "r2")), val=r4(tr("val", "r2")), test=r4(tr("test", "r2"))),
        final_median_trial_mae=dict(fit=r4(tr("fit", "mae")), val=r4(tr("val", "mae")), test=r4(tr("test", "mae"))),
        final_toes_down_trials_r2=dict(fit=r4(tr_toes("fit", "r2")), val=r4(tr_toes("val", "r2")), test=r4(tr_toes("test", "r2"))),
        final_toes_down_trials_mae=dict(fit=r4(tr_toes("fit", "mae")), val=r4(tr_toes("val", "mae")), test=r4(tr_toes("test", "mae"))),
        n_windows=d.get("n_windows"), config=d.get("config"), seconds=r4(d.get("seconds"), 0))


E = {}
for exp in [FRAMEWORK, D01]:
    E[exp] = {s: lc_summary(exp, s) for s in SUBJ}
for exp, per in E.items():
    ok = [v for v in per.values() if v]
    if not ok:
        continue
    per["_median_over_subjects"] = dict(
        epochs_run=r4(med([v["epochs_run"] for v in ok]), 1), best_epoch=r4(med([v["best_epoch"] for v in ok]), 1),
        mae_fit=r4(med([v["mae_fit"] for v in ok])), mae_val=r4(med([v["mae_val"] for v in ok])), mae_test=r4(med([v["mae_test"] for v in ok])),
        ratio_test_over_val=r4(med([v["ratio_test_over_val"] for v in ok]), 3), ratio_val_over_fit=r4(med([v["ratio_val_over_fit"] for v in ok]), 3),
        ratio_test_over_fit=r4(med([v["ratio_test_over_fit"] for v in ok]), 3),
        n_subjects_with_test_gt_val=int(sum(v["mae_test"] > v["mae_val"] for v in ok)),
        n_subjects_with_val_gt_fit=int(sum(v["mae_val"] > v["mae_fit"] for v in ok)))
facts["E_learning_curves"] = E
facts["E_note"] = ("fit/val/test MAE come from the epoch the run kept (restore_best_weights on val_loss). Val = the 20 % of each S1 trial that follows "
                   "the first 80 % in time (dataset_builder._temporal_split lines 520-528), i.e. same session, same angle, adjacent in time. "
                   "Test = the retest session (S2). So test/val above 1 = session shift (plus extra drift inside the trial), val/fit above 1 = classic over-fitting.")

# ---------------------------------------------------------------- F  seed-to-seed spread
def spread(c):
    out = {}
    for name, m in [("toes_down", c.toes_down == 1), ("other", c.toes_down == 0), ("all", c.toes_down >= 0)]:
        x = c[m]
        out[name] = dict(
            n_cells=int(len(x)),
            median_r2_range_max_minus_min_raw=r4(med(x.r2_range)), median_r2_range_after_floor0=r4(med(x.r2f_range)),
            median_mae_sd_nm=r4(med(x.mae_sd)), median_mae_range_nm=r4(med(x.mae_range)),
            median_mae_sd_over_median_mae=r4(med(x.mae_sd / x.mae_med)),
            median_n_seeds_r2_negative=r4(med(x.n_seeds_r2_negative), 1),
            cells_with_at_least_one_negative_seed=int((x.n_seeds_r2_negative > 0).sum()),
            cells_with_all_seeds_negative=int((x.n_seeds_r2_negative == x.n_seeds).sum()),
        )
    return out


cells["X00_baseline_seeds0-2_only"] = cell_table(sets["X00_baseline"][sets["X00_baseline"].seed < 3])
facts["F_seed_spread"] = {k: spread(c) for k, c in cells.items()}
facts["F_seed_counts"] = {k: int(c.n_seeds.median()) for k, c in cells.items()}

json.dump(facts, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------------------------------------------------------- print
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
print("=" * 100, "\nA  X00_baseline per cell (median over 10 seeds)")
print(cb[["subject", "position_rad", "toes_down", "r2_med", "r2_floor0_med", "r2_demeaned_med", "mae_med", "bias_med", "peak_torque",
          "n_seeds_r2_negative", "r2_min", "r2_max", "r2_p5", "r2_p95"]].round(3).to_string(index=False))
print("=" * 100, "\nB  group summaries (R2 floored at 0; median over seeds per cell, then median over cells)")
for k, v in facts["B_group_summaries"].items():
    print(f"\n[{k}]")
    print(pd.DataFrame(v).T.to_string())
print("\nB2 per-seed group median R2 (floor 0) p5/p50/p95:", json.dumps(facts["B_per_seed_group_median_r2_floor0"]))
print("=" * 100, f"\nC  cells with median R2 < 0.5: {len(bad)} (toes-down {facts['C_n_below_0p5_that_are_toes_down']}); of those demeaned R2 > 0.7: {facts['C_below_0p5_demeaned_above_0p7']}")
print(pd.DataFrame(facts["C_cells_median_r2_below_0p5"]).to_string(index=False))
print(f"\nsingle runs with R2<0: {facts['C_n_single_runs_r2_negative']} of {facts['C_n_single_runs_total']}; with demeaned > 0.7: {facts['C_n_single_runs_r2_neg_demeaned_gt_0p7']}")
print("relaxed examples (R2<0, demeaned>0.4, one per cell):")
print(pd.DataFrame(facts["C_relaxed_examples_r2_neg_demeaned_gt_0p4_one_per_cell"]).to_string(index=False))
print("negative runs by toes_down:", facts["C_n_negative_runs_by_toes_down"])
print("JM -0.078:", facts["C_JM_minus0p078_evidence"], "| in D01:", facts["C_JM_minus0p078_in_D01"])
print("other sets, cells <0.5:", json.dumps(facts["C_other_sets_cells_median_r2_below_0p5"]))
print("best 3 (highest demeaned R2, distinct cells):")
print(pd.DataFrame(facts["C_best3_examples_by_demeaned_r2"]).to_string(index=False))
print("top 10 candidates:")
print(pd.DataFrame(facts["C_top10_candidates"]).to_string(index=False))
print("=" * 100, "\nD  offset-only reference")
print("identity check:", facts["D_identity_check"])
print(json.dumps(facts["D_tracking"], indent=1))
print(json.dumps(facts["D_loss_decomposition_median_per_run"], indent=1))
print(pd.DataFrame(facts["D_toes_down_table"]).to_string(index=False))
print("=" * 100, "\nE  learning curves (seed 0, S1toS2)")
for exp, per in E.items():
    print(f"\n[{exp}]")
    rows = [dict(subject=s, **{k: v[k] for k in ["epochs_run", "best_epoch", "mae_fit", "mae_val", "mae_test", "ratio_val_over_fit", "ratio_test_over_val",
                                                    "kept_epoch_matches_min_val_mse", "epoch_of_min_mae_test"]}) for s, v in per.items() if s != "_median_over_subjects" and v]
    print(pd.DataFrame(rows).to_string(index=False))
    print("median over subjects:", per.get("_median_over_subjects"))
print("=" * 100, "\nF  seed spread")
for k, v in facts["F_seed_spread"].items():
    print(f"\n[{k}]  (seeds per cell: {facts['F_seed_counts'][k]})")
    print(pd.DataFrame(v).T.to_string())
print("\nwrote", OUT)
