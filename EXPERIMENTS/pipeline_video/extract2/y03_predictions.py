"""y03: the BASELINE's predictions and per-cell scores (X00_baseline, original LSTM-8 code, seeds 0-9, learn S1 -> test S2).

Reads  : DWT/EXPERIMENTS/results/X00_baseline/<subject>/S1toS2/seed<k>.npz   (y_true_original, y_pred, position, position_snapped)
         DWT/ML/research  (load_subject, PassiveModel -> the same passive tables score.py uses; map_lookup)
         EXPERIMENTS/dashboard_react/src/data/results.json   (cross-check only)
         EXPERIMENTS/pipeline_video/data2/facts_y01.json     (cross-check only)
         EMG_DATA_DIR = ML/subjects of the main checkout
Writes : EXPERIMENTS/pipeline_video/data2/y03.npz
         EXPERIMENTS/pipeline_video/data2/facts_y03.json      (one top-level key: "y03")
Run    : OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
         .venv311/Scripts/python.exe EXPERIMENTS/pipeline_video/extract2/y03_predictions.py
         (no training; ~1 min)

Answer key ("passive_trial", the dashboard's primary key), exactly as DWT/EXPERIMENTS/code/score.py does for X00_baseline
(score.py cannot be imported: it scores every experiment at import time, so its X00 block is re-implemented line for line
and checked against results.json, which was produced by presentation_metrics.py with the same formulas):
    key = y_true_original + map_lookup(maps['legacy']['S2'], position) - map_lookup(maps['static']['S2'], position)
For YES and IES the legacy and the static S2 tables are identical (checked below), so the correction is zero for them.

Conventions
    angle: negative = plantarflexion = toes down.  Toes-down cells = each person's 2 most negative positions.
    cell  = subject x position (8 per subject, windows grouped by position_snapped, label = nearest S2 trial angle, rounded to 3 dp)
    bias  = mean(prediction - key)   -> positive = the prediction sits ABOVE the truth
    R2        = 1 - sum((pred-key)^2) / sum((key-mean(key))^2)                (vs the cell's own mean)
    R2_shape  = same with both key and prediction demeaned (level removed)
    sd_truth  = np.std(key) (ddof 0);  peak = max|key|;  corr = Pearson(pred, key)
    identity  : R2 = R2_shape - bias^2 / var(key)   (exact, since sum(e^2) = sum(e_demeaned^2) + N*bias^2)
    time axis : window i of a trial is one sample (0.01 s) after window i-1.  t_window_s = i*0.01 (0 .. 88.90 s);
                t_trial_s = (i+19)*0.01 = time since the start of the trial (the first window ends after 20 samples).
npz arrays are float64 and unrounded.  Display numbers (rounded) are only in facts['y03']['display'].
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["EMG_DATA_DIR"] = "C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/ML/subjects"

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = pathlib.Path(__file__).resolve().parent
VWT = HERE.parents[2]
DWT = pathlib.Path("C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137")
sys.path.insert(0, str(DWT))
from ML.research.data import load_subject                    # noqa: E402
from ML.research.passive import map_lookup                   # noqa: E402

RESULTS_JSON = VWT / "EXPERIMENTS/dashboard_react/src/data/results.json"
FACTS_Y01 = VWT / "EXPERIMENTS/pipeline_video/data2/facts_y01.json"
OUT_DIR = VWT / "EXPERIMENTS/pipeline_video/data2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS = DWT / "EXPERIMENTS/results/X00_baseline"

SUBJ = ["CH", "EG", "HM", "IES", "JM", "YES"]
SEEDS = list(range(10))
FS = 100.0
NPOS = 8
checks = []


def check(name, ok, detail):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("OK   " if ok else "FAIL ") + name + " :: " + str(detail))
    return ok


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    ed = (p - p.mean()) - (y - y.mean())
    var = ss / len(y)
    bias = float(np.mean(e))
    r2 = float(1 - np.sum(e ** 2) / ss)
    r2s = float(1 - np.sum(ed ** 2) / ss)
    return dict(r2=r2, r2_shape=r2s, bias=bias, mae=float(np.mean(np.abs(e))), sd=float(np.sqrt(var)), var=float(var),
                peak=float(np.max(np.abs(y))), corr=float(np.corrcoef(p, y)[0, 1]),
                mae_shape=float(np.mean(np.abs(ed))), identity_res=float(r2 - (r2s - bias ** 2 / var)))


# ------------------------------------------------------------------ load all runs, build the key, score every cell
M_KEYS = ["r2", "r2_shape", "bias", "mae", "sd", "var", "peak", "corr", "mae_shape", "identity_res"]
A = {k: np.full((6, 10, NPOS), np.nan) for k in M_KEYS}
POS = np.full((6, NPOS), np.nan)          # position label (rad) of each cell
POS_SNAP = np.full((6, NPOS), np.nan)     # raw snapped angle
TD = np.zeros((6, NPOS), int)
KEEP = {}                                  # (subject, seed) -> dict(key, pred, y_orig, snapped) for plot arrays
corr_info = {}
npz_consistency = {}

for si, s in enumerate(SUBJ):
    b = load_subject(s)
    maps = b["maps"]
    s2_angles = sorted(float(r["angle"]) for r in b["active"] if r["session"] == "S2")
    ref = None
    tab_equal = maps["legacy"]["S2"] == maps["static"]["S2"]
    for ki, seed in enumerate(SEEDS):
        z = np.load(RUNS / s / "S1toS2" / f"seed{seed}.npz")
        pos, snapped = z["position"].astype(np.float64), z["position_snapped"].astype(np.float64)
        y_orig, pred = z["y_true_original"].astype(np.float64), z["y_pred"].astype(np.float64)
        if ref is None:
            ref = (pos, snapped, y_orig)
            # correction depends only on the position value: evaluate it once per distinct position (exact, same function)
            u, inv = np.unique(pos, return_inverse=True)
            corr_u = map_lookup(maps["legacy"]["S2"], u) - map_lookup(maps["static"]["S2"], u)
            corr = corr_u[inv]
            corr_info[s] = dict(max_abs_correction_nm=float(np.abs(corr).max()), mean_correction_nm=float(corr.mean()),
                                legacy_equals_static_S2_table=bool(tab_equal))
            # direct per-sample evaluation of the first 3000 samples as a guard for the unique-trick
            d = map_lookup(maps["legacy"]["S2"], pos[:3000]) - map_lookup(maps["static"]["S2"], pos[:3000])
            corr_info[s]["unique_trick_max_diff"] = float(np.abs(d - corr[:3000]).max())
            y_key = y_orig + corr
            labels = []
            for a in np.unique(snapped):
                labels.append((float(a), min(s2_angles, key=lambda x: abs(x - a))))
            assert len(labels) == NPOS, (s, len(labels))
            order = sorted(range(NPOS), key=lambda j: labels[j][1])
            POS[si] = [round(labels[j][1], 3) for j in order]
            POS_SNAP[si] = [labels[j][0] for j in order]
            TD[si, :2] = 1
            uniq_snap = [labels[j][0] for j in order]
        else:
            same = (np.array_equal(pos, ref[0]) and np.array_equal(snapped, ref[1]) and np.array_equal(y_orig, ref[2]))
            npz_consistency.setdefault(s, []).append(same)
        key = y_orig + corr
        for pj, a in enumerate(uniq_snap):
            m = snapped == a
            assert m.sum() == 8891, (s, seed, a, m.sum())
            mt = metrics(key[m], pred[m])
            for k in M_KEYS:
                A[k][si, ki, pj] = mt[k]
        KEEP[(s, seed)] = dict(key=key, pred=pred, y_orig=y_orig, snapped=snapped)
    print("done", s, flush=True)

check("windows per cell", True, "every cell has 8891 windows (assert), 8 cells per subject, 6 subjects, 10 seeds")
check("inputs identical over seeds", all(all(v) for v in npz_consistency.values()),
      "position, position_snapped and y_true_original are identical for seeds 0-9 in every subject")
check("YES/IES passive correction is zero", corr_info["YES"]["max_abs_correction_nm"] == 0 and corr_info["IES"]["max_abs_correction_nm"] == 0,
      {s: round(v["max_abs_correction_nm"], 4) for s, v in corr_info.items()})
check("unique-position shortcut equals the per-sample lookup", max(v["unique_trick_max_diff"] for v in corr_info.values()) == 0.0,
      "max difference on the first 3000 samples of each subject = 0")
for k in ("sd", "peak", "var"):
    assert np.allclose(A[k], A[k][:, :1, :], rtol=0, atol=0), k
check("truth statistics do not depend on the seed", True, "sd, var and peak|key| are identical over the 10 seeds in every cell")
check("toes-down rule", bool((TD.sum(1) == 2).all() and (TD[:, :2] == 1).all()), "toes_down=1 for the 2 most negative positions of each person (12 cells), 0 for the other 36")

# ------------------------------------------------------------------ cross-check against results.json (experiment X00_baseline)
raw = json.load(open(RESULTS_JSON))
rj = pd.DataFrame(raw["rows"], columns=raw["cols"])
rj = rj[(rj.experiment == "X00_baseline") & (rj.direction == "S1toS2")].copy()
rj["pos"] = rj.position_rad.round(3)
rows = []
for si, s in enumerate(SUBJ):
    for ki, seed in enumerate(SEEDS):
        for pj in range(NPOS):
            rows.append(dict(subject=s, seed=seed, pos=round(float(POS[si, pj]), 3), si=si, ki=ki, pj=pj))
mine = pd.DataFrame(rows).merge(rj, on=["subject", "seed", "pos"], how="outer", indicator=True)
check("results.json has the same 480 runs", bool((mine._merge == "both").all() and len(mine) == 480), f"{(mine._merge == 'both').sum()} matched of {len(mine)}")
mine = mine[mine._merge == "both"]
cmp_cols = {"r2": "r2", "mae": "mae", "r2_shape": "r2_demeaned", "bias": "bias", "peak": "peak_torque", "mae_shape": "mae_demeaned"}
diffs = {}
for mk, jk in cmp_cols.items():
    mv = np.array([A[mk][r.si, r.ki, r.pj] for r in mine.itertuples()])
    diffs[mk] = float(np.max(np.abs(mv - mine[jk].values)))
# relative error for r2 (large negatives, results.json rounds to 4 decimals)
mv = np.array([A["r2"][r.si, r.ki, r.pj] for r in mine.itertuples()])
rel = float(np.max(np.abs(mv - mine.r2.values) / np.maximum(1, np.abs(mine.r2.values))))
check("R2 and MAE agree with results.json (tolerance 1e-3)", diffs["r2"] < 1e-3 and diffs["mae"] < 1e-3,
      f"(results.json rounds to 4 dp, peak to 2 dp) max |diff|: R2 {diffs['r2']:.2e}, MAE {diffs['mae']:.2e}, R2_shape {diffs['r2_shape']:.2e}, bias {diffs['bias']:.2e}, peak {diffs['peak']:.2e}, "
      f"MAE_shape {diffs['mae_shape']:.2e}; max relative R2 diff {rel:.2e}")
tdj = mine.groupby(["subject", "pos"]).toes_down.first()
tdm = {(SUBJ[si], round(float(POS[si, pj]), 3)): int(TD[si, pj]) for si in range(6) for pj in range(NPOS)}
check("toes_down flag equals results.json", all(tdm[k] == int(v) for k, v in tdj.items()), "all 48 cells")
idr = np.abs(A["identity_res"])
check("identity R2 = R2_shape - bias^2/var", float(idr.max()) < 1e-9, f"max |residual| over all 480 runs = {idr.max():.2e}")

# ------------------------------------------------------------------ cell summaries
med = lambda a: np.median(a, axis=1)
C = dict(
    r2_med=med(A["r2"]), r2_floor0_med=med(np.clip(A["r2"], 0, None)), r2_shape_med=med(A["r2_shape"]), bias_med=med(A["bias"]),
    mae_med=med(A["mae"]), corr_med=med(A["corr"]), r2_min=A["r2"].min(1), r2_max=A["r2"].max(1), r2_range=A["r2"].max(1) - A["r2"].min(1),
    r2_floor0_range=np.clip(A["r2"], 0, None).max(1) - np.clip(A["r2"], 0, None).min(1), n_neg=(A["r2"] < 0).sum(1),
    sd_truth=A["sd"][:, 0, :], peak_truth=A["peak"][:, 0, :],
)
tdm_ = TD == 1
oth = TD == 0


def gmed(x, mask):
    return float(np.median(x[mask]))


groups = {}
for name, mk in (("toes_down_12", tdm_), ("other_36", oth), ("all_48", np.ones_like(tdm_))):
    # per-run loss decomposition (same selection as y01 section D: r2 > -5)
    lt = 1 - A["r2"]
    lo = A["bias"] ** 2 / A["var"]
    sel = np.broadcast_to(mk[:, None, :], A["r2"].shape) & (A["r2"] > -5)
    groups[name] = dict(
        n_cells=int(mk.sum()),
        r2_range_median=gmed(C["r2_range"], mk), r2_floor0_range_median=gmed(C["r2_floor0_range"], mk),
        r2_floor0_median=gmed(C["r2_floor0_med"], mk), r2_raw_median=gmed(C["r2_med"], mk), r2_shape_median=gmed(C["r2_shape_med"], mk),
        mae_median_nm=gmed(C["mae_med"], mk), abs_bias_median_nm=float(np.median(np.abs(C["bias_med"][mk]))),
        peak_median_nm=gmed(C["peak_truth"], mk), sd_truth_median_nm=gmed(C["sd_truth"], mk),
        sd_truth_min_nm=float(C["sd_truth"][mk].min()), sd_truth_max_nm=float(C["sd_truth"][mk].max()),
        mae_over_peak_median=float(np.median(C["mae_med"][mk] / C["peak_truth"][mk])),
        corr_median=gmed(C["corr_med"], mk), corr_min=float(C["corr_med"][mk].min()), corr_max=float(C["corr_med"][mk].max()),
        offset_share_of_lost_r2_median_per_run=float(np.median((lo[sel] / lt[sel]))),
        n_runs_for_offset_share=int(sel.sum()),
        n_cells_with_a_negative_seed=int((C["n_neg"][mk] > 0).sum()),
    )

# ------------------------------------------------------------------ checks against the earlier y01 facts
f01 = json.load(open(FACTS_Y01))
b01 = f01["B_group_summaries"]["X00_baseline"]
s01 = f01["F_seed_spread"]["X00_baseline"]
check("cell R2 range (toes-down) equals y01", abs(groups["toes_down_12"]["r2_range_median"] - s01["toes_down"]["median_r2_range_max_minus_min_raw"]) < 1e-3,
      f"y03 {groups['toes_down_12']['r2_range_median']:.4f} vs y01 {s01['toes_down']['median_r2_range_max_minus_min_raw']}")
check("cell R2 range (other) equals y01", abs(groups["other_36"]["r2_range_median"] - s01["other"]["median_r2_range_max_minus_min_raw"]) < 1e-3,
      f"y03 {groups['other_36']['r2_range_median']:.4f} vs y01 {s01['other']['median_r2_range_max_minus_min_raw']}")
for g, kk in (("toes_down_12", "toes_down_12"), ("other_36", "other_36")):
    y = b01[kk]
    check(f"group summary {g} equals y01", all(abs(a - b) < 1e-3 for a, b in [
        (groups[g]["r2_floor0_median"], y["r2_floor0"]), (groups[g]["mae_median_nm"], y["mae_nm"]), (groups[g]["peak_median_nm"], y["peak_torque_nm"]),
        (groups[g]["r2_shape_median"], y["r2_demeaned"]), (groups[g]["abs_bias_median_nm"], y["abs_bias_nm"])]),
        f"R2 {groups[g]['r2_floor0_median']:.4f}/{y['r2_floor0']}, MAE {groups[g]['mae_median_nm']:.4f}/{y['mae_nm']}, peak {groups[g]['peak_median_nm']:.3f}/{y['peak_torque_nm']}, "
        f"shape R2 {groups[g]['r2_shape_median']:.4f}/{y['r2_demeaned']}, |bias| {groups[g]['abs_bias_median_nm']:.4f}/{y['abs_bias_nm']}")

# ------------------------------------------------------------------ plot arrays
def cell_index(s, p):
    si = SUBJ.index(s)
    pj = int(np.argmin(np.abs(POS[si] - p)))
    assert abs(POS[si, pj] - p) < 0.0015, (s, p, POS[si])
    return si, pj


def trial_series(s, pj, seeds=SEEDS):
    si = SUBJ.index(s)
    a = POS_SNAP[si, pj]
    out_k, out_p, out_o = None, [], None
    for seed in seeds:
        d = KEEP[(s, seed)]
        m = d["snapped"] == a
        if out_k is None:
            out_k, out_o = d["key"][m], d["y_orig"][m]
        else:
            assert np.array_equal(out_k, d["key"][m])
        out_p.append(d["pred"][m])
    return out_k, np.array(out_p), out_o


NWIN = 8891
t_window = np.arange(NWIN) / FS
t_trial = (np.arange(NWIN) + 19) / FS

si_y, pj_y = cell_index("YES", -0.486)
yes_key, yes_pred, _ = trial_series("YES", pj_y)
# YES strongest position = largest SD of the truth
pj_ys = int(np.argmax(C["sd_truth"][si_y]))
ys_key, ys_pred, _ = trial_series("YES", pj_ys)
si_j, pj_j = cell_index("JM", -0.078)
jm_key, jm_pred, jm_orig = trial_series("JM", pj_j, seeds=[0])
si_e, pj_e = cell_index("EG", -0.739)
eg_key, eg_pred, _ = trial_series("EG", pj_e, seeds=[4])

# consistency of plot arrays with the cell metrics
m1 = metrics(yes_key, yes_pred[1])
check("YES -0.486 seed 1 series reproduces the cell table", abs(m1["r2"] - A["r2"][si_y, 1, pj_y]) < 1e-12, f"R2 {m1['r2']:.6f}")
mj = metrics(jm_key, jm_pred[0])
check("JM -0.078 seed 0 series reproduces the cell table", abs(mj["r2"] - A["r2"][si_j, 0, pj_j]) < 1e-12, f"R2 {mj['r2']:.4f}")
check("sample length of plot arrays", yes_key.shape == (NWIN,) and yes_pred.shape == (10, NWIN) and ys_pred.shape == (10, NWIN), "8891 windows = 88.91 s per trial")

# suggest a 30 s piece of the YES -0.486 trial with several clear pushes
W = 30.0
n_w = int(W * FS)
k_s = np.convolve(yes_key, np.ones(15) / 15, mode="same")                         # 0.15 s moving average
sd_cell = float(yes_key.std())
pk, prop = find_peaks(k_s, prominence=1.0 * sd_cell, distance=int(1.0 * FS))
best = None
cands = []
for st in range(0, NWIN - n_w + 1, 50):
    inside = pk[(pk >= st + 50) & (pk < st + n_w - 50)]
    pr = prop["prominences"][np.isin(pk, inside)]
    seg_k, seg_p = yes_key[st:st + n_w], yes_pred[1][st:st + n_w]
    cc = np.corrcoef(seg_k, seg_p)[0, 1]
    cands.append((len(inside), float(pr.sum()), float(cc), st))
cands.sort(key=lambda c: (-c[0], -c[1]))
n_best = cands[0][0]
top = [c for c in cands if c[0] == n_best]
top.sort(key=lambda c: (-c[1] - 0.0 * c[2]))
st = top[0][3]
w_start, w_end = st / FS, (st + n_w) / FS
seg_k, seg_p = yes_key[st:st + n_w], yes_pred[1][st:st + n_w]
win = dict(start_s=w_start, end_s=w_end, start_s_trial_time=w_start + 0.19, n_pushes_prominent=int(n_best),
           push_prominence_threshold_nm=1.0 * sd_cell, window_mean_bias_seed1_nm=float(np.mean(seg_p - seg_k)),
           window_corr_seed1=float(np.corrcoef(seg_k, seg_p)[0, 1]), window_key_sd_nm=float(seg_k.std()), window_key_range_nm=float(seg_k.max() - seg_k.min()),
           window_r2_seed1=float(1 - np.sum((seg_p - seg_k) ** 2) / np.sum((seg_k - seg_k.mean()) ** 2)),
           n_pushes_in_whole_trial=int(len(pk)),
           push_peak_times_in_window_s=[float(x / FS) for x in pk if st <= x < st + n_w],
           method="peaks of the 0.15 s moving average of the key with prominence >= 1 SD of the trial and >= 1 s apart; 30 s window (0.5 s steps) "
                  "with the most peaks, ties -> largest summed prominence")
# the window must show the offset: seed-1 prediction above the truth at most of the time
win["fraction_of_window_prediction_above_truth_seed1"] = float(np.mean(seg_p > seg_k))
print("window", win)

# ------------------------------------------------------------------ numbers
si, pj = si_y, pj_y
E = {}
mm = metrics(yes_key, yes_pred[1])
ss_tot = float(np.sum((yes_key - yes_key.mean()) ** 2))
E["YES_-0.486_seed1"] = dict(
    r2=mm["r2"], r2_shape=mm["r2_shape"], bias_nm=mm["bias"], sd_truth_nm=mm["sd"], var_truth_nm2=mm["var"], mae_nm=mm["mae"], mae_shape_nm=mm["mae_shape"],
    peak_abs_truth_nm=mm["peak"], corr=mm["corr"],
    penalty_bias2_over_var=mm["bias"] ** 2 / mm["var"], r2_from_identity=mm["r2_shape"] - mm["bias"] ** 2 / mm["var"],
    identity_residual=mm["r2"] - (mm["r2_shape"] - mm["bias"] ** 2 / mm["var"]), ss_tot=ss_tot,
    bias_over_sd=mm["bias"] / mm["sd"], n_windows=int(len(yes_key)),
    mean_truth_nm=float(yes_key.mean()), mean_pred_nm=float(yes_pred[1].mean()),
    sign_convention="bias = mean(prediction - key); positive = prediction above the truth")
check("YES -0.486 seed 1: R2 matches the number in y01", abs(E["YES_-0.486_seed1"]["r2"] - (-0.263)) < 2e-3, f"R2 {mm['r2']:.4f} vs y01/storyboard -0.263")
for_idx = A["r2"][si_y, :, pj_y]
E["YES_-0.486_across_seeds"] = dict(
    r2_per_seed=[float(x) for x in for_idx], r2_min=float(for_idx.min()), r2_max=float(for_idx.max()), r2_range=float(for_idx.max() - for_idx.min()),
    r2_shape_per_seed=[float(x) for x in A["r2_shape"][si_y, :, pj_y]], bias_per_seed=[float(x) for x in A["bias"][si_y, :, pj_y]],
    mae_per_seed=[float(x) for x in A["mae"][si_y, :, pj_y]], corr_per_seed=[float(x) for x in A["corr"][si_y, :, pj_y]],
    r2_median=float(np.median(for_idx)), r2_shape_median=float(np.median(A["r2_shape"][si_y, :, pj_y])),
    bias_median=float(np.median(A["bias"][si_y, :, pj_y])), mae_median=float(np.median(A["mae"][si_y, :, pj_y])),
    seed_with_min_r2=int(np.argmin(for_idx)), seed_with_max_r2=int(np.argmax(for_idx)),
    bias_min=float(A["bias"][si_y, :, pj_y].min()), bias_max=float(A["bias"][si_y, :, pj_y].max()))
y01_range = (-2.0, 0.01)
check("YES -0.486 seed range", abs(for_idx.min() - y01_range[0]) < 0.05 and abs(for_idx.max() - y01_range[1]) < 0.01,
      f"min {for_idx.min():.3f}, max {for_idx.max():.3f} vs storyboard -2.0 to 0.01")

# bundle tightness at the strong position vs -0.486 (c10_b03)
def bundle(pred, key):
    spread = pred.max(0) - pred.min(0)
    sdx = pred.std(0)
    lv = pred.mean(1)
    return dict(median_over_time_of_max_minus_min_across_seeds_nm=float(np.median(spread)), median_over_time_of_sd_across_seeds_nm=float(np.median(sdx)),
                seed_mean_level_range_nm=float(lv.max() - lv.min()), seed_mean_levels_nm=[float(x) for x in lv],
                truth_sd_nm=float(key.std()), truth_mean_nm=float(key.mean()))


E["bundle_YES_-0.486"] = bundle(yes_pred, yes_key)
E["bundle_YES_strong"] = bundle(ys_pred, ys_key)
E["YES_strong_cell"] = dict(position_rad=float(POS[si_y, pj_ys]), sd_truth_nm=float(C["sd_truth"][si_y, pj_ys]), peak_nm=float(C["peak_truth"][si_y, pj_ys]),
                            r2_per_seed=[float(x) for x in A["r2"][si_y, :, pj_ys]], r2_range=float(C["r2_range"][si_y, pj_ys]),
                            r2_median=float(C["r2_med"][si_y, pj_ys]), bias_median=float(C["bias_med"][si_y, pj_ys]), mae_median=float(C["mae_med"][si_y, pj_ys]),
                            toes_down=int(TD[si_y, pj_ys]),
                            all_YES_cells_sd=[float(x) for x in C["sd_truth"][si_y]], all_YES_positions=[float(x) for x in POS[si_y]])

E["JM_-0.078_seed0"] = dict(
    r2=mj["r2"], r2_shape=mj["r2_shape"], bias_nm=mj["bias"], mae_nm=mj["mae"], sd_truth_nm=mj["sd"], peak_abs_truth_nm=mj["peak"], corr=mj["corr"],
    r2_on_models_own_target_y_true_original=float(metrics(jm_orig, jm_pred[0])["r2"]), bias_vs_own_target_nm=float(metrics(jm_orig, jm_pred[0])["bias"]),
    mean_key_nm=float(jm_key.mean()), mean_y_true_original_nm=float(jm_orig.mean()), mean_pred_nm=float(jm_pred[0].mean()),
    key_minus_own_target_mean_nm=float(np.mean(jm_key - jm_orig)), key_minus_own_target_min_nm=float((jm_key - jm_orig).min()), key_minus_own_target_max_nm=float((jm_key - jm_orig).max()),
    median_r2_over_seeds=float(C["r2_med"][si_j, pj_j]), r2_per_seed=[float(x) for x in A["r2"][si_j, :, pj_j]], median_bias=float(C["bias_med"][si_j, pj_j]),
    r2_shape_median=float(C["r2_shape_med"][si_j, pj_j]), toes_down=int(TD[si_j, pj_j]),
    note="y_true_original is the model's own training-time target (measured minus the ORIGINAL passive table, which has JM's bad entry); the key replaces it by the clean static table")
E["EG_-0.739_seed4"] = dict(
    r2=metrics(eg_key, eg_pred[0])["r2"], r2_shape=metrics(eg_key, eg_pred[0])["r2_shape"], bias_nm=metrics(eg_key, eg_pred[0])["bias"],
    mae_nm=metrics(eg_key, eg_pred[0])["mae"], sd_truth_nm=float(eg_key.std()), peak_abs_truth_nm=float(np.abs(eg_key).max()),
    corr=metrics(eg_key, eg_pred[0])["corr"], median_r2_over_seeds=float(C["r2_med"][si_e, pj_e]),
    r2_per_seed=[float(x) for x in A["r2"][si_e, :, pj_e]], toes_down=int(TD[si_e, pj_e]))

# storyboard cells (named in c12/c14)
named = {}
for s in SUBJ:
    for pj in range(NPOS):
        si_ = SUBJ.index(s)
        if TD[si_, pj] or (s == "JM" and abs(POS[si_, pj] + 0.078) < 0.002):
            named[f"{s}_{POS[si_, pj]:.3f}"] = dict(
                r2_med=float(C["r2_med"][si_, pj]), r2_shape_med=float(C["r2_shape_med"][si_, pj]), bias_med=float(C["bias_med"][si_, pj]),
                mae_med=float(C["mae_med"][si_, pj]), sd_truth=float(C["sd_truth"][si_, pj]), peak=float(C["peak_truth"][si_, pj]),
                corr_med=float(C["corr_med"][si_, pj]), r2_range=float(C["r2_range"][si_, pj]), toes_down=int(TD[si_, pj]))

# per-cell table (all 48)
cell_rows = []
for si_, s in enumerate(SUBJ):
    for pj in range(NPOS):
        cell_rows.append(dict(subject=s, position_rad=float(POS[si_, pj]), toes_down=int(TD[si_, pj]),
                              r2_med=float(C["r2_med"][si_, pj]), r2_floor0_med=float(C["r2_floor0_med"][si_, pj]), r2_shape_med=float(C["r2_shape_med"][si_, pj]),
                              bias_med=float(C["bias_med"][si_, pj]), mae_med=float(C["mae_med"][si_, pj]), sd_truth=float(C["sd_truth"][si_, pj]),
                              peak=float(C["peak_truth"][si_, pj]), corr_med=float(C["corr_med"][si_, pj]), r2_min=float(C["r2_min"][si_, pj]),
                              r2_max=float(C["r2_max"][si_, pj]), r2_range=float(C["r2_range"][si_, pj]), n_seeds_negative=int(C["n_neg"][si_, pj])))

# lowest R2 cells: is every cell below some SD? scatter summary (c14_b01)
cdf = pd.DataFrame(cell_rows)
scatter = dict(
    spearman_sd_vs_r2_all=float(cdf.sd_truth.rank().corr(cdf.r2_med.rank())),
    n_cells_r2_med_below_0p5=int((cdf.r2_med < 0.5).sum()),
    sd_of_cells_with_r2_med_below_0p5=[float(x) for x in cdf[cdf.r2_med < 0.5].sd_truth.sort_values()],
    max_sd_among_cells_with_r2_med_below_0p5=float(cdf[cdf.r2_med < 0.5].sd_truth.max()),
    min_sd_among_cells_with_r2_med_at_least_0p9=float(cdf[cdf.r2_med >= 0.9].sd_truth.min()),
    cells_off_the_bottom_at_minus1=[f"{r.subject}@{r.position_rad:+.3f} R2 {r.r2_med:.2f}" for r in cdf[cdf.r2_med < -1].itertuples()],
    lowest_r2_cell=dict(cdf.loc[cdf.r2_med.idxmin(), ["subject", "position_rad", "r2_med", "sd_truth"]].to_dict()))

sdq = [float(np.percentile(C["sd_truth"][oth], q)) for q in (25, 75)]
sd_strong_med = float(np.median(C["sd_truth"][oth & (C["peak_truth"] > 15)]))
scatter["sd_truth_other_36_quartiles_nm"] = sdq
scatter["sd_truth_other_36_p90_nm"] = float(np.percentile(C["sd_truth"][oth], 90))
scatter["sd_truth_median_of_other_cells_with_peak_above_15nm"] = sd_strong_med
scatter["n_other_cells_with_peak_above_15nm"] = int((oth & (C["peak_truth"] > 15)).sum())
scatter["sd_truth_median_of_each_persons_strongest_cell"] = float(np.median(C["sd_truth"].max(1)))
scatter["sd_truth_of_each_persons_strongest_cell"] = {SUBJ[i]: float(C["sd_truth"][i].max()) for i in range(6)}

# ------------------------------------------------------------------ storyboard comparison
sbm = []
def sb(label, story, mine_, ok):
    if not ok:
        sbm.append(f"{label}: storyboard {story} vs y03 {mine_}")
    checks.append({"check": f"storyboard: {label}", "ok": bool(ok), "detail": f"storyboard {story}; y03 {mine_}"})
    print(("OK   " if ok else "DIFF ") + f"storyboard {label}: {story} vs {mine_}")


e1 = E["YES_-0.486_seed1"]
sb("c01_b02 YES -0.486 seed 1 R2", "-0.26", round(e1["r2"], 3), abs(e1["r2"] + 0.26) < 0.005)
sb("c12_b01 SD of truth", "0.40 Nm", round(e1["sd_truth_nm"], 3), abs(e1["sd_truth_nm"] - 0.40) < 0.005)
sb("c12_b01 offset", "+0.38 Nm", round(e1["bias_nm"], 3), abs(e1["bias_nm"] - 0.38) < 0.005)
sb("c12_b01 MAE", "0.40 Nm", round(e1["mae_nm"], 3), abs(e1["mae_nm"] - 0.40) < 0.005)
sb("c12_b02 shape-only R2", "0.66", round(e1["r2_shape"], 3), abs(e1["r2_shape"] - 0.66) < 0.005)
sb("c12_b03 penalty", "~0.9", round(e1["penalty_bias2_over_var"], 3), abs(e1["penalty_bias2_over_var"] - 0.9) < 0.05)
sb("c10_b04 R2 range toes-down", "0.42", round(groups["toes_down_12"]["r2_range_median"], 3), abs(groups["toes_down_12"]["r2_range_median"] - 0.42) < 0.005)
sb("c10_b04 R2 range other", "0.02", round(groups["other_36"]["r2_range_median"], 3), abs(groups["other_36"]["r2_range_median"] - 0.02) < 0.005)
sb("c10_b04 YES -0.486 min", "-2.0", round(E["YES_-0.486_across_seeds"]["r2_min"], 3), abs(E["YES_-0.486_across_seeds"]["r2_min"] + 2.0) < 0.05)
sb("c10_b04 YES -0.486 max", "0.01", round(E["YES_-0.486_across_seeds"]["r2_max"], 3), abs(E["YES_-0.486_across_seeds"]["r2_max"] - 0.01) < 0.005)
sb("c11_b06 R2 toes-down", "0.77", round(groups["toes_down_12"]["r2_floor0_median"], 3), abs(groups["toes_down_12"]["r2_floor0_median"] - 0.77) < 0.005)
sb("c11_b06 R2 other", "0.97", round(groups["other_36"]["r2_floor0_median"], 3), abs(groups["other_36"]["r2_floor0_median"] - 0.97) < 0.005)
sb("c11_b06 MAE toes-down", "0.83 Nm", round(groups["toes_down_12"]["mae_median_nm"], 3), abs(groups["toes_down_12"]["mae_median_nm"] - 0.83) < 0.005)
sb("c11_b06 MAE other", "1.07 Nm", round(groups["other_36"]["mae_median_nm"], 3), abs(groups["other_36"]["mae_median_nm"] - 1.07) < 0.005)
sb("c11_b06 peak toes-down", "6 Nm", round(groups["toes_down_12"]["peak_median_nm"], 2), abs(groups["toes_down_12"]["peak_median_nm"] - 6.05) < 0.1)
sb("c11_b06 peak other", "19 Nm", round(groups["other_36"]["peak_median_nm"], 2), abs(groups["other_36"]["peak_median_nm"] - 19.0) < 0.1)
sb("c11_b06 MAE/peak toes-down", "13 %", round(100 * groups["toes_down_12"]["mae_over_peak_median"], 2), abs(100 * groups["toes_down_12"]["mae_over_peak_median"] - 13.1) < 0.3)
sb("c11_b06 MAE/peak other", "4.6 %", round(100 * groups["other_36"]["mae_over_peak_median"], 2), abs(100 * groups["other_36"]["mae_over_peak_median"] - 4.6) < 0.3)
sb("c12_b04 median |offset| toes-down", "0.53 Nm", round(groups["toes_down_12"]["abs_bias_median_nm"], 3), abs(groups["toes_down_12"]["abs_bias_median_nm"] - 0.53) < 0.005)
sb("c12_b04 median |offset| other", "0.61 Nm", round(groups["other_36"]["abs_bias_median_nm"], 3), abs(groups["other_36"]["abs_bias_median_nm"] - 0.61) < 0.005)
sb("c12_b05 offset share of lost R2 toes-down", "55 %", round(100 * groups["toes_down_12"]["offset_share_of_lost_r2_median_per_run"], 1),
   abs(100 * groups["toes_down_12"]["offset_share_of_lost_r2_median_per_run"] - 55) < 1.5)
sb("c12_b05 offset share of lost R2 other", "21 %", round(100 * groups["other_36"]["offset_share_of_lost_r2_median_per_run"], 1),
   abs(100 * groups["other_36"]["offset_share_of_lost_r2_median_per_run"] - 21) < 1.5)
sb("c12_b05 shape-only toes-down median", "0.91", round(groups["toes_down_12"]["r2_shape_median"], 3), abs(groups["toes_down_12"]["r2_shape_median"] - 0.91) < 0.005)
for key_, story in [("YES_-0.486", 0.42), ("JM_-0.644", 0.59), ("EG_-0.739", 0.69)]:
    v = named[key_]["r2_shape_med"]
    sb(f"c12_b06 shape-only R2 {key_}", str(story), round(v, 3), abs(v - story) < 0.005)
sb("c12_b06 corr range toes-down cells", "0.80-0.99", f"{groups['toes_down_12']['corr_min']:.3f}-{groups['toes_down_12']['corr_max']:.3f}",
   abs(groups["toes_down_12"]["corr_min"] - 0.80) < 0.01 and abs(groups["toes_down_12"]["corr_max"] - 0.99) < 0.01)
jmv = E["JM_-0.078_seed0"]
sb("c12_b07 JM -0.078 offset (seed 0, as plotted)", "-12.4 Nm", f"seed 0 {jmv['bias_nm']:.2f}; median over seeds {jmv['median_bias']:.2f}", abs(jmv["bias_nm"] + 12.4) < 0.05)
sb("c12_b07 JM -0.078 peak", "5.1 Nm", round(jmv["peak_abs_truth_nm"], 2), abs(jmv["peak_abs_truth_nm"] - 5.1) < 0.05)
sb("c11_b07 JM -0.078 median R2", "-54", f"median over seeds {jmv['median_r2_over_seeds']:.2f} (seed 0: {jmv['r2']:.2f})", abs(jmv["median_r2_over_seeds"] + 54.3) < 0.1)
sb("c12_b07 JM -0.078 R2 of the plotted seed 0", "-54", round(jmv["r2"], 2), abs(jmv["r2"] + 54) < 0.5)
sb("c12_b07 JM -0.078 shape R2 (y01)", "0.687", round(jmv["r2_shape_median"], 3), abs(jmv["r2_shape_median"] - 0.687) < 0.01)
for key_, story in [("YES_-0.486", 0.40), ("HM_-0.789", 0.77), ("EG_-0.739", 2.06)]:
    v = named[key_]["sd_truth"]
    sb(f"c14_b02 SD of truth {key_}", str(story), round(v, 3), abs(v - story) < 0.05 or (key_ == "EG_-0.739" and abs(v - 2.1) < 0.05))
jm_td = sorted(v["sd_truth"] for k, v in named.items() if k.startswith("JM") and v["toes_down"])
sb("c14_b02 SD of truth JM toes-down cells", "0.60 and 0.78 Nm", [round(x, 3) for x in jm_td],
   len(jm_td) == 2 and abs(jm_td[0] - 0.60) < 0.02 and abs(jm_td[1] - 0.78) < 0.02)
sb("c14_b02 SD of truth at strong positions", "~15 Nm", f"median of the 36 other cells {groups['other_36']['sd_truth_median_nm']:.1f} (quartiles {sdq[0]:.1f}-{sdq[1]:.1f}); "
   f"YES strongest cell {E['YES_strong_cell']['sd_truth_nm']:.1f}; cells with peak > 15 Nm: median {sd_strong_med:.1f}",
   abs(groups["other_36"]["sd_truth_median_nm"] - 15) < 3)
sb("c12_b04 spread at a strong position ~10 Nm", "~10 Nm", f"median SD of the 36 other cells {groups['other_36']['sd_truth_median_nm']:.1f}; of cells with peak > 15 Nm {sd_strong_med:.1f}", abs(sd_strong_med - 10) < 2)
sb("c12_b04 penalty (0.38/10)^2", "0.0014", round((0.38 / 10) ** 2, 4), abs((0.38 / 10) ** 2 - 0.0014) < 0.0001)

# ------------------------------------------------------------------ write
display = {
    "yes_s1_r2": f"{e1['r2']:.2f}", "yes_s1_r2dm": f"{e1['r2_shape']:.2f}", "yes_s1_bias": f"{e1['bias_nm']:+.2f}", "yes_sd": f"{e1['sd_truth_nm']:.2f}",
    "yes_s1_mae": f"{e1['mae_nm']:.2f}", "yes_penalty": f"{e1['penalty_bias2_over_var']:.2f}",
    "range_toes_down": f"{groups['toes_down_12']['r2_range_median']:.2f}", "range_other": f"{groups['other_36']['r2_range_median']:.2f}",
    "yes_m486_min": f"{E['YES_-0.486_across_seeds']['r2_min']:.1f}", "yes_m486_max": f"{E['YES_-0.486_across_seeds']['r2_max']:.2f}",
}
facts = {"y03": dict(
    docstring="see the module docstring of extract2/y03_predictions.py",
    subjects=SUBJ, seeds=SEEDS, n_windows_per_trial=NWIN, seconds_per_window=1 / FS,
    conventions=dict(angle="negative = plantarflexion = toes down", bias="mean(prediction - key); positive = prediction above the truth",
                     r2="1 - SS_res/SS_tot against the cell's own mean of the key", r2_shape="R2 after removing the mean of both key and prediction",
                     identity="R2 = R2_shape - bias^2/var(key)", key="passive_trial (score.py): y_true_original + legacy_S2(angle) - static_S2(angle); zero correction for YES and IES",
                     time="t_window_s = i*0.01; t_trial_s = (i+19)*0.01"),
    passive_correction_by_subject=corr_info,
    npz_arrays_note="see facts['y03']['npz_arrays']",
    cross_check_results_json=dict(n_runs=int(len(mine)), max_abs_diff=diffs, max_relative_r2_diff=rel, tolerance=1e-3),
    groups=groups, named_cells=named, cells=cell_rows, scatter_summary=scatter,
    YES_minus0p486_seed1=E["YES_-0.486_seed1"], YES_minus0p486_across_seeds=E["YES_-0.486_across_seeds"],
    bundle_YES_minus0p486=E["bundle_YES_-0.486"], bundle_YES_strong=E["bundle_YES_strong"], YES_strong_cell=E["YES_strong_cell"],
    JM_minus0p078_seed0=E["JM_-0.078_seed0"], EG_minus0p739_seed4=E["EG_-0.739_seed4"],
    suggested_30s_window_YES_minus0p486=win,
    display=display, checks=checks, storyboard_mismatches=sbm,
)}

arrays = dict(
    subjects=np.array(SUBJ), seeds=np.array(SEEDS), positions_rad=POS, positions_snapped_rad=POS_SNAP, toes_down=TD,
    r2=A["r2"], r2_shape=A["r2_shape"], bias=A["bias"], mae=A["mae"], corr=A["corr"], mae_shape=A["mae_shape"],
    sd_truth=C["sd_truth"], peak_truth=C["peak_truth"],
    r2_med=C["r2_med"], r2_floor0_med=C["r2_floor0_med"], r2_shape_med=C["r2_shape_med"], bias_med=C["bias_med"], mae_med=C["mae_med"], corr_med=C["corr_med"],
    r2_min=C["r2_min"], r2_max=C["r2_max"], r2_range=C["r2_range"], n_seeds_negative=C["n_neg"],
    t_window_s=t_window, t_trial_s=t_trial,
    yes_m486_key=yes_key, yes_m486_pred=yes_pred, yes_m486_position=np.array(POS[si_y, pj_y]),
    yes_strong_key=ys_key, yes_strong_pred=ys_pred, yes_strong_position=np.array(POS[si_y, pj_ys]),
    jm_m078_key=jm_key, jm_m078_pred_seed0=jm_pred[0], jm_m078_y_true_original=jm_orig,
    eg_m739_key=eg_key, eg_m739_pred_seed4=eg_pred[0],
    window_yes_m486_start_end_s=np.array([w_start, w_end]),
)
meaning = {
    "subjects": ("(6,)", "subject codes in array order CH EG HM IES JM YES"), "seeds": ("(10,)", "seed numbers 0-9"),
    "positions_rad": ("(6,8)", "cell position label (rad, negative = toes down), sorted most negative first"),
    "positions_snapped_rad": ("(6,8)", "raw snapped angle of the cell in the npz"), "toes_down": ("(6,8)", "1 for each person's 2 most negative positions"),
    "r2": ("(6,10,8)", "R2 per subject x seed x position (key = passive_trial)"), "r2_shape": ("(6,10,8)", "R2 after removing the mean of key and prediction"),
    "bias": ("(6,10,8)", "mean(prediction - key) in Nm; positive = prediction above truth"), "mae": ("(6,10,8)", "mean absolute error, Nm"),
    "corr": ("(6,10,8)", "Pearson correlation prediction vs key"), "mae_shape": ("(6,10,8)", "MAE after demeaning both"),
    "sd_truth": ("(6,8)", "np.std of the key in the cell, Nm"), "peak_truth": ("(6,8)", "max |key| in the cell, Nm"),
    "r2_med": ("(6,8)", "median over the 10 seeds of R2"), "r2_floor0_med": ("(6,8)", "median over seeds of max(R2,0) (dashboard display)"),
    "r2_shape_med": ("(6,8)", "median over seeds of shape-only R2"), "bias_med": ("(6,8)", "median over seeds of bias"), "mae_med": ("(6,8)", "median over seeds of MAE"),
    "corr_med": ("(6,8)", "median over seeds of correlation"), "r2_min": ("(6,8)", "min R2 over seeds"), "r2_max": ("(6,8)", "max R2 over seeds"),
    "r2_range": ("(6,8)", "max - min R2 over the 10 seeds"), "n_seeds_negative": ("(6,8)", "number of seeds with R2 < 0"),
    "t_window_s": ("(8891,)", "i*0.01 s, time of window i from the first window"), "t_trial_s": ("(8891,)", "(i+19)*0.01 s, time from the start of the trial"),
    "yes_m486_key": ("(8891,)", "YES -0.486 rad session-2 trial: answer key (passive_trial), Nm"), "yes_m486_pred": ("(10,8891)", "its predictions, seeds 0-9"),
    "yes_m486_position": ("()", "position label"), "yes_strong_key": ("(8891,)", "YES strongest position (largest SD of key): key"),
    "yes_strong_pred": ("(10,8891)", "its predictions, seeds 0-9"), "yes_strong_position": ("()", "position label of the strongest YES cell"),
    "jm_m078_key": ("(8891,)", "JM -0.078 rad: passive_trial key"), "jm_m078_pred_seed0": ("(8891,)", "seed 0 prediction"),
    "jm_m078_y_true_original": ("(8891,)", "the model's own (buggy) target at that position"),
    "eg_m739_key": ("(8891,)", "EG -0.739 rad: key"), "eg_m739_pred_seed4": ("(8891,)", "seed 4 prediction"),
    "window_yes_m486_start_end_s": ("(2,)", "suggested 30 s piece [start, end] in t_window_s"),
}
facts["y03"]["npz_arrays"] = {k: dict(shape=str(arrays[k].shape), meaning=meaning[k][1]) for k in arrays}
assert set(arrays) == set(meaning), set(arrays) ^ set(meaning)


def plain(v):
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


np.savez_compressed(OUT_DIR / "y03.npz", **{k: np.asarray(v) for k, v in arrays.items()})
json.dump(plain(facts), open(OUT_DIR / "facts_y03.json", "w", encoding="utf-8"), indent=1)
assert all(c["ok"] for c in checks if not c["check"].startswith("storyboard")), [c for c in checks if not c["ok"] and not c["check"].startswith("storyboard")]
print("\nwrote", OUT_DIR / "y03.npz", "and", OUT_DIR / "facts_y03.json")
print("storyboard mismatches:", len(sbm))
for m in sbm:
    print("  ", m)
