"""y05: the PASSIVE TABLE, the JM bug, YES's offset and the loss share of the toes-down positions
(storyboard chapters c03, c10, c14, c15).

Reads
  recordings            EMG_DATA_DIR (default <main checkout>/ML/subjects), .flb files of the six people
  frozen baseline code  DWT/EXPERIMENTS/baseline/original_code/ML   (imported FIRST, as the package ``ML``):
                          dataset_builder._split_passive_segments, get_passive_torque_map,
                          lookup_passive_torque, downsample_trial, classify_trials, config (YES worksheet values)
  framework code        DWT/ML/research + DWT/ML/preprocessing  (imported AFTER the frozen modules were purged):
                          active_torque.static_hold_passive_map (the static / "clean" rule), research.data
                          (cached bundles, _split_trials), passive.PassiveModel, truth (EMG-rest baseline),
                          splits.build (fit-set windows)
  saved runs            DWT/EXPERIMENTS/results/X00_baseline/{JM,YES}/S1toS2/seed0.npz   (check of the answer key)
                        DWT/EXPERIMENTS/results/learning_curves/X00f_baseline_framework/<subj>/S1toS2/seed0.json
Writes
  EXPERIMENTS/pipeline_video/data2/y05.npz            arrays (names in facts['npz_arrays'])
  EXPERIMENTS/pipeline_video/data2/facts_y05.json     {"y05": {...}}   (facts['display'] holds rounded numbers)
Run    (no training, ~1-2 min)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
  C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.venv311/Scripts/python.exe extract2/y05_passive.py

Conventions
  * Trial numbers: ``trial_index`` is 0-based as in the code; ``file_trial`` = trial_index + 1 is the number used in the
    storyboard ("#21").  Angle: negative = plantarflexion = toes down.
  * Legacy rule (frozen code): a piece of a passive trial is kept when torque SD (pandas, ddof=1) <= 20 Nm and
    |mean torque| <= 20 Nm; kept pieces within 0.05 rad are merged by running mean.
  * Static ("clean") rule (framework): piece >= 5 s, angle SD <= 0.005 rad, torque SD (numpy, ddof=0) <= 1 Nm.
  * 100 Hz traces: the study's own FIR decimation by 10 (scipy.signal.decimate(ftype='fir', zero_phase=True),
    dataset_builder.downsample_trial).  Whole passive trials are decimated WITHOUT the 1 s trim, time = index/100 s
    from the first sample.  Active trials are trimmed by 1 s first (as the pipeline does): time = seconds since the
    start of the recording (first sample 1.0 s), exactly the bundle's ``t``.
  * "Passive value subtracted" = mean over the 8891 window-end samples (samples 19..end of the 8910-sample trial)
    of table(position), i.e. over the targets the network is trained / scored on.
  * Relaxed level between pushes (YES): ``truth.session_rest`` finds EMG-silent stretches (Schmitt trigger on the
    4 envelopes; the study's own rest detector); 'rest_median' = median torque over all samples inside them;
    'B_mean' = mean of the smooth in-trial baseline B(t) (Gaussian kernel, sigma 8 s, through the median torque of
    each rest stretch; truth.baseline).  Both use measured torque only inside EMG-silent stretches.
"""
import json
import os
import pathlib
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
VWT = HERE.parents[2]
OUT = VWT / "EXPERIMENTS/pipeline_video/data2"
OUT.mkdir(parents=True, exist_ok=True)
PROJ = pathlib.Path("C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling")
DWT = PROJ / ".claude/worktrees/dorsiflexion-r2-drop-341137"
FROZEN = DWT / "EXPERIMENTS/baseline/original_code"
DATA_DIR = pathlib.Path(os.environ.get("EMG_DATA_DIR", PROJ / "ML/subjects"))
os.environ["EMG_DATA_DIR"] = str(DATA_DIR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

SUBJ = ["CH", "EG", "HM", "IES", "JM", "YES"]
LC_EXP = "X00f_baseline_framework"

facts, A, checks = {}, {}, []


def chk(name, ok, detail):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print(("OK   " if ok else "FAIL ") + name + " :: " + str(detail))


def r(x, n=3):
    return None if x is None else round(float(x), n)


def purge_ml():
    for k in [k for k in sys.modules if k == "ML" or k.startswith("ML.")]:
        del sys.modules[k]


# =====================================================================================================
# PHASE 1: the FROZEN baseline code  (legacy rule, YES worksheet values, JM bug, 100 Hz traces)
# =====================================================================================================
sys.path.insert(0, str(FROZEN))
import ML.config as fcfg                                                    # noqa: E402
assert pathlib.Path(fcfg.__file__).resolve().is_relative_to(FROZEN), fcfg.__file__
from ML.preprocessing import dataset_builder as fdb                          # noqa: E402
from ML.preprocessing.flb_reader import read_flb as fread                    # noqa: E402

PCOL, TCOL = fdb.POSITION_COL, fdb.TORQUE_COL
STD_LIM, MEAN_LIM, POS_TOL = fdb._PASSIVE_STD_THRESHOLD, fdb._PASSIVE_TQ_BOUND, fdb._POSITION_TOLERANCE


def seg_stats(seg):
    p, q = seg[PCOL].values, seg[TCOL].values
    return dict(start=int(seg.index[0]), end=int(seg.index[-1]) + 1, n=int(len(q)),
                mean_angle=float(p.mean()), angle_sd_rad=float(p.std()),
                mean_tq=float(q.mean()), tq_sd_ddof1=float(q.std(ddof=1)), tq_sd_ddof0=float(q.std()))


def legacy_accept(st):
    """the two tests of get_passive_torque_map (strictly greater rejects)"""
    if st["tq_sd_ddof1"] > STD_LIM:
        return False, "torque SD > 20 Nm"
    if abs(st["mean_tq"]) > MEAN_LIM:
        return False, "|mean torque| > 20 Nm"
    return True, ""


def merge_clusters(raw):
    """same running-mean merge as get_passive_torque_map, but remembers the members. raw: [(pos, tq, key)]"""
    raw = sorted(raw, key=lambda x: x[0])
    cl = []
    for pos, tq, key in raw:
        if cl and abs(pos - cl[-1][0]) <= POS_TOL:
            pp, tt, c, m = cl[-1]
            cl[-1] = ((pp * c + pos) / (c + 1), (tt * c + tq) / (c + 1), c + 1, m + [key])
        else:
            cl.append((pos, tq, 1, [key]))
    return cl


def lookup_series(entries, pos):
    return np.array([fdb.lookup_passive_torque(entries, float(p)) for p in pos])


def ds_trial(df, trim=0):
    """the pipeline's trim + downsample on (time, position, torque) only; returns a DataFrame at 100 Hz"""
    d = df[["time", PCOL, TCOL]].iloc[trim:].reset_index(drop=True)
    d.attrs.update(df.attrs)
    return fdb.downsample_trial(d, factor=10)


leg = {}                       # per subject: maps, segment table
jm = {}
yes = {}
for s in SUBJ:
    fn, sid = fcfg.SUBJECTS[s]
    trials = fread(str(DATA_DIR / fn), subject_id=sid)
    mvc, passive, active = fdb.classify_trials(trials)
    ov_idx = fcfg.TRIAL_OVERRIDES.get(s)
    if ov_idx:                                         # build_dataset rescue rule
        forced = {i - 1 for k in ("test", "retest") for i in ov_idx.get(k, [])}
        passive = [d for d in passive if d.attrs.get("trial_index") not in forced]
    p1 = [d for d in passive if d.attrs.get("session") != "retest"]
    p2 = [d for d in passive if d.attrs.get("session") == "retest"]
    ws = fcfg.PASSIVE_TORQUE_OVERRIDES.get(s)
    maps, members = {}, {}
    for sess, plist in (("S1", p1), ("S2", p2)):
        rows = []
        for d in plist:
            for seg in fdb._split_passive_segments(d):
                st = seg_stats(seg)
                acc, why = legacy_accept(st)
                comment = d.attrs.get("comment", "")
                rows.append(dict(trial_index=int(d.attrs["trial_index"]), file_trial=int(d.attrs["trial_index"]) + 1,
                                 comment=comment, session=sess, legacy_accepted=acc, legacy_reject_reason=why, **st))
        leg.setdefault(s, {}).setdefault("segments", []).extend(rows)
        if ws:                                              # worksheet replaces the auto-detected table
            maps[sess] = sorted(ws["test" if sess == "S1" else "retest"], key=lambda x: x[0])
        elif sess == "S1" or p2:
            maps[sess] = fdb.get_passive_torque_map(plist)
        else:
            maps[sess] = maps["S1"]                       # no retest passive trials: session-1 table is reused
        # members of each table entry (only for the auto-detected case)
        acc_rows = [x for x in rows if x["legacy_accepted"]]
        cl = merge_clusters([(x["mean_angle"], x["mean_tq"], (x["trial_index"], x["start"])) for x in acc_rows])
        members[sess] = [dict(angle=c[0], tq=c[1], members=[list(k) for k in c[3]]) for c in cl]
        if not ws and plist:
            ref = np.array(maps[sess])
            mine = np.array([(c[0], c[1]) for c in cl])
            ok = ref.shape == mine.shape and (ref.size == 0 or np.max(np.abs(ref - mine)) < 1e-12)
            chk(f"legacy_table_rebuild_{s}_{sess}", ok, "table rebuilt from the kept pieces == get_passive_torque_map output")
    leg[s]["maps"] = maps
    leg[s]["members"] = members
    leg[s]["worksheet"] = bool(ws)

    if s == "JM":
        # ---- the two passive trials at p7 and the two active p7 trials, from the frozen pipeline ----
        def find(kind, sess):
            for d in trials:
                k, ss, pos = fdb._parse_comment(d.attrs.get("comment", ""))
                if k == kind and pos == "p7" and ((ss == "retest") == (sess == "S2")):
                    return d
        for sess in ("S1", "S2"):
            dp, da = find("passive", sess), find("active", sess)
            jm[sess] = {"passive_idx": int(dp.attrs["trial_index"]), "active_idx": int(da.attrs["trial_index"]),
                        "passive_comment": dp.attrs["comment"], "active_comment": da.attrs["comment"]}
            q1k, p1k = dp[TCOL].values, dp[PCOL].values
            nb = len(q1k) // 10000
            jm[sess]["block_mean"] = [float(q1k[i * 10000:(i + 1) * 10000].mean()) for i in range(nb)]
            jm[sess]["block_sd"] = [float(q1k[i * 10000:(i + 1) * 10000].std(ddof=1)) for i in range(nb)]
            jm[sess]["block_dropped_samples"] = int(len(q1k) - nb * 10000)
            sec1 = [float(q1k[i * 1000:(i + 1) * 1000].std(ddof=1)) for i in range(len(q1k) // 1000)]
            jm[sess]["first_s_with_sd_gt_1Nm"] = int(next(i for i, v in enumerate(sec1) if v > 1.0))
            jm[sess]["hold0_20s_mean"] = float(q1k[:20000].mean())
            jm[sess]["hold0_20s_sd_ddof1"] = float(q1k[:20000].std(ddof=1))
            jm[sess]["hold0_20s_angle_sd_rad"] = float(p1k[:20000].std())
            jm[sess]["rest_70s_mean_sd_ddof1"] = [float(q1k[20000:].mean()), float(q1k[20000:].std(ddof=1))]
            jm[sess]["rest_70s_min_max"] = [float(q1k[20000:].min()), float(q1k[20000:].max())]
            jm[sess]["whole_trial_angle_sd_rad"] = float(p1k.std())
            dd = ds_trial(dp)
            A[f"jmP7{sess}_t"] = np.arange(len(dd)) / 100.0
            A[f"jmP7{sess}_pos"] = dd[PCOL].values
            A[f"jmP7{sess}_tq"] = dd[TCOL].values
            A[f"jmP7{sess}_block10_mean"] = np.array(jm[sess]["block_mean"])
            A[f"jmP7{sess}_block10_sd"] = np.array(jm[sess]["block_sd"])
            # active p7 trial: pipeline trim 1 s, decimate, passive lookup at each sample
            da100 = ds_trial(da, trim=1000)
            pos, tq = da100[PCOL].values, da100[TCOL].values
            P = lookup_series(leg[s]["maps"][sess], pos)
            jm[sess]["active_pos100"], jm[sess]["active_tq100"], jm[sess]["active_P_legacy"] = pos, tq, P
            jm[sess]["legacy_P_mean_windowends"] = float(P[19:].mean())
            jm[sess]["legacy_P_mean_all"] = float(P.mean())
    if s == "YES":
        for sess, ti in (("S1", 20), ("S2", 36)):
            d = trials[ti]
            assert d.attrs["trial_index"] == ti
            d100 = ds_trial(d, trim=1000)
            pos, tq = d100[PCOL].values, d100[TCOL].values
            assert abs(pos.mean() + 0.486) < 0.003, pos.mean()
            tab = leg[s]["maps"][sess]
            yes[sess] = dict(trial_index=ti, comment=d.attrs.get("comment", ""), pos=pos, tq=tq,
                             P=lookup_series(tab, pos), table_at_m0487=[e[1] for e in tab if abs(e[0] + 0.487) < 1e-9][0])
            yes[sess]["P_mean_windowends"] = float(yes[sess]["P"][19:].mean())
    del trials, mvc, passive, active
    print("phase 1", s, "done")

lookup_frozen = fdb.lookup_passive_torque                 # keep the frozen function for later (it survives the purge)
purge_ml()
sys.path.remove(str(FROZEN))

# =====================================================================================================
# PHASE 2: the FRAMEWORK code (static rule, bundles, truth, learning-curve fit sets)
# =====================================================================================================
sys.path.insert(0, str(DWT))
import ML.config as dcfg                                                     # noqa: E402
assert pathlib.Path(dcfg.__file__).resolve().is_relative_to(DWT), dcfg.__file__
from ML.preprocessing import active_torque as at                              # noqa: E402
from ML.preprocessing import dataset_builder as ddb                           # noqa: E402
from ML.preprocessing.flb_reader import read_flb as dread                     # noqa: E402
from ML.research import config as RC                                          # noqa: E402
from ML.research import data as rdata                                         # noqa: E402
from ML.research import splits, truth                                         # noqa: E402
from ML.research.data import load_subject                                     # noqa: E402
from ML.research.passive import PassiveModel, map_lookup                      # noqa: E402

bundles = {s: load_subject(s) for s in SUBJ}

# ---- legacy tables: frozen code == bundle (framework "legacy") -------------------------------------
for s in SUBJ:
    for sess in ("S1", "S2"):
        a = np.array(leg[s]["maps"][sess], float).reshape(-1, 2)
        b = np.array(bundles[s]["maps"]["legacy"][sess], float).reshape(-1, 2)
        ok = a.shape == b.shape and (a.size == 0 or np.max(np.abs(a - b)) < 1e-9)
        chk(f"legacy_table_frozen_vs_bundle_{s}_{sess}", ok, f"{len(a)} entries" + ("" if ok else f" vs {len(b)}"))

# ---- static scan of every passive piece, + the framework's hold finder -----------------------------
seg_rows, holdB_rows, static_maps = {}, {}, {}
for s in SUBJ:
    fn, sid = dcfg.SUBJECTS[s]
    trials = dread(str(DATA_DIR / fn), subject_id=sid)
    mvc, passive, active, s1, s2 = rdata._split_trials(trials, s)
    rows = []
    for d in passive:
        sess = "S2" if d.attrs.get("session") == "retest" else "S1"
        for seg in ddb._split_passive_segments(d):
            p, q = seg[ddb.POSITION_COL].values, seg[ddb.TORQUE_COL].values
            long_ok = len(q) >= at.STATIC_MIN_DUR_S * 1000
            pos_ok, tq_ok = p.std() <= at.STATIC_MAX_POS_STD, q.std() <= at.STATIC_MAX_TQ_STD
            rows.append(dict(trial_index=int(d.attrs["trial_index"]), session=sess, start=int(seg.index[0]),
                             static_accepted=bool(long_ok and pos_ok and tq_ok), tq_sd_ddof0=float(q.std()),
                             tq_sd_ddof1=float(q.std(ddof=1)), angle_sd_rad=float(p.std()), n=int(len(q)),
                             mean_angle=float(p.mean()), mean_tq=float(q.mean())))
    seg_rows[s] = rows
    # static map rebuilt with the framework function
    p1 = [d for d in passive if d.attrs.get("session") != "retest"]
    p2 = [d for d in passive if d.attrs.get("session") == "retest"]
    sm = {}
    for sess, pl in (("S1", p1), ("S2", p2)):
        if pl:
            e = at.static_hold_passive_map(pl, ddb._split_passive_segments, ddb.POSITION_COL, ddb.TORQUE_COL,
                                           ddb._POSITION_TOLERANCE)
            sm[sess] = e if len(e) >= 2 else ddb.get_passive_torque_map(pl)
    if "S2" not in sm and "S1" in sm:
        sm["S2"] = sm["S1"]
    static_maps[s] = sm
    if bundles[s]["maps"]["worksheet"] is None and sm:
        for sess in ("S1", "S2"):
            a, b = np.array(sm[sess], float), np.array(bundles[s]["maps"]["static"][sess], float)
            ok = a.shape == b.shape and np.max(np.abs(a - b)) < 1e-9
            chk(f"static_table_rebuild_{s}_{sess}", ok, f"{len(a)} entries == cached bundle")
    # framework holds (data._find_holds): every non-active, non-MVC trial
    excl = {rec["idx"] for rec in bundles[s]["active"]}
    hb = []
    for d in trials:
        i = int(d.attrs["trial_index"])
        if i in excl or d.attrs.get("comment", "").lower().strip().startswith("mvc"):
            continue
        for seg in ddb._split_passive_segments(d):
            p, q = seg[ddb.POSITION_COL].values, seg[ddb.TORQUE_COL].values
            if len(q) < rdata.HOLD_MIN_DUR_S * RC.FS_RAW or p.std() > rdata.HOLD_MAX_POS_STD or q.std() > rdata.HOLD_MAX_TQ_STD:
                continue
            hb.append(dict(trial_index=i, tq_sd_ddof0=float(q.std()), mean_tq=float(q.mean()), mean_angle=float(p.mean()),
                           dur_s=len(q) / RC.FS_RAW))
    bh = bundles[s]["holds"]
    ok = len(hb) == len(bh) and all(abs(x["mean_tq"] - h["mean"]) < 1e-9 and x["trial_index"] == h["idx"] for x, h in zip(hb, bh))
    chk(f"framework_holds_rebuild_{s}", ok, f"{len(hb)} holds == cached bundle holds ({len(bh)})")
    for x, h in zip(hb, bh):
        x["session"] = h["session"]
    holdB_rows[s] = hb
    del trials, mvc, passive, active
    print("phase 2 scan", s, "done")

# ---------------------------------------------------------------------------------------------------
# (b) torque SD of every accepted static hold
# ---------------------------------------------------------------------------------------------------
fb = {}
setA = [(si, x) for si, s in enumerate(SUBJ) for x in seg_rows[s] if x["static_accepted"]]
setB = [(si, x) for si, s in enumerate(SUBJ) for x in holdB_rows[s]]
for tag, st in (("static_map_pieces", setA), ("framework_holds", setB)):
    sd = np.array([x["tq_sd_ddof0"] for _, x in st])
    fb[tag] = dict(n=len(st), sd_min=float(sd.min()), sd_max=float(sd.max()), sd_median=float(np.median(sd)),
                   sd_min_ddof1=float(min(x.get("tq_sd_ddof1", np.nan) for _, x in st)) if tag == "static_map_pieces" else None,
                   per_subject={s: dict(n=sum(1 for si, _ in st if si == k),
                                        sd_min=float(min([x["tq_sd_ddof0"] for si, x in st if si == k] or [np.nan])),
                                        sd_max=float(max([x["tq_sd_ddof0"] for si, x in st if si == k] or [np.nan])))
                                for k, s in enumerate(SUBJ)})
    fb[tag]["sd_quantiles_5_50_90_95_99"] = [float(v) for v in np.percentile(sd, [5, 50, 90, 95, 99])]
    fb[tag]["sd_top5"] = [dict(subject=SUBJ[si], session=x["session"], angle=x["mean_angle"], sd=x["tq_sd_ddof0"])
                          for si, x in sorted(st, key=lambda z: -z[1]["tq_sd_ddof0"])[:5]]
    fb[tag]["n_sd_le_0.35"] = int(np.sum(sd <= 0.35))
    fb[tag]["sd_max_without_top1"] = float(np.sort(sd)[-2])
    A[f"{tag}_sd"] = sd
    A[f"{tag}_mean_tq"] = np.array([x["mean_tq"] for _, x in st])
    A[f"{tag}_angle"] = np.array([x["mean_angle"] for _, x in st])
    A[f"{tag}_subject"] = np.array([si for si, _ in st])
    A[f"{tag}_session"] = np.array([1 if x["session"] == "S1" else 2 for _, x in st])
fb["note"] = ("sd = numpy std (ddof=0), the quantity tested by the static rule (<= 1.0 Nm). 'static_map_pieces' = pieces "
              "of passive-classified trials accepted by the rule (these build the static tables); 'framework_holds' = "
              "the framework's hold list (every non-active, non-MVC trial, so YES's mis-labelled trials are included). "
              "subject index into SUBJ = CH, EG, HM, IES, JM, YES; session 1/2.")
# legacy side for contrast
legacy_acc = [(s, x) for s in SUBJ if not leg[s]["worksheet"] for x in leg[s]["segments"] if x["legacy_accepted"]]
fb["legacy_accepted_pieces"] = dict(
    n=len(legacy_acc), sd_ddof1_max=float(max(x["tq_sd_ddof1"] for _, x in legacy_acc)),
    n_with_sd_gt_1Nm=int(sum(x["tq_sd_ddof1"] > 1.0 for _, x in legacy_acc)),
    which_gt_1Nm=[dict(subject=s, trial_index=x["trial_index"], file_trial=x["file_trial"], session=x["session"],
                       mean_angle=x["mean_angle"], mean_tq=x["mean_tq"], tq_sd_ddof1=x["tq_sd_ddof1"])
                  for s, x in legacy_acc if x["tq_sd_ddof1"] > 1.0])
A["legacy_pieces_sd_ddof1"] = np.array([x["tq_sd_ddof1"] for s in SUBJ if not leg[s]["worksheet"] for x in leg[s]["segments"]])
A["legacy_pieces_accepted"] = np.array([x["legacy_accepted"] for s in SUBJ if not leg[s]["worksheet"] for x in leg[s]["segments"]])
A["legacy_pieces_subject"] = np.array([si for si, s in enumerate(SUBJ) if not leg[s]["worksheet"] for _ in leg[s]["segments"]])
facts["b_static_hold_sd"] = fb

# ---------------------------------------------------------------------------------------------------
# (a) JM
# ---------------------------------------------------------------------------------------------------
jmb = bundles["JM"]
fa = {"trials": {}}
# segment table of all JM passive pieces, both rules
key = {(x["trial_index"], x["start"]): x for x in seg_rows["JM"]}
jm_rows = []
for x in leg["JM"]["segments"]:
    y = key[(x["trial_index"], x["start"])]
    assert y["n"] == x["n"]
    jm_rows.append(dict(trial_index=x["trial_index"], file_trial=x["file_trial"], comment=x["comment"], session=x["session"],
                        start_sample=x["start"], end_sample=x["end"], start_s=x["start"] / 1000.0, end_s=x["end"] / 1000.0,
                        mean_angle=x["mean_angle"], mean_tq=x["mean_tq"], tq_sd_ddof1=x["tq_sd_ddof1"],
                        tq_sd_ddof0=x["tq_sd_ddof0"], angle_sd_rad=x["angle_sd_rad"],
                        legacy_accepted=x["legacy_accepted"], legacy_reject_reason=x["legacy_reject_reason"],
                        static_accepted=y["static_accepted"]))
fa["segments_all"] = jm_rows
fa["segments_near_m007"] = [x for x in jm_rows if -0.12 < x["mean_angle"] < -0.03]
fa["table_entry_members"] = {sess: [dict(angle=c["angle"], tq=c["tq"],
                                         members=[dict(trial_index=k[0], file_trial=k[0] + 1, start_sample=k[1]) for k in c["members"]])
                                    for c in leg["JM"]["members"][sess] if -0.12 < c["angle"] < -0.03]
                             for sess in ("S1", "S2")}
for sess in ("S1", "S2"):
    j = jm[sess]
    fa["trials"][sess] = {k: v for k, v in j.items() if not k.startswith("active_") and k != "legacy_P_mean_all"}
    pt = next(x for x in jm_rows if x["trial_index"] == j["passive_idx"])
    fa["trials"][sess]["whole_trial_piece"] = pt
for sess in ("S1", "S2"):
    fa[f"legacy_table_{sess}"] = leg["JM"]["maps"][sess]
    fa[f"static_table_{sess}"] = static_maps["JM"][sess]
    A[f"jm_legacy_{sess}_pos"] = np.array([e[0] for e in leg["JM"]["maps"][sess]])
    A[f"jm_legacy_{sess}_tq"] = np.array([e[1] for e in leg["JM"]["maps"][sess]])
    A[f"jm_static_{sess}_pos"] = np.array([e[0] for e in static_maps["JM"][sess]])
    A[f"jm_static_{sess}_tq"] = np.array([e[1] for e in static_maps["JM"][sess]])
curve_ang = np.linspace(-0.9, 0.3, 241)
A["jm_curve_angle"] = curve_ang
for sess in ("S1", "S2"):
    A[f"jm_curve_legacy_{sess}"] = np.array([lookup_frozen(leg["JM"]["maps"][sess], float(a)) for a in curve_ang])
    A[f"jm_curve_static_{sess}"] = map_lookup(static_maps["JM"][sess], curve_ang)

# active p7 trials: what each rule subtracts
for sess in ("S1", "S2"):
    j = jm[sess]
    rec = next(x for x in jmb["active"] if x["idx"] == j["active_idx"])
    pos_ok = np.max(np.abs(rec["pos"] - j["active_pos100"])) < 1e-9 and np.max(np.abs(rec["tq"] - j["active_tq100"])) < 1e-9
    chk(f"jm_p7_{sess}_frozen_downsample_equals_bundle", pos_ok, "frozen trim+decimate == bundle position and torque")
    Pl = PassiveModel("legacy", jmb).predict(rec)
    Ps = PassiveModel("static", jmb).predict(rec)
    chk(f"jm_p7_{sess}_legacy_lookup_frozen_equals_framework", np.max(np.abs(Pl - j["active_P_legacy"])) < 1e-9,
        f"max abs diff {np.max(np.abs(Pl - j['active_P_legacy'])):.2e}")
    # independent static value: framework map rebuilt from the raw recording, frozen interpolation function
    Ps2 = np.array([lookup_frozen(static_maps["JM"][sess], float(p)) for p in rec["pos"]])
    chk(f"jm_p7_{sess}_static_lookup_rebuilt_equals_framework", np.max(np.abs(Ps - Ps2)) < 1e-9, "static table from raw recording")
    t = rec["t"]
    A[f"jmact{sess}_t"], A[f"jmact{sess}_tq"], A[f"jmact{sess}_pos"] = t, rec["tq"], rec["pos"]
    A[f"jmact{sess}_P_legacy"], A[f"jmact{sess}_P_static"] = Pl, Ps
    fa.setdefault("active_p7", {})[sess] = dict(
        trial_index=j["active_idx"], file_trial=j["active_idx"] + 1, comment=j["active_comment"], angle=float(rec["angle"]),
        legacy_P_mean_windowends=float(Pl[19:].mean()), static_P_mean_windowends=float(Ps[19:].mean()),
        shift_mean_windowends=float((Pl - Ps)[19:].mean()), legacy_P_range=[float(Pl.min()), float(Pl.max())],
        static_P_range=[float(Ps.min()), float(Ps.max())], measured_tq_mean_windowends=float(rec["tq"][19:].mean()),
        measured_tq_sd_windowends=float(rec["tq"][19:].std()),
        target_legacy_mean=float((rec["tq"] - Pl)[19:].mean()), target_static_mean=float((rec["tq"] - Ps)[19:].mean()))
# interpolation geometry of the static value (no p7 entry in the static table)
sm1 = np.array(static_maps["JM"]["S1"])
fa["static_table_has_entry_near_m0.07"] = bool(np.any(np.abs(sm1[:, 0] + 0.07) < 0.05))
fa["static_value_is_interpolation_between"] = {
    sess: [list(map(float, e)) for e in static_maps["JM"][sess] if abs(e[0] + 0.217) < 0.01 or abs(e[0] - 0.058) < 0.01]
    for sess in ("S1", "S2")}

# check against the saved X00 run: answer key (y_true_original) = measured - legacy S2 table, per window
def check_saved_key(s):
    z = np.load(DWT / f"EXPERIMENTS/results/X00_baseline/{s}/S1toS2/seed0.npz")
    yt, pos = z["y_true_original"].reshape(8, -1), z["position"].reshape(8, -1)
    recs2 = [x for x in bundles[s]["active"] if x["session"] == "S2"]
    worst = 0.0
    used = set()
    for k in range(8):
        rec = min(recs2, key=lambda x: abs(x["angle"] - pos[k].mean()))
        assert rec["idx"] not in used
        used.add(rec["idx"])
        key_ = rec["tq"] - PassiveModel("legacy", bundles[s]).predict(rec)
        worst = max(worst, float(np.max(np.abs(key_[19:] - yt[k]))))
    return worst
for s in ("JM", "YES"):
    w = check_saved_key(s)
    chk(f"saved_X00_answer_key_{s}", w < 1e-3, f"max |measured - legacy S2 table - y_true_original| over 8 trials x 8891 windows = {w:.2e} Nm")
fa["note"] = ("file_trial = trial_index + 1 (storyboard '#21' = trial_index 20 = 'passive at p7, test'). Static table has "
              "no entry near -0.07 (the p7 passive trial is rejected as a whole: SD 14.6 Nm), so the static rule gives the "
              "value interpolated between the neighbouring table entries; the first 20 s of that trial (a real hold) "
              "is not used by either rule because the device never steps in this trial, so it is not cut into pieces.")
ent_s1 = fa["table_entry_members"]["S1"][0]
ent_s2 = fa["table_entry_members"]["S2"][0]
fa["table_entry_notes"] = [
    "S1 table entry near -0.07 = running mean of %d accepted pieces: %s" % (
        len(ent_s1["members"]), "; ".join("file trial %d piece starting at sample %d" % (m["file_trial"], m["start_sample"])
                                           for m in ent_s1["members"])),
    "S2 table entry near -0.07 = running mean of %d accepted piece(s): %s" % (
        len(ent_s2["members"]), "; ".join("file trial %d piece starting at sample %d" % (m["file_trial"], m["start_sample"])
                                           for m in ent_s2["members"])),
    "the p7 passive trial of S2 (file trial %d) is REJECTED by the legacy |mean| <= 20 Nm test (mean %+.2f Nm, SD %.2f Nm)" % (
        fa["trials"]["S2"]["whole_trial_piece"]["file_trial"], fa["trials"]["S2"]["whole_trial_piece"]["mean_tq"],
        fa["trials"]["S2"]["whole_trial_piece"]["tq_sd_ddof1"]),
]
fa["methods"] = {
    "traces_100Hz": "dataset_builder.downsample_trial(factor=10): scipy.signal.decimate(ftype='fir', zero_phase=True); whole passive "
                    "trial, no 1 s trim; t = index/100 s from the first sample; 9010 samples for the 90.1 s trial",
    "blocks": "10-s blocks of the raw 1 kHz torque, from t=0 (9 blocks; the last 0.1 s is dropped); mean and SD with ddof=1",
    "hold_0_20s": "mean / SD (ddof=1) of the raw torque over the first 20 s",
}
facts["a_jm"] = fa

# ---------------------------------------------------------------------------------------------------
# (c) YES at -0.49 rad
# ---------------------------------------------------------------------------------------------------
yb = bundles["YES"]
fc = {}
segs_by_sess = {sess: truth.session_rest(yb, sess) for sess in ("S1", "S2")}
cache_truth = truth.subject_truth("YES")
for sess in ("S1", "S2"):
    y = yes[sess]
    rec = next(x for x in yb["active"] if x["idx"] == y["trial_index"])
    ok = np.max(np.abs(rec["pos"] - y["pos"])) < 1e-9 and np.max(np.abs(rec["tq"] - y["tq"])) < 1e-9
    chk(f"yes_{sess}_frozen_downsample_equals_bundle", ok, "frozen trim+decimate == bundle")
    Pb = PassiveModel("legacy", yb).predict(rec)
    chk(f"yes_{sess}_worksheet_lookup_frozen_equals_framework", np.max(np.abs(Pb - y["P"])) < 1e-9, "worksheet table lookup")
    segs = segs_by_sess[sess][rec["idx"]]
    B = truth.baseline(rec, segs)
    cB = cache_truth[rec["idx"]]["B"]
    chk(f"yes_{sess}_relaxed_baseline_equals_cached_truth", np.max(np.abs(B - cB)) < 1e-3,
        f"max abs diff {np.max(np.abs(B - cB)):.2e} Nm (cache is float32)")
    mask = truth.rest_mask(len(rec["tq"]), segs)
    tq = rec["tq"]
    seg_tab = [dict(start_s=float(rec["t"][s_]), end_s=float(rec["t"][e_ - 1]), n=int(e_ - s_), median_tq=float(np.median(tq[s_:e_])),
                    sd_tq=float(tq[s_:e_].std())) for s_, e_ in segs]
    hold = rec["hold"]
    hseries = next(h for h in yb["holds"] if h["idx"] == hold["idx"])["series"]
    meds = np.array([x["median_tq"] for x in seg_tab])
    fc[sess] = dict(
        trial_index=rec["idx"], file_trial=rec["idx"] + 1, comment=y["comment"], angle=float(rec["angle"]),
        worksheet_value_at_m0487=float(y["table_at_m0487"]), table_mean_windowends=float(Pb[19:].mean()),
        table_range=[float(Pb.min()), float(Pb.max())],
        recorded_hold=dict(file_trial=hold["idx"] + 1, angle=hold["angle"], mean=hold["mean"], start2s=hold["start"], last5s=hold["end"],
                           slope_Nm_per_s=hold["slope"], dur_s=hold["dur"]),
        measured_tq_windowends=dict(mean=float(tq[19:].mean()), sd=float(tq[19:].std()), p5=float(np.percentile(tq[19:], 5)),
                                    median=float(np.median(tq[19:])), p95=float(np.percentile(tq[19:], 95)),
                                    min=float(tq[19:].min()), max=float(tq[19:].max())),
        active_target_windowends=dict(mean=float((tq - Pb)[19:].mean()), sd=float((tq - Pb)[19:].std())),
        rest_n_segments=len(segs), rest_fraction=float(mask.mean()), rest_segments=seg_tab,
        relaxed_level=dict(rest_median=float(np.median(tq[mask])) if mask.any() else None,
                           rest_mean=float(tq[mask].mean()) if mask.any() else None,
                           rest_seg_medians_min_max=[float(meds.min()), float(meds.max())] if len(meds) else None,
                           B_mean_windowends=float(B[19:].mean()), B_first=float(B[0]), B_last=float(B[-1]),
                           B_min_max=[float(B.min()), float(B.max())]),
        table_minus_relaxed=dict(vs_rest_median=float(Pb[19:].mean() - np.median(tq[mask])) if mask.any() else None,
                                 vs_B_mean=float(Pb[19:].mean() - B[19:].mean())),
        table_minus_hold_mean=float(Pb[19:].mean() - hold["mean"]))
    A[f"yes{sess}_t"], A[f"yes{sess}_tq"], A[f"yes{sess}_pos"] = rec["t"], tq, rec["pos"]
    A[f"yes{sess}_P_table"], A[f"yes{sess}_B_relaxed"], A[f"yes{sess}_rest_mask"] = Pb, B, mask
    A[f"yes{sess}_hold_tq"] = np.asarray(hseries, np.float64)
    A[f"yes{sess}_hold_t"] = np.arange(len(hseries)) / 100.0
fc["session_change_of_table"] = dict(S1=fc["S1"]["worksheet_value_at_m0487"], S2=fc["S2"]["worksheet_value_at_m0487"],
                                     change=fc["S2"]["worksheet_value_at_m0487"] - fc["S1"]["worksheet_value_at_m0487"])
fc["worksheet_vs_recorded_hold_max_abs_diff_all_positions"] = float(max(
    abs(e[1] - next(h["mean"] for h in yb["holds"] if h["session"] == sess and abs(h["angle"] - e[0]) < 0.02))
    for sess in ("S1", "S2") for e in yb["maps"]["worksheet"][sess]
    if any(h["session"] == sess and abs(h["angle"] - e[0]) < 0.02 for h in yb["holds"])))
fc["note"] = ("relaxed level = median torque over EMG-detected rest stretches inside the active trial (truth.session_rest, "
              "EMG only); B(t) = smooth baseline through them (truth.baseline). The worksheet value is what the baseline "
              "subtracts; the recorded hold is the separate trial that preceded the active trial at the same angle.")
facts["c_yes_m049"] = fc

# audit item 2: table vs relaxed level at the toes-down positions (20 trials = 5 people with a table x 2 sessions x 2 cells)
au = []
for s in ("CH", "EG", "HM", "JM", "YES"):
    tr_ = truth.subject_truth(s)
    for sess in ("S1", "S2"):
        recs_s = sorted([x for x in bundles[s]["active"] if x["session"] == sess], key=lambda x: x["angle"])[:2]
        for rec in recs_s:
            m = tr_[rec["idx"]]["rest"]
            for kind in ("legacy", "static"):
                P = PassiveModel(kind, bundles[s]).predict(rec)
                au.append(dict(subject=s, session=sess, angle=float(rec["angle"]), table=kind,
                               table_mean=float(P[19:].mean()), relaxed_B_mean=float(tr_[rec["idx"]]["B"][19:].mean()),
                               table_minus_relaxed=float(P[19:].mean() - tr_[rec["idx"]]["B"][19:].mean()),
                               n_rest_segments=int(tr_[rec["idx"]]["n_rest_segs"])))
for kind in ("legacy", "static"):
    d_ = [x["table_minus_relaxed"] for x in au if x["table"] == kind]
    facts.setdefault("audit_item2_check", {})[kind] = dict(n=len(d_), n_positive=int(sum(v > 0 for v in d_)), min=min(d_), max=max(d_),
                                                          median=float(np.median(d_)))
facts["audit_item2_check"]["rows"] = au
facts["audit_item2_check"]["note"] = ("table mean minus mean of the in-trial EMG-rest baseline B (truth cache), toes-down cells "
                                      "(2 most negative angles) of CH, EG, HM, JM, YES in both sessions = 20 trials; audit "
                                      "00_audit.md section 6 item 2 says 'too high in 20 of 20, by 0.6-2 Nm'.")

# ---------------------------------------------------------------------------------------------------
# (d) loss share per position (X00f learning curves, seed 0, kept epoch)
# ---------------------------------------------------------------------------------------------------
fd = {}
for s in SUBJ:
    b = bundles[s]
    pm = PassiveModel("legacy", b)                      # X00f config: passive='legacy'
    recs = b["active"]
    feats = {x["idx"]: np.zeros((len(x["t"]), 4)) for x in recs}
    targets = {x["idx"]: x["tq"] - pm.predict(x) for x in recs}
    d = splits.build("S1toS2", recs, feats, targets, window=RC.WINDOW)
    y, trl, ang = d["y_train"].astype(np.float64), d["meta_train"]["trial"], d["meta_train"]["angle"]
    j = json.load(open(DWT / f"EXPERIMENTS/results/learning_curves/{LC_EXP}/{s}/S1toS2/seed0.json"))
    fit = sorted(j["final_per_trial"]["fit"], key=lambda f: f["angle"])
    angle = np.array([f["angle"] for f in fit])
    n = np.array([f["n"] for f in fit], float)
    mse = np.array([f["mse"] for f in fit])
    sq = mse * n                                                         # summed squared error of the fit set at the kept epoch
    ssd = np.empty(len(fit)); ss0 = np.empty(len(fit)); mean_y = np.empty(len(fit))
    for k, f in enumerate(fit):
        m = trl == f["trial"]
        assert m.sum() == f["n"] and abs(ang[m][0] - f["angle"]) < 1e-9
        yy = y[m]
        ssd[k], ss0[k], mean_y[k] = np.sum((yy - yy.mean()) ** 2), np.sum(yy ** 2), yy.mean()
    gm = np.sum(n * mean_y) / n.sum()
    ssg = ssd + n * (mean_y - gm) ** 2                                   # about the global fit-set mean
    sh = dict(sqerr=sq / sq.sum(), ssd=ssd / ssd.sum(), ss0=ss0 / ss0.sum(), ssglobal=ssg / ssg.sum())
    best = j["best_epoch"]
    hist_mse_fit = j["history"][best - 1]["mse_fit"]
    chk(f"loss_{s}_mean_mse_equals_history_mse_fit", abs(sq.sum() / n.sum() - hist_mse_fit) < 1e-6 * max(1, hist_mse_fit),
        f"mean per-trial MSE {sq.sum() / n.sum():.6f} vs history mse_fit at epoch {best} {hist_mse_fit:.6f}")
    fd[s] = dict(
        best_epoch=best, epochs=j["epochs"], n_fit_windows=int(n.sum()), angles=angle, mse_per_position=mse,
        share_squared_error=sh["sqerr"], share_target_variation_within_position=sh["ssd"],
        share_sum_squares_about_zero=sh["ss0"], share_variation_about_global_mean=sh["ssglobal"],
        equal_share=1.0 / len(fit),
        toes_down_two=dict(angles=angle[:2], squared_error=float(sh["sqerr"][:2].sum()),
                           target_variation_within_position=float(sh["ssd"][:2].sum()),
                           sum_squares_about_zero=float(sh["ss0"][:2].sum()),
                           variation_about_global_mean=float(sh["ssglobal"][:2].sum()),
                           most_toes_down_only=dict(squared_error=float(sh["sqerr"][0]), target_variation=float(sh["ssd"][0]))),
        target_sd_per_position=np.sqrt(ssd / n))
    for nm, v in (("angle", angle), ("mse", mse), ("n", n), ("share_sqerr", sh["sqerr"]), ("share_ssd", sh["ssd"]),
                  ("share_ss0", sh["ss0"]), ("share_ssglobal", sh["ssglobal"]), ("target_sd", np.sqrt(ssd / n))):
        A[f"loss_{s}_{nm}"] = np.asarray(v, float)
fd["definitions"] = {
    "share_squared_error": "mse x n of each position's fit windows / total, from final_per_trial['fit'] at the kept (restored) "
                           "epoch of the seed-0 X00f run = the share of the fit-set squared error (what the optimiser minimised)",
    "share_target_variation_within_position": "sum of squared deviations of the fit targets from each POSITION's own mean / total; "
                                              "= what is left to explain once the per-position level is known (the quantity R2 is built on)",
    "share_sum_squares_about_zero": "sum of y^2 per position / total: the loss of a network that predicted 0 everywhere",
    "share_variation_about_global_mean": "deviations from the fit set's overall mean (within + between positions)",
    "targets": "fit targets = measured - legacy session-1 table (X00f passive='legacy'), first 80 % of each session-1 trial, rebuilt "
               "with splits.build; trial list, n and angle checked against the json",
    "toes_down_two": "the two most negative angles of the person"}
facts["d_loss_share"] = fd

# ---------------------------------------------------------------------------------------------------
# storyboard comparison
# ---------------------------------------------------------------------------------------------------
S1j, S2j = fa["trials"]["S1"], fa["trials"]["S2"]
act1, act2 = fa["active_p7"]["S1"], fa["active_p7"]["S2"]
sbc = []


def sb(name, claim, got, ok):
    sbc.append(dict(item=name, storyboard=claim, computed=got, matches=bool(ok)))


pt1 = S1j["whole_trial_piece"]
sb("JM p7 passive trial S1: torque SD / mean", "14.6 / +15.6", f"{pt1['tq_sd_ddof1']:.2f} / {pt1['mean_tq']:+.2f}",
   round(pt1["tq_sd_ddof1"], 1) == 14.6 and round(pt1["mean_tq"], 1) == 15.6)
ent = [c for c in fa["table_entry_members"]["S1"]][0]
ent2 = [c for c in fa["table_entry_members"]["S2"]][0]
sb("JM legacy table entry near -0.07, S1 / S2", "+11.72 / +10.75", f"{ent['tq']:+.2f} / {ent2['tq']:+.2f}",
   round(ent["tq"], 2) == 11.72 and round(ent2["tq"], 2) == 10.75)
sb("legacy value subtracted at JM p7 active trial S1 / S2", "+11.26 / +9.78",
   f"{act1['legacy_P_mean_windowends']:+.2f} / {act2['legacy_P_mean_windowends']:+.2f}",
   round(act1["legacy_P_mean_windowends"], 2) == 11.26 and round(act2["legacy_P_mean_windowends"], 2) == 9.78)
sb("static value subtracted S1 / S2", "-1.97 / -1.90",
   f"{act1['static_P_mean_windowends']:+.2f} / {act2['static_P_mean_windowends']:+.2f}",
   round(act1["static_P_mean_windowends"], 2) == -1.97 and round(act2["static_P_mean_windowends"], 2) == -1.90)
sb("shift S1 / S2", "13.22 / 11.68", f"{act1['shift_mean_windowends']:.2f} / {act2['shift_mean_windowends']:.2f}",
   round(act1["shift_mean_windowends"], 2) == 13.22 and round(act2["shift_mean_windowends"], 2) == 11.68)
sb("static clean-hold torque SD range (static-map pieces)", "0.02-0.33 Nm",
   f"{fb['static_map_pieces']['sd_min']:.3f}-{fb['static_map_pieces']['sd_max']:.3f} Nm (n={fb['static_map_pieces']['n']})",
   round(fb["static_map_pieces"]["sd_min"], 2) == 0.02 and round(fb["static_map_pieces"]["sd_max"], 2) == 0.33)
sb("static clean-hold torque SD range (framework holds incl. YES)", "0.02-0.33 Nm",
   f"{fb['framework_holds']['sd_min']:.3f}-{fb['framework_holds']['sd_max']:.3f} Nm (n={fb['framework_holds']['n']})",
   round(fb["framework_holds"]["sd_min"], 2) == 0.02 and round(fb["framework_holds"]["sd_max"], 2) == 0.33)
sb("JM trial first 20 s hold level", "-1.8 Nm", f"{S1j['hold0_20s_mean']:.3f} Nm (S2 {S2j['hold0_20s_mean']:.3f})", round(S1j["hold0_20s_mean"], 1) == -1.8)
bm = [v for s_ in ("S1", "S2") for v in jm[s_]["block_mean"][2:]]
sb("JM trial #21 (S1) 10-s block means after 20 s", "12-31 Nm (S1 trial)",
   f"S1 {min(jm['S1']['block_mean'][2:]):.1f}-{max(jm['S1']['block_mean'][2:]):.1f}; S2 {min(jm['S2']['block_mean'][2:]):.1f}-{max(jm['S2']['block_mean'][2:]):.1f}; union {min(bm):.1f}-{max(bm):.1f}",
   round(min(jm['S1']['block_mean'][2:])) == 12 and round(max(jm['S1']['block_mean'][2:])) == 31)
sb("JM p7 passive trial S2 (#37) accepted by legacy rule", "implied: 'this trial passed' (both sessions wrong in nearly the same way)",
   f"S2 trial mean {S2j['whole_trial_piece']['mean_tq']:+.2f} Nm > 20: rejected; S2 entry {ent2['tq']:+.2f} comes from file trial "
   f"{fa['table_entry_members']['S2'][0]['members'][0]['file_trial']} only; S1 entry is the mean of file trials "
   f"{[m['file_trial'] for m in fa['table_entry_members']['S1'][0]['members']]}", False)
chk("yes_worksheet_vs_recorded_holds_within_0.15", fc["worksheet_vs_recorded_hold_max_abs_diff_all_positions"] <= 0.15,
    f"max |worksheet - recorded hold mean| over 16 angle/session pairs = {fc['worksheet_vs_recorded_hold_max_abs_diff_all_positions']:.3f} Nm")
sb("YES worksheet passive at -0.487 S1 / S2", "7.9 / 10.6", f"{fc['S1']['table_mean_windowends']:.2f} / {fc['S2']['table_mean_windowends']:.2f} (table entries {fc['S1']['worksheet_value_at_m0487']} / {fc['S2']['worksheet_value_at_m0487']})",
   round(fc["S1"]["worksheet_value_at_m0487"], 1) == 7.9 and round(fc["S2"]["worksheet_value_at_m0487"], 1) == 10.6)
sb("YES active torque spread at -0.49", "0.4 Nm", f"{fc['S1']['measured_tq_windowends']['sd']:.2f} (S1) / {fc['S2']['measured_tq_windowends']['sd']:.2f} (S2)",
   round(fc["S2"]["measured_tq_windowends"]["sd"], 1) == 0.4)
yl = fd["YES"]["toes_down_two"]
sb("YES two most toes-down positions' share of the loss", "about 1 %",
   f"squared error at kept epoch {100 * yl['squared_error']:.2f} %; target variation {100 * yl['target_variation_within_position']:.2f} %; "
   f"sum of squares about 0 {100 * yl['sum_squares_about_zero']:.2f} %  (only the target-variation definition gives about 1 %)",
   round(100 * yl["squared_error"]) == 1)
ai = facts["audit_item2_check"]["legacy"]
sb("table too high vs relaxed level at toes-down", "20 of 20, by 0.6-2 Nm",
   f"legacy table: {ai['n_positive']} of {ai['n']} positive, {ai['min']:.2f} to {ai['max']:.2f} Nm", ai["n_positive"] == 20 and ai["min"] >= 0.55 and ai["max"] <= 2.05)
facts["storyboard_comparison"] = sbc

# =====================================================================================================
# display block, checks, save
# =====================================================================================================
facts["display"] = {
    "jm_trial21_S1": dict(sd=r(pt1["tq_sd_ddof1"], 1), mean=r(pt1["mean_tq"], 1), hold_0_20s=r(S1j["hold0_20s_mean"], 2),
                          block_means_after_20s=[r(v, 1) for v in jm["S1"]["block_mean"][2:]]),
    "jm_legacy_entry_S1_S2": [r(ent["tq"], 2), r(ent2["tq"], 2)],
    "jm_p7_legacy_vs_static": dict(S1=[r(act1["legacy_P_mean_windowends"], 2), r(act1["static_P_mean_windowends"], 2), r(act1["shift_mean_windowends"], 2)],
                                   S2=[r(act2["legacy_P_mean_windowends"], 2), r(act2["static_P_mean_windowends"], 2), r(act2["shift_mean_windowends"], 2)]),
    "static_hold_sd_range": [r(fb["static_map_pieces"]["sd_min"], 3), r(fb["static_map_pieces"]["sd_max"], 3)],
    "yes_m049": dict(table=[r(fc["S1"]["worksheet_value_at_m0487"], 1), r(fc["S2"]["worksheet_value_at_m0487"], 1)],
                     relaxed_rest_median=[r(fc[s_]["relaxed_level"]["rest_median"], 2) for s_ in ("S1", "S2")],
                     relaxed_B_mean=[r(fc[s_]["relaxed_level"]["B_mean_windowends"], 2) for s_ in ("S1", "S2")],
                     measured_mean=[r(fc[s_]["measured_tq_windowends"]["mean"], 2) for s_ in ("S1", "S2")],
                     table_minus_relaxed=[r(fc[s_]["table_minus_relaxed"]["vs_rest_median"], 2) for s_ in ("S1", "S2")]),
    "loss_share_toes_down_two_pct": {s_: dict(squared_error=r(100 * fd[s_]["toes_down_two"]["squared_error"], 1),
                                              target_variation=r(100 * fd[s_]["toes_down_two"]["target_variation_within_position"], 1),
                                              sum_squares_about_zero=r(100 * fd[s_]["toes_down_two"]["sum_squares_about_zero"], 1))
                                     for s_ in SUBJ}}
facts["checks"] = checks
facts["npz_arrays"] = {k: list(v.shape) for k, v in sorted(A.items())}
facts["inputs"] = dict(data_dir=str(DATA_DIR), frozen_code=str(FROZEN), framework=str(DWT))


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


np.savez_compressed(OUT / "y05.npz", **A)
with open(OUT / "facts_y05.json", "w", encoding="utf-8") as f:
    json.dump({"y05": _plain(facts)}, f, indent=1, sort_keys=True)
bad = [c for c in checks if not c["ok"]]
print(f"\n{len(checks)} checks, {len(bad)} failed")
for c in bad:
    print("  FAILED:", c["name"], c["detail"])
for x in sbc:
    print(("  match   " if x["matches"] else "  DIFFERS ") + x["item"] + " | storyboard " + x["storyboard"] + " | computed " + x["computed"])
