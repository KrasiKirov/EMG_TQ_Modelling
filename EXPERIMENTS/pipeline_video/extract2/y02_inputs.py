"""y02: the BASELINE's inputs for the running example EG (storyboard chapters c03-c07).

Builds every subject's dataset with the FROZEN last-semester code
(DWT/EXPERIMENTS/baseline/original_code/ML: read_flb, classify_trials, build_dataset) exactly as
run_original_baseline.py / training.load_subject call it, then re-derives the per-trial series the video draws
(100 Hz envelopes, angle, torque, passive value, target, windows) from the same data frames and checks them
bit for bit against the arrays build_dataset returned, and the session-2 targets against the saved
X00_baseline seed0.npz y_true_original (all six people).

Reads  : EMG_DATA_DIR/*.flb                                  (recordings, main checkout ML/subjects)
         DWT/EXPERIMENTS/baseline/original_code/ML            (frozen code; imported, not modified)
         DWT/EXPERIMENTS/results/X00_baseline/<S>/S1toS2/seed0.{npz,json}   (checks only)
Writes : EXPERIMENTS/pipeline_video/data2/y02.npz
         EXPERIMENTS/pipeline_video/data2/facts_y02.json      (top-level key "y02")
Run    : OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv311/Scripts/python.exe extract2/y02_inputs.py
         (numpy / scipy / pandas only; no TensorFlow, no training)

Conventions
  * "file trial index" = df.attrs['trial_index'], the 0-based position in the .flb file (EG: 21 = session 1 at
    -0.038 rad, 37 = session 2 at the same angle). Printed trial numbers in the original logs are index + 1.
  * 100 Hz sample k of a trial (after the 1 s trim) sits at trial time t = 1.001 + 0.01 k seconds (the .flb time axis starts at
    0.001 s; t is seconds since the start of that trial's recording).
  * negative angle = plantarflexion = toes down; EG's pushes are negative torque.
  * window j of a trial = samples j .. j+19, target = active torque at sample j+19.
"""
import contextlib
import sys
sys.dont_write_bytecode = True                                  # never write .pyc into the frozen-code folder
import io
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
VWT = HERE.parents[2]                                           # video worktree
DWT = pathlib.Path("C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137")
ORIG = DWT / "EXPERIMENTS/baseline/original_code"
X00 = DWT / "EXPERIMENTS/results/X00_baseline"
DATA_DIR = pathlib.Path(os.environ.get("EMG_DATA_DIR", "C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/ML/subjects"))
os.environ["EMG_DATA_DIR"] = str(DATA_DIR)
OUT = VWT / "EXPERIMENTS/pipeline_video/data2"
OUT.mkdir(parents=True, exist_ok=True)

# ---- import the FROZEN code (nothing from any other 'ML' package may shadow it)
for k in [k for k in sys.modules if k == "ML" or k.startswith("ML.")]:
    del sys.modules[k]
sys.path.insert(0, str(ORIG))
import ML.config as cfg                                         # noqa: E402
from ML.preprocessing import dataset_builder as db              # noqa: E402
from ML.preprocessing.flb_reader import read_flb                # noqa: E402

assert pathlib.Path(cfg.__file__).resolve().is_relative_to(ORIG.resolve()), cfg.__file__

SUBJ = ["CH", "EG", "HM", "IES", "JM", "YES"]
CH = ["gm", "gl", "sol", "ta"]                                  # the model's feature order, then angle
W = cfg.N_STEPS
TRIM = db.TRANSIENT_SAMPLES                                     # 1000 raw samples = 1 s
EG_S1, EG_S2 = 21, 37
T0 = 1.001                                                      # time of 100 Hz sample 0 (flb time axis starts at 0.001 s, 1 s trimmed)

facts = {"checks": [], "display": {}}
arrays = {}


def check(name, ok, detail):
    facts["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(("OK   " if ok else "FAIL ") + name + ": " + str(detail))
    assert ok, name


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ----------------------------------------------------------------------------------------------- build a subject
def split_sets(trials, test_idx, retest_idx):
    """The session split exactly as build_dataset step 2 decides it (trial_index sets, 0-based)."""
    if test_idx is not None and retest_idx is not None:
        return {i - 1 for i in test_idx}, {i - 1 for i in retest_idx}
    mvc, passive, active = db.classify_trials(trials)
    test_set, retest_set, unl = set(), set(), set()
    for df in active:
        if df[db.TORQUE_COL].std() > db._ACTIVE_STD_THRESHOLD:
            continue
        s = df.attrs.get("session")
        (retest_set if s == "retest" else test_set if s == "test" else unl).add(df.attrs["trial_index"])
    if not test_set and retest_set and unl:
        test_set = unl
    return test_set, retest_set


def build_subject(s):
    fname, sid = cfg.SUBJECTS[s]
    ov = cfg.TRIAL_OVERRIDES.get(s, {})
    test_idx, retest_idx = ov.get("test"), ov.get("retest")
    pov = cfg.PASSIVE_TORQUE_OVERRIDES.get(s)
    trials = read_flb(str(DATA_DIR / fname), subject_id=sid)
    out = quiet(db.build_dataset, trials, test_trial_indices=test_idx, retest_trial_indices=retest_idx,
                subtract_passive=True, normalize_position=False, normalize_mvc=False, passive_overrides=pov)
    Xtr, ytr, Xv, yv, Xte, yte, emg_max, pass_s1, op_pos, _, _ = out
    test_set, retest_set = split_sets(trials, test_idx, retest_idx)

    # session-2 passive map, as build_dataset step 4 / run_original_baseline.py build it
    _, passive, _ = db.classify_trials(trials)
    if test_idx is not None or retest_idx is not None:
        forced = {i - 1 for i in (test_idx or [])} | {i - 1 for i in (retest_idx or [])}
        passive = [d for d in passive if d.attrs.get("trial_index") not in forced]
    if pov and "retest" in pov:
        pass_s2 = sorted(pov["retest"], key=lambda e: e[0])
    else:
        rp = [d for d in passive if d.attrs.get("session") == "retest"]
        pass_s2 = db.get_passive_torque_map(rp) if rp else pass_s1

    # per-trial series from the (in place) processed frames, in file order like build_dataset's active_trials
    # order = build_dataset's: active trials in file order, then the 'rescued' mislabelled-passive ones (YES) appended
    mvc_t, passive_t, active_t = db.classify_trials(trials)
    if test_idx is not None or retest_idx is not None:
        forced = {i - 1 for i in (test_idx or [])} | {i - 1 for i in (retest_idx or [])}
        active_t = active_t + [d for d in passive_t if d.attrs.get("trial_index") in forced]
    act = [d for d in active_t if d[db.TORQUE_COL].std() <= db._ACTIVE_STD_THRESHOLD
           and d.attrs["trial_index"] in (test_set | retest_set)]
    assert all(f"{CH[0]}_env" in d.columns for d in act)
    per = []
    for d in act:
        i = d.attrs["trial_index"]
        sess = "S1" if i in test_set else "S2"
        d2 = d.iloc[TRIM:].reset_index(drop=True)
        ds = db.downsample_trial(d2, factor=db.DOWNSAMPLE_FACTOR)
        pm = pass_s1 if sess == "S1" else pass_s2
        X, y, _ = db.build_windows(ds, CH, pm, mvc_entries=None, window=W, stride=db.STRIDE_STEPS)
        pos = ds[db.POSITION_COL].values
        passive_v = np.array([db.lookup_passive_torque(pm, p) for p in pos], dtype=np.float32)
        per.append(dict(idx=i, sess=sess, ds=ds, X=X, y=y, passive=passive_v, full=d,
                        raw_env=np.column_stack([ds[f"{c}_env"].values for c in CH]),
                        env=np.column_stack([ds[f"{c}_env_norm"].values for c in CH]),
                        angle=pos, torque=ds[db.TORQUE_COL].values))
    return dict(trials=trials, Xtr=Xtr, ytr=ytr, Xv=Xv, yv=yv, Xte=Xte, yte=yte, emg_max=emg_max, pass_s1=pass_s1,
                pass_s2=pass_s2, op_pos=op_pos, per=per, test_set=test_set, retest_set=retest_set)


B = {}
for s in SUBJ:
    B[s] = build_subject(s)
    print("built", s, "S1 trials", sorted(B[s]["test_set"]), "S2 trials", sorted(B[s]["retest_set"]))

# ----------------------------------------------------------------------------------------------- checks vs build_dataset
for s in SUBJ:
    b = B[s]
    s1 = [p for p in b["per"] if p["sess"] == "S1"]
    s2 = [p for p in b["per"] if p["sess"] == "S2"]
    n_split = lambda n: (int(n * db.TRAIN_FRAC), int(n * (1.0 - db.TRAIN_FRAC)))   # build_dataset passes val_frac = 1 - train_frac
    tr = [(p["X"][:n_split(len(p["X"]))[0]], p["y"][:n_split(len(p["X"]))[0]]) for p in s1]
    va = [(p["X"][n_split(len(p["X"]))[0]:n_split(len(p["X"]))[0] + n_split(len(p["X"]))[1]],
           p["y"][n_split(len(p["X"]))[0]:n_split(len(p["X"]))[0] + n_split(len(p["X"]))[1]]) for p in s1]
    same = (np.array_equal(np.concatenate([a for a, _ in tr]), b["Xtr"]) and np.array_equal(np.concatenate([a for _, a in tr]), b["ytr"])
            and np.array_equal(np.concatenate([a for a, _ in va]), b["Xv"]) and np.array_equal(np.concatenate([a for _, a in va]), b["yv"])
            and np.array_equal(np.concatenate([p["X"] for p in s2]), b["Xte"]) and np.array_equal(np.concatenate([p["y"] for p in s2]), b["yte"]))
    check(f"rebuilt per-trial windows == build_dataset arrays ({s})", same,
          f"X_train {b['Xtr'].shape} X_val {b['Xv'].shape} X_test {b['Xte'].shape}")

    z = np.load(X00 / s / "S1toS2" / "seed0.npz")
    yt = z["y_true_original"].astype(np.float64)
    d = float(np.max(np.abs(b["yte"].astype(np.float64) - yt)))
    pdiff = float(np.max(np.abs(b["Xte"][:, -1, 4].astype(np.float64) - z["position"].astype(np.float64))))
    check(f"session-2 targets == X00_baseline seed0 y_true_original ({s})", d < 1e-5 and len(yt) == len(b["yte"]),
          f"n={len(yt)} max|diff|={d:.3g} Nm; angle feature max|diff|={pdiff:.3g} rad")
    js = json.load(open(X00 / s / "S1toS2" / "seed0.json"))
    ref = js["legacy_passive_map_retest"]
    check(f"session-2 passive table == X00_baseline seed0.json legacy_passive_map_retest ({s})",
          len(ref) == len(b["pass_s2"]) and all(abs(a[0] - c[0]) < 1e-9 and abs(a[1] - c[1]) < 1e-9 for a, c in zip(ref, b["pass_s2"])),
          f"{len(ref)} entries" if ref else "empty table (no passive trials)")
    check(f"window counts == X00_baseline seed0.json n_windows ({s})",
          [len(b["ytr"]), len(b["yv"]), len(b["yte"])] == [js["n_windows"][k] for k in ("train", "val", "test")],
          [len(b["ytr"]), len(b["yv"]), len(b["yte"])])

# ----------------------------------------------------------------------------------------------- counts
counts = {}
for s in SUBJ:
    b = B[s]
    per_tr = {}
    for p in b["per"]:
        n_s = len(p["ds"])
        n_w = len(p["X"])
        n_fit = int(n_w * db.TRAIN_FRAC)
        n_val = int(n_w * (1.0 - db.TRAIN_FRAC))
        per_tr[p["idx"]] = dict(session=p["sess"], n_samples_100hz=n_s, n_windows=n_w,
                                n_fit=n_fit if p["sess"] == "S1" else None, n_val=n_val if p["sess"] == "S1" else None,
                                n_dropped=(n_w - n_fit - n_val) if p["sess"] == "S1" else None,
                                angle_mean=float(np.mean(p["angle"])))
    counts[s] = dict(per_trial=per_tr, s1_order=[p["idx"] for p in b["per"] if p["sess"] == "S1"],
                     s2_order=[p["idx"] for p in b["per"] if p["sess"] == "S2"], n_s1_trials=len(b["test_set"]), n_s2_trials=len(b["retest_set"]),
                     n_fit=len(b["ytr"]), n_val=len(b["yv"]), n_test=len(b["yte"]))
facts["counts"] = counts
eg = counts["EG"]
facts["eg_counts"] = dict(n_samples_per_trial=sorted({v["n_samples_100hz"] for v in eg["per_trial"].values()}),
                          windows_per_trial=sorted({v["n_windows"] for v in eg["per_trial"].values()}),
                          fit_per_trial=sorted({v["n_fit"] for v in eg["per_trial"].values() if v["session"] == "S1"}),
                          val_per_trial=sorted({v["n_val"] for v in eg["per_trial"].values() if v["session"] == "S1"}),
                          dropped_per_s1_trial=sorted({v["n_dropped"] for v in eg["per_trial"].values() if v["session"] == "S1"}),
                          n_fit=eg["n_fit"], n_val=eg["n_val"], n_test=eg["n_test"],
                          fit_plus_val_plus_test=eg["n_fit"] + eg["n_val"] + eg["n_test"])
check("EG counts: 8891 windows, 7112 fit, 1778 val per trial; 56896 / 14224 / 71128 in total",
      facts["eg_counts"]["windows_per_trial"] == [8891] and facts["eg_counts"]["fit_per_trial"] == [7112]
      and facts["eg_counts"]["val_per_trial"] == [1778] and (eg["n_fit"], eg["n_val"], eg["n_test"]) == (56896, 14224, 71128),
      facts["eg_counts"])
facts["all_subjects_same_counts"] = all((c["n_fit"], c["n_val"], c["n_test"]) == (56896, 14224, 71128) for c in counts.values())
facts["n_dropped_window_per_s1_trial_note"] = ("int(8891*0.8)=7112 and int(8891*0.2)=1778 leave 1 window (the last of the trial) "
                                              "unused: _temporal_split discards the remainder")

# boundary of fit and validation
n_fit_t, n_val_t = 7112, 1778
facts["boundary"] = dict(
    last_fit_window_index=n_fit_t - 1, first_val_window_index=n_fit_t,
    last_fit_window_samples=[n_fit_t - 1, n_fit_t - 1 + W - 1], first_val_window_samples=[n_fit_t, n_fit_t + W - 1],
    shared_samples=[n_fit_t, n_fit_t - 1 + W - 1], n_shared_samples=W - 1,
    last_val_window_index=n_fit_t + n_val_t - 1, dropped_window_index=n_fit_t + n_val_t,
    t_first_val_window_start=T0 + 0.01 * n_fit_t, t_last_fit_target=T0 + 0.01 * (n_fit_t - 1 + W - 1),
    t_first_val_target=T0 + 0.01 * (n_fit_t + W - 1))
# verify with the real data: first val window of EG trial 21 equals the 1-sample shift of the last fit window
p21 = next(p for p in B["EG"]["per"] if p["idx"] == EG_S1)
check("overlap: last fit window and first val window share 19 of 20 samples (EG trial 21)",
      np.array_equal(p21["X"][n_fit_t - 1][1:], p21["X"][n_fit_t][:-1]) and p21["X"].shape[0] == 8891,
      "X[7111][1:] == X[7112][:-1]; last fit window = samples 7111..7130, first val window = 7112..7131")

# ----------------------------------------------------------------------------------------------- EG trial 21 and 37
p37 = next(p for p in B["EG"]["per"] if p["idx"] == EG_S2)
check("EG trials: file index 21 is S1 and 37 is S2, both at -0.038 rad",
      p21["sess"] == "S1" and p37["sess"] == "S2" and abs(np.mean(p21["angle"]) + 0.038) < 5e-4 and abs(np.mean(p37["angle"]) + 0.038) < 5e-4,
      dict(angle_21=float(np.mean(p21["angle"])), angle_37=float(np.mean(p37["angle"])),
           comment_21=p21["full"].attrs["comment"], comment_37=p37["full"].attrs["comment"]))


def trial_arrays(p, prefix):
    n = len(p["ds"])
    t = T0 + 0.01 * np.arange(n)
    check(f"{prefix}: 100 Hz time axis t = T0 + 0.01 k equals the frame's own 'time' column", np.allclose(t, p["ds"]["time"].values, atol=1e-5),
          f"max|diff|={np.max(np.abs(t - p['ds']['time'].values)):.2g} s")
    tgt = (p["torque"].astype(np.float32) - p["passive"]).astype(np.float32)
    check(f"{prefix}: target = torque - passive equals the y build_windows produced", np.array_equal(tgt[W - 1:], p["y"]),
          f"n windows {len(p['y'])}")
    return {f"{prefix}_t": t, **{f"{prefix}_{c}": p["env"][:, i] for i, c in enumerate(CH)},
            f"{prefix}_angle": p["angle"], f"{prefix}_torque": p["torque"], f"{prefix}_passive": p["passive"].astype(np.float64),
            f"{prefix}_target": tgt}


arrays.update(trial_arrays(p21, "s1"))
arrays.update({f"s1_raw_{c}": p21["raw_env"][:, i] for i, c in enumerate(CH)})
arrays.update(trial_arrays(p37, "s2"))
arrays.update({f"s2_raw_{c}": p37["raw_env"][:, i] for i, c in enumerate(CH)})

emax_eg = B["EG"]["emg_max"]
check("normalised = raw envelope / session-1 max (EG trial 21)",
      all(np.allclose(arrays[f"s1_{c}"], arrays[f"s1_raw_{c}"] / emax_eg[c], rtol=1e-9, atol=1e-12) for c in CH), "ratio recomputed for 4 channels")

# passive tables as used
arrays["passive_s1"] = np.array(B["EG"]["pass_s1"], dtype=np.float64)
arrays["passive_s2"] = np.array(B["EG"]["pass_s2"], dtype=np.float64)
facts["passive_tables"] = {s: dict(S1=[[float(a), float(b)] for a, b in B[s]["pass_s1"]], S2=[[float(a), float(b)] for a, b in B[s]["pass_s2"]]) for s in SUBJ}
facts["passive_tables_note"] = ("IES has no passive trials: empty tables, nothing is subtracted. YES: typed-in worksheet values "
                                "(config.PASSIVE_TORQUE_OVERRIDES). Others: auto-detected from the passive holds. Straight lines between the points; "
                                "outside the measured range the end segments are extended linearly (dataset_builder._extrapolate).")

# passive value subtracted at the example angle
for tag, p in (("S1", p21), ("S2", p37)):
    v = p["passive"].astype(np.float64)
    ang = float(np.mean(p["angle"]))
    facts[f"eg_{tag}_passive_subtracted"] = dict(mean_Nm=float(np.mean(v)), median_Nm=float(np.median(v)), min_Nm=float(v.min()), max_Nm=float(v.max()),
                                                 sd_Nm=float(v.std()), at_mean_angle_Nm=float(db.lookup_passive_torque(B["EG"]["pass_s1" if tag == "S1" else "pass_s2"], ang)),
                                                 at_exactly_minus_0p038_Nm=float(db.lookup_passive_torque(B["EG"]["pass_s1" if tag == "S1" else "pass_s2"], -0.038)),
                                                 mean_angle_rad=ang)
check("EG passive subtracted at -0.038 rad: about -3.5 Nm (S1) and -2.6 Nm (S2)",
      abs(facts["eg_S1_passive_subtracted"]["median_Nm"] + 3.5) < 0.1 and abs(facts["eg_S2_passive_subtracted"]["median_Nm"] + 2.6) < 0.1,
      dict(S1=round(facts["eg_S1_passive_subtracted"]["median_Nm"], 3), S2=round(facts["eg_S2_passive_subtracted"]["median_Nm"], 3)))
facts["display"]["eg_passive_S1_Nm"] = round(facts["eg_S1_passive_subtracted"]["median_Nm"], 2)
facts["display"]["eg_passive_S2_Nm"] = round(facts["eg_S2_passive_subtracted"]["median_Nm"], 2)

# the torque / target of the example trial
for tag, p in (("S1", p21), ("S2", p37)):
    tg = (p["torque"].astype(np.float32) - p["passive"]).astype(np.float64)
    facts[f"eg_{tag}_trial"] = dict(idx=p["idx"], comment=p["full"].attrs["comment"], n_samples=len(p["ds"]), angle_mean_rad=float(np.mean(p["angle"])),
                                    angle_sd_mrad=float(np.std(p["angle"]) * 1000), torque_min_Nm=float(p["torque"].min()), torque_max_Nm=float(p["torque"].max()),
                                    target_min_Nm=float(tg.min()), target_max_Nm=float(tg.max()), target_p5_Nm=float(np.percentile(tg, 5)),
                                    target_p95_Nm=float(np.percentile(tg, 95)),
                                    raw_env_max=[float(p["raw_env"][:, i].max()) for i in range(4)],
                                    norm_env_max_in_trial=[float(p["env"][:, i].max()) for i in range(4)],
                                    raw_env_p95=[float(np.percentile(p["raw_env"][:, i], 95)) for i in range(4)])

# ----------------------------------------------------------------------------------------------- the 24 maxima
cat_names = {0: "trimmed_first_second", 1: "fit_only", 2: "fit_and_val_overlap", 3: "val_only", 4: "discarded_tail"}
maxima, codes, in_val_strict, in_val_loose = [], np.zeros((6, 4), int), np.zeros((6, 4), bool), np.zeros((6, 4), bool)
mx_val, mx_trial, mx_i, mx_k = np.zeros((6, 4)), np.zeros((6, 4), int), np.zeros((6, 4), int), np.zeros((6, 4), int)
mx_second = np.zeros((6, 4))
for si, s in enumerate(SUBJ):
    b = B[s]
    s1 = [p for p in b["per"] if p["sess"] == "S1"]
    for ci, c in enumerate(CH):
        best = None
        for p in s1:
            e = p["full"][f"{c}_env"].values
            i = int(np.argmax(e))
            if best is None or e[i] > best[0]:
                best = (float(e[i]), p["idx"], i, len(e))
        v, tr, i, n_raw = best
        check(f"session-1 maximum recomputed == emg_max returned by build_dataset ({s} {c})", v == b["emg_max"][c], f"{v:.6g}")
        # where does that raw sample fall in the 100 Hz trial? (trim 1000 samples, then keep every 10th)
        k = (i - TRIM) / 10.0
        n_s = len(s1[0]["ds"])
        nw = n_s - W + 1
        nfit, nval = int(nw * db.TRAIN_FRAC), int(nw * (1 - db.TRAIN_FRAC))
        if i < TRIM:
            code = 0
        elif k < nfit:
            code = 1
        elif k < nfit + W - 1:                                      # samples 7112..7130 sit in the last fit windows AND the first val windows
            code = 2
        elif k <= nfit + nval - 1 + W - 1:                          # val targets 7131..8908
            code = 3
        else:
            code = 4
        codes[si, ci] = code
        in_val_strict[si, ci] = code == 3
        in_val_loose[si, ci] = code in (2, 3)
        mx_val[si, ci], mx_trial[si, ci], mx_i[si, ci], mx_k[si, ci] = v, tr, i, int(np.floor(k)) if i >= TRIM else -1
        # largest value the same channel reaches in a DIFFERENT S1 trial, and in the fit part only (to show how much the max matters)
        others = [float(p["full"][f"{c}_env"].values.max()) for p in s1 if p["idx"] != tr]
        mx_second[si, ci] = max(others)
        allv = np.concatenate([p["full"][f"{c}_env"].values for p in s1])
        maxima.append(dict(subject=s, channel=c, max=v, trial_idx=tr, raw_sample=i, t_s=(i + 1) / 1000.0, sample_100hz=(None if i < TRIM else k),
                           trial_len_s=n_raw / 1000.0, frac_of_trial=i / n_raw, where=cat_names[code],
                           in_validation_part=bool(code == 3), touches_validation_windows=bool(code in (2, 3)),
                           next_best_other_trial=float(mx_second[si, ci]), ratio_max_over_next_best_trial=float(v / mx_second[si, ci]),
                           ratio_max_over_p99_of_all_s1_samples=float(v / np.percentile(allv, 99))))
facts["maxima"] = maxima
facts["max_where_counts"] = {cat_names[k]: int((codes == k).sum()) for k in cat_names}
facts["n_max_in_validation_part"] = int(in_val_strict.sum())
facts["n_max_touching_validation_windows"] = int(in_val_loose.sum())
facts["max_in_validation_list"] = [f"{m['subject']} {m['channel']}" for m in maxima if m["in_validation_part"]]
facts["max_touching_validation_list"] = [f"{m['subject']} {m['channel']}" for m in maxima if m["touches_validation_windows"]]
facts["max_in_trimmed_first_second_list"] = [f"{m['subject']} {m['channel']}" for m in maxima if m["where"] == "trimmed_first_second"]
facts["max_in_discarded_tail_list"] = [f"{m['subject']} {m['channel']}" for m in maxima if m["where"] == "discarded_tail"]
facts["max_definition"] = ("validation part = the 1 kHz sample lies where a validation window can see it: 100 Hz sample >= 7131 and <= 8908 "
                           "('val_only'; the target-time rule) ; 'touches_validation_windows' also counts 100 Hz samples 7112..7130, which belong to "
                           "the last fit windows and the first validation windows. Trimmed first second = raw sample < 1000.")
arrays.update(emg_max=np.array([[B[s]["emg_max"][c] for c in CH] for s in SUBJ]), emg_max_trial_idx=mx_trial, emg_max_raw_sample=mx_i,
              emg_max_code=codes, emg_max_next_best_other_trial=mx_second,
              emg_max_subjects=np.array(SUBJ), emg_max_channels=np.array(CH))
facts["emg_max_code_legend"] = {str(k): v for k, v in cat_names.items()}
claim = {"YES ta", "IES gl", "IES sol", "EG gm", "CH sol"}
facts["storyboard_claim_check"] = dict(claimed_6_of_24_listed=sorted(claim), recomputed_strict=facts["max_in_validation_list"],
                                       recomputed_touching=facts["max_touching_validation_list"],
                                       claimed_list_subset_of_recomputed_touching=bool(claim <= set(facts["max_touching_validation_list"])))

# the 100 Hz peak of the normalised envelope as the network sees it (fit windows only / val / all S1) -- 'tops out at exactly one'?
peaks = {}
for s in SUBJ:
    b = B[s]
    s1 = [p for p in b["per"] if p["sess"] == "S1"]
    peaks[s] = {}
    for ci, c in enumerate(CH):
        pk_all = max(float(p["env"][:, ci].max()) for p in s1)
        pk_fit = max(float(p["env"][: int(len(p["X"]) * db.TRAIN_FRAC) + W - 1, ci].max()) for p in s1)
        pk_s2 = max(float(p["env"][:, ci].max()) for p in b["per"] if p["sess"] == "S2")
        peaks[s][c] = dict(s1_all_100hz=pk_all, s1_fit_part_100hz=pk_fit, s2_100hz=pk_s2)
facts["normalised_peaks_100hz"] = peaks
facts["normalised_peaks_note"] = ("The scale uses the 1 kHz envelope maximum; after the x10 decimation the 100 Hz peak can fall a hair "
                                  "below 1.0. s1_fit_part is what the network sees in the fit windows only.")
facts["eg_max"] = {c: dict(value=float(B['EG']['emg_max'][c]), trial_idx=int(mx_trial[1, ci]), t_s=float((mx_i[1, ci] + 1) / 1000.0),
                           where=cat_names[int(codes[1, ci])]) for ci, c in enumerate(CH)}
facts["eg_max_in_trial21"] = [int(mx_trial[1, ci]) == EG_S1 for ci in range(4)]

# ----------------------------------------------------------------------------------------------- the example window
tg21 = arrays["s1_target"].astype(np.float64)               # full 100 Hz trial series (samples 0..8909)
tw = np.concatenate([np.full(W - 1, np.nan), p21["y"].astype(np.float64)])   # target aligned to samples
mid = 0.5 * (np.nanpercentile(tw, 5) + np.nanpercentile(tw, 95))
below = np.r_[False, np.nan_to_num(tw, nan=0) < mid, False]
edges = np.flatnonzero(np.diff(below.astype(int)))
runs = [(a, e) for a, e in zip(edges[::2], edges[1::2]) if e - a >= 50]
facts["pushes"] = dict(n=len(runs), midpoint_Nm=float(mid), starts_sample=[int(r[0]) for r in runs], starts_t=[float(T0 + 0.01 * r[0]) for r in runs],
                       method="runs where the target < midpoint of its 5th-95th percentile for >= 0.5 s (same rule as x01)")
check("EG trial 21: 22 pushes found (storyboard c02_b06)", len(runs) == 22, len(runs))
onset = int(runs[2][0])                                      # start of the third push (sample index in the trial)
wlast = onset + 15
w = wlast - (W - 1)                                          # window index within the trial; window ends (target) at sample wlast
X21 = p21["X"]
win = X21[w]
calf_rise = float(win[-5:, :3].mean() - win[:5, :3].mean())
check("example window: lies in the fit part, EMG (GM, GL, SOL) rises across it, target is on the push (below midpoint)",
      w < n_fit_t and calf_rise > 0 and p21["y"][w] < mid, dict(window_index=w, calf_rise=round(calf_rise, 4), target_Nm=float(p21["y"][w]), midpoint=float(mid)))
glob = int(np.flatnonzero(np.concatenate([[p["idx"]] * int(len(p["X"]) * db.TRAIN_FRAC) for p in B["EG"]["per"] if p["sess"] == "S1"]) == EG_S1)[w])
check("example window equals the row of X_train build_dataset returned", np.array_equal(B["EG"]["Xtr"][glob], win) and B["EG"]["ytr"][glob] == p21["y"][w],
      f"X_train row {glob}")
check("example window matches the 100 Hz series (columns gm, gl, sol, ta, angle)",
      np.allclose(win[:, :4], p21["env"][w:w + W], atol=1e-6) and np.allclose(win[:, 4], p21["angle"][w:w + W], atol=1e-6), "columns checked")
arrays.update(window=win.astype(np.float32), window_target=np.float32(p21["y"][w]), window_index=np.int64(w), window_first_sample=np.int64(w),
              window_last_sample=np.int64(wlast), window_t=T0 + 0.01 * np.arange(w, w + W), fit_val_split_index=np.int64(n_fit_t),
              push_start_samples=np.array([r[0] for r in runs]), push_end_samples=np.array([r[1] for r in runs]))
facts["example_window"] = dict(trial_idx=EG_S1, window_index_in_trial=w, samples=[w, wlast], t_first=float(T0 + 0.01 * w), t_last=float(T0 + 0.01 * wlast),
                               target_Nm=float(p21["y"][w]), x_train_row=glob, feature_order=CH + ["angle"], shape=list(win.shape),
                               gm_first_last=[float(win[0, 0]), float(win[-1, 0])], calf_env_rise=calf_rise,
                               window_min=[float(x) for x in win.min(0)], window_max=[float(x) for x in win.max(0)],
                               third_push_onset_sample=onset, note="window ends 0.15 s after the start of the third push; same rule as x05")
facts["fit_val_split"] = dict(trial_idx=EG_S1, n_windows=8891, n_fit=n_fit_t, n_val=n_val_t, n_dropped=1,
                              first_val_window_index=n_fit_t, split_fraction=n_fit_t / 8891,
                              t_split_start_of_first_val_window=T0 + 0.01 * n_fit_t, t_split_first_val_target=T0 + 0.01 * (n_fit_t + W - 1),
                              t_trial_end=T0 + 0.01 * (len(p21["ds"]) - 1), val_seconds_of_targets=n_val_t * 0.01)

# ----------------------------------------------------------------------------------------------- finish
facts["conventions"] = dict(trial_index="df.attrs['trial_index'] = 0-based position in the .flb file (original logs print index+1)",
                            time="t = 1.001 + 0.01 k s since the start of that trial's recording; k = 100 Hz sample after the 1 s trim",
                            feature_order=CH + ["angle"], envelopes="normalised = raw envelope / session-1 maximum of the channel; raw = un-normalised envelope (mV), decimated to 100 Hz",
                            target="torque - passive(angle), float32 as the network sees it")
facts["seed_note"] = "no training here; seed0.npz is used only to check the session-2 targets"
facts["array_keys"] = sorted(arrays.keys())

np.savez_compressed(OUT / "y02.npz", **arrays)
from pathlib import Path  # noqa: E402


def plain(v):
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


with open(OUT / "facts_y02.json", "w", encoding="utf-8") as f:
    json.dump({"y02": plain(facts)}, f, indent=1, sort_keys=True)

print("\n--- summary ---")
print("EG passive S1 / S2 at -0.038:", facts["display"])
print("maxima in validation part (strict):", facts["n_max_in_validation_part"], facts["max_in_validation_list"])
print("maxima touching validation windows:", facts["n_max_touching_validation_windows"], facts["max_touching_validation_list"])
print("maxima in trimmed first second:", facts["max_in_trimmed_first_second_list"], "| in discarded tail:", facts["max_in_discarded_tail_list"])
print("example window:", facts["example_window"]["window_index_in_trial"], "split", n_fit_t)
print("wrote", OUT / "y02.npz", OUT / "facts_y02.json")
