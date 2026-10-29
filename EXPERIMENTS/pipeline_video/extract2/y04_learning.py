"""y04: training and learning curves of the baseline (chapters c09 and c13 of the v2 video).

Reads  : DWT/EXPERIMENTS/results/learning_curves/<exp>/<subj>/S1toS2/seed0.json     (exp = X00f_baseline_framework and D01_...)
         DWT/EXPERIMENTS/results/X00_baseline/<subj>/S1toS2/seed<k>.json           (6 subjects x seeds 0-9: epochs_run, best_epoch, train/val loss)
         DWT/EXPERIMENTS/baseline/original_code/ML/config.py                       (hyper-parameters, imported as text, not executed)
         EXPERIMENTS/pipeline_video/data2/facts_y01.json                           (cross-check of the kept-epoch MAEs, if present)
Writes : EXPERIMENTS/pipeline_video/data2/y04.npz
         EXPERIMENTS/pipeline_video/data2/facts_y04.json     (one top-level key: "y04")
Run    : .venv311/Scripts/python.exe EXPERIMENTS/pipeline_video/extract2/y04_learning.py     (numpy + json only, no training)

Conventions
  - epochs are 1-based (as in the saved histories).  best_epoch = the epoch whose weights are kept (min validation loss).
  - keras_loss = training loss as Keras reports it (running mean over the epoch's batches, dropout ON, weights changing).
    keras_val_loss = loss on the validation windows after the epoch (dropout off).  mae_/mse_ fit/val/test = recomputed
    with dropout off after every epoch on the fit / val / test (session 2) windows, in Nm / Nm^2.
  - lr[e] = learning rate logged for epoch e AFTER that epoch's end-of-epoch callbacks (verified below by emulating Keras'
    ReduceLROnPlateau on keras_val_loss: the cut fires at the end of the epoch where the logged lr first drops, after 10 epochs
    without improvement).  So "lr_cut_epoch" = first epoch whose logged lr is lower; the smaller rate first trains epoch lr_cut_epoch+1.
"""
import json
import pathlib
import re

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
VWT = HERE.parents[2]
DWT = pathlib.Path("C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137")
RES = DWT / "EXPERIMENTS/results"
LC = RES / "learning_curves"
CFG = DWT / "EXPERIMENTS/baseline/original_code/ML/config.py"
OUTDIR = VWT / "EXPERIMENTS/pipeline_video/data2"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUBJ = ["CH", "EG", "HM", "IES", "JM", "YES"]
DIR = "S1toS2"
FRAMEWORK = "X00f_baseline_framework"
D01 = "D01_cleanpassive+autogain+equalweight"
SEEDS = range(10)
FIELDS = ["epoch", "keras_loss", "keras_val_loss", "lr", "mae_fit", "mae_val", "mae_test", "mse_fit", "mse_val", "mse_test"]

checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    assert ok, f"CHECK FAILED: {name} {detail}"


def med(x):
    return float(np.median(x))


# ------------------------------------------------------------------ hyper-parameters from the frozen config
cfg_txt = CFG.read_text(encoding="utf-8")
hp = {}
for k in ["LEARNING_RATE", "LR_FACTOR", "LR_PATIENCE", "LR_MIN", "EARLY_PATIENCE", "BATCH_SIZE", "EPOCHS"]:
    m = re.search(rf"^{k}\s*=\s*([0-9.eE+-]+)", cfg_txt, re.M)
    hp[k] = float(m.group(1))
for k in ["LR_PATIENCE", "EARLY_PATIENCE", "BATCH_SIZE", "EPOCHS"]:
    hp[k] = int(hp[k])
check("config.py hyper-parameters", (hp["LEARNING_RATE"], hp["LR_FACTOR"], hp["LR_PATIENCE"], hp["EARLY_PATIENCE"], hp["BATCH_SIZE"], hp["EPOCHS"])
      == (0.003, 0.7, 10, 25, 8, 200), str(hp))


# ------------------------------------------------------------------ one learning curve
def load_curve(exp, s):
    f = LC / exp / s / DIR / "seed0.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    h = d["history"]
    arr = {k: np.array([e[k] for e in h], float) for k in FIELDS}
    n = len(h)
    best = int(d["best_epoch"])
    out = dict(raw=d, arr=arr, n=n, best=best)
    ep = arr["epoch"].astype(int)
    check(f"{exp}/{s}: epochs are 1..n", (ep == np.arange(1, n + 1)).all())
    check(f"{exp}/{s}: json 'epochs' == len(history)", int(d["epochs"]) == n, f"{d['epochs']} vs {n}")
    check(f"{exp}/{s}: best_epoch == argmin(keras_val_loss)+1", int(np.argmin(arr["keras_val_loss"])) + 1 == best,
          f"argmin {int(np.argmin(arr['keras_val_loss'])) + 1} vs {best}")
    check(f"{exp}/{s}: epochs == best_epoch + 25 (early stop)", n == best + hp["EARLY_PATIENCE"] or n == hp["EPOCHS"], f"{n} vs {best}+25")
    # keras val loss is MSE on the validation windows (dropout off): agrees with the recomputed mse_val
    if exp == FRAMEWORK:    # D01 trains on a per-position-weighted loss, so keras_val_loss is not the plain MSE there
        check(f"{exp}/{s}: keras_val_loss ~ mse_val (rel < 1e-3)", np.allclose(arr["keras_val_loss"], arr["mse_val"], rtol=1e-3, atol=1e-6),
              f"max rel {np.max(np.abs(arr['keras_val_loss'] / arr['mse_val'] - 1)):.2e}")
    else:
        out_note = f"D01 keras_val_loss differs from plain mse_val (max rel {np.max(np.abs(arr['keras_val_loss'] / arr['mse_val'] - 1)):.2f}): weighted loss"
        out["val_loss_note"] = out_note
    # LR cuts
    lr = arr["lr"]
    cut_idx = np.where(lr[1:] < lr[:-1] * (1 - 1e-6))[0]            # index e (0-based) where lr drops between epoch e+1 and e+2
    cut_triggered = (cut_idx + 2).astype(int)                       # 1-based epoch whose end fires the cut (= first epoch with the lower logged lr)
    cut_first_new = cut_triggered + 1                               # first 1-based epoch actually trained with the smaller lr
    ratios = lr[cut_idx + 1] / lr[cut_idx]
    check(f"{exp}/{s}: every lr cut is x0.7", np.allclose(ratios, hp["LR_FACTOR"], rtol=1e-5), str(ratios.round(5).tolist()))
    check(f"{exp}/{s}: lr starts at 0.003", abs(lr[0] - hp["LEARNING_RATE"]) < 1e-9, str(lr[0]))
    # emulate Keras ReduceLROnPlateau(patience 10, min_delta 1e-4) and EarlyStopping(patience 25) on keras_val_loss
    v = arr["keras_val_loss"]
    best_r, wait_r, cuts_sim = np.inf, 0, []
    best_e, best_x = np.inf, 0
    stop_sim = None
    for e in range(n):
        if v[e] < best_r - 1e-4:
            best_r, wait_r = v[e], 0
        else:
            wait_r += 1
            if wait_r >= hp["LR_PATIENCE"]:
                cuts_sim.append(e + 1)
                wait_r = 0
        if v[e] < best_e:
            best_e, best_x = v[e], e + 1
        if stop_sim is None and (e + 1) - best_x >= hp["EARLY_PATIENCE"]:
            stop_sim = e + 1
    check(f"{exp}/{s}: lr cut epochs == Keras ReduceLROnPlateau emulation", cuts_sim == cut_triggered.tolist(), f"{cuts_sim} vs {cut_triggered.tolist()}")
    check(f"{exp}/{s}: stop epoch == EarlyStopping emulation", stop_sim == n or (stop_sim is None and n == hp["EPOCHS"]), f"{stop_sim} vs {n}")
    out.update(lr_cut_triggered=cut_triggered, lr_cut_first_new_epoch=cut_first_new, n_cuts=int(len(cut_idx)))
    out["epoch_min_mae_test"] = int(np.argmin(arr["mae_test"])) + 1
    out["epoch_min_mae_val"] = int(np.argmin(arr["mae_val"])) + 1
    out["epoch_min_mae_fit"] = int(np.argmin(arr["mae_fit"])) + 1
    b = best - 1
    out["kept"] = {k: float(arr[k][b]) for k in FIELDS}
    out["kept"]["ratio_test_over_val"] = float(arr["mae_test"][b] / arr["mae_val"][b])
    out["kept"]["ratio_val_over_fit"] = float(arr["mae_val"][b] / arr["mae_fit"][b])
    out["kept"]["ratio_test_over_fit"] = float(arr["mae_test"][b] / arr["mae_fit"][b])
    out["final_lr"] = float(lr[-1])
    out["kept_lr"] = float(lr[b])
    # early drop: how much of the first-to-best validation improvement is done in the first 3 epochs
    tot = v[0] - v[b]
    out["share_of_val_drop_in_first_3_epochs"] = float((v[0] - v[min(2, n - 1)]) / tot) if (tot > 0 and n > 2 and b >= 2) else None
    out["val_first_over_best"] = float(v[0] / v[b])
    return out


def curve_facts(c):
    r = c["raw"]
    a = c["arr"]
    return dict(
        epochs_run=c["n"], best_epoch=c["best"], patience_epochs_after_best=c["n"] - c["best"],
        lr_cut_at_end_of_epoch=c["lr_cut_triggered"].tolist(), lr_cut_first_epoch_with_new_lr=c["lr_cut_first_new_epoch"].tolist(),
        n_lr_cuts=c["n_cuts"], lr_values_visited=sorted(set(a["lr"].tolist()), reverse=True), kept_epoch_lr=c["kept_lr"], final_lr=c["final_lr"],
        epoch_with_lowest_mae_test=c["epoch_min_mae_test"], lowest_mae_test=float(a["mae_test"].min()),
        mae_test_at_kept=c["kept"]["mae_test"], test_best_is_earlier_than_kept=bool(c["epoch_min_mae_test"] < c["best"]),
        test_best_is_later_than_kept=bool(c["epoch_min_mae_test"] > c["best"]),
        epoch_with_lowest_mae_val=c["epoch_min_mae_val"], epoch_with_lowest_mae_fit=c["epoch_min_mae_fit"],
        kept=c["kept"], share_of_val_drop_in_first_3_epochs=c["share_of_val_drop_in_first_3_epochs"],
        val_loss_first_epoch=float(a["keras_val_loss"][0]), val_loss_at_kept=float(a["keras_val_loss"][c["best"] - 1]),
        keras_loss_first_epoch=float(a["keras_loss"][0]), keras_loss_last_epoch=float(a["keras_loss"][-1]),
        seconds=float(r.get("seconds", np.nan)), config=r.get("config"), n_windows=r.get("n_windows"),
        final_per_trial_angles_fit=[t["angle"] for t in sorted(r["final_per_trial"]["fit"], key=lambda t: t["trial"])],
    )


arrays = {}
facts = {
    "source": "learning_curves seed0 json (X00f, D01), X00_baseline seed jsons, original_code config.py",
    "epoch_convention": "1-based; logged lr[e] is read after the end-of-epoch callbacks, so the first epoch with a lower lr is the epoch whose end fired the cut; the smaller rate first trains epoch e+1",
    "hyperparameters": hp,
}

# ------------------------------------------------------------------ X00f and D01 learning curves
curves = {}
for tag, exp in [("x00f", FRAMEWORK), ("d01", D01)]:
    curves[tag] = {}
    for s in SUBJ:
        c = load_curve(exp, s)
        curves[tag][s] = c
        if c is None:
            continue
        a = c["arr"]
        for k in FIELDS:
            arrays[f"{tag}_{s}_{k}"] = a[k]
        arrays[f"{tag}_{s}_best_epoch"] = np.array(c["best"])
        arrays[f"{tag}_{s}_lr_cut_triggered"] = c["lr_cut_triggered"]
        arrays[f"{tag}_{s}_lr_cut_first_new_epoch"] = c["lr_cut_first_new_epoch"]
        arrays[f"{tag}_{s}_epoch_min_mae_test"] = np.array(c["epoch_min_mae_test"])
    facts[f"{tag}_available_subjects"] = [s for s in SUBJ if curves[tag][s] is not None]
    facts[f"{tag}_experiment"] = exp
    facts[f"{tag}_curves"] = {s: curve_facts(curves[tag][s]) for s in SUBJ if curves[tag][s] is not None}

# ------------------------------------------------------------------ steps per epoch
n_fit = 56896
facts["steps_per_epoch"] = dict(n_fit_windows=n_fit, batch_size=hp["BATCH_SIZE"], steps=int(np.ceil(n_fit / hp["BATCH_SIZE"])),
                                exact_division=bool(n_fit % hp["BATCH_SIZE"] == 0), n_val_windows=14224, n_test_windows=71128,
                                per_trial_fit=7112, per_trial_val=1778, n_trials_session1=8)
check("steps per epoch = 56896/8 = 7112", facts["steps_per_epoch"]["steps"] == 7112 and facts["steps_per_epoch"]["exact_division"])
for tag in curves:
    for s in SUBJ:
        c = curves[tag][s]
        if c:
            check(f"{tag}/{s}: n_windows fit/val/test", c["raw"]["n_windows"] == {"fit": 56896, "val": 14224, "test": 71128}, str(c["raw"]["n_windows"]))
facts["total_weight_updates_example"] = {s: curves["x00f"][s]["n"] * 7112 for s in SUBJ}

# ------------------------------------------------------------------ the 60 X00 runs
ep_run = np.zeros((len(SUBJ), 10), int)
ep_best = np.zeros((len(SUBJ), 10), int)
per_run = []
eg0 = None
for i, s in enumerate(SUBJ):
    for k in SEEDS:
        j = json.load(open(RES / "X00_baseline" / s / DIR / f"seed{k}.json"))
        tl, vl = np.array(j["train_loss"], float), np.array(j["val_loss"], float)
        ep_run[i, k] = int(j["epochs_run"])
        ep_best[i, k] = int(j["best_epoch"])
        check(f"X00 {s} seed{k}: len(train_loss)==len(val_loss)==epochs_run", len(tl) == len(vl) == ep_run[i, k])
        check(f"X00 {s} seed{k}: best_epoch == argmin(val_loss)+1", int(np.argmin(vl)) + 1 == ep_best[i, k], f"{int(np.argmin(vl)) + 1} vs {ep_best[i, k]}")
        check(f"X00 {s} seed{k}: epochs_run == best_epoch + 25", ep_run[i, k] == ep_best[i, k] + 25, f"{ep_run[i, k]} vs {ep_best[i, k]}+25")
        check(f"X00 {s} seed{k}: settings match config", j["settings"]["LR_FACTOR"] == 0.7 and j["settings"]["EARLY_PATIENCE"] == 25
              and j["settings"]["EPOCHS"] == 200 and j["settings"]["BATCH_SIZE"] == 8)
        per_run.append(dict(subject=s, seed=k, epochs_run=int(ep_run[i, k]), best_epoch=int(ep_best[i, k]),
                            n_params=j["n_params"], train_s=float(j["train_s"]), min_val_loss=float(vl.min())))
        if s == "EG" and k == 0:
            eg0 = (tl, vl, j)
check("all 60 runs have 1001 parameters", all(r["n_params"] == 1001 for r in per_run))
arrays["x00_epochs_run"] = ep_run
arrays["x00_best_epoch"] = ep_best
arrays["x00_subjects"] = np.array(SUBJ)
arrays["x00_eg_seed0_train_loss"] = eg0[0]
arrays["x00_eg_seed0_val_loss"] = eg0[1]
arrays["x00_eg_seed0_epoch"] = np.arange(1, len(eg0[0]) + 1)

facts["x00_runs"] = dict(
    n_runs=len(per_run), epochs_run_min=int(ep_run.min()), epochs_run_max=int(ep_run.max()), epochs_run_median=med(ep_run),
    best_epoch_min=int(ep_best.min()), best_epoch_max=int(ep_best.max()), best_epoch_median=med(ep_best),
    n_runs_reaching_200_epochs=int((ep_run >= 200).sum()), all_epochs_run_equals_best_plus_25=bool((ep_run == ep_best + 25).all()),
    argmin_epochs_run=dict(zip(["subject", "seed"], [SUBJ[int(np.unravel_index(ep_run.argmin(), ep_run.shape)[0])], int(np.unravel_index(ep_run.argmin(), ep_run.shape)[1])])),
    argmax_epochs_run=dict(zip(["subject", "seed"], [SUBJ[int(np.unravel_index(ep_run.argmax(), ep_run.shape)[0])], int(np.unravel_index(ep_run.argmax(), ep_run.shape)[1])])),
    argmin_best_epoch=dict(zip(["subject", "seed"], [SUBJ[int(np.unravel_index(ep_best.argmin(), ep_best.shape)[0])], int(np.unravel_index(ep_best.argmin(), ep_best.shape)[1])])),
    argmax_best_epoch=dict(zip(["subject", "seed"], [SUBJ[int(np.unravel_index(ep_best.argmax(), ep_best.shape)[0])], int(np.unravel_index(ep_best.argmax(), ep_best.shape)[1])])),
    per_subject={s: dict(epochs_run=ep_run[i].tolist(), best_epoch=ep_best[i].tolist(), epochs_run_min=int(ep_run[i].min()), epochs_run_max=int(ep_run[i].max()),
                         best_epoch_min=int(ep_best[i].min()), best_epoch_max=int(ep_best[i].max())) for i, s in enumerate(SUBJ)},
    train_hours_total=float(sum(r["train_s"] for r in per_run) / 3600),
    per_run=per_run,
)
eg_tl, eg_vl, eg_j = eg0
facts["x00_eg_seed0"] = dict(epochs_run=len(eg_tl), best_epoch=int(eg_j["best_epoch"]), train_loss_first=float(eg_tl[0]), train_loss_last=float(eg_tl[-1]),
                             val_loss_first=float(eg_vl[0]), val_loss_best=float(eg_vl.min()), val_loss_last=float(eg_vl[-1]),
                             train_loss_above_val_loss_all_epochs=bool((eg_tl > eg_vl).all()),
                             note="train_loss is Keras' running mean over the epoch with dropout ON, so it sits above val_loss (dropout OFF)")

# ------------------------------------------------------------------ numbers at the kept epoch (X00f, seed 0) and storyboard cross-check
rows = []
for s in SUBJ:
    k = curves["x00f"][s]["kept"]
    rows.append(dict(subject=s, best_epoch=curves["x00f"][s]["best"], mae_fit=k["mae_fit"], mae_val=k["mae_val"], mae_test=k["mae_test"],
                     ratio_test_over_val=k["ratio_test_over_val"], ratio_val_over_fit=k["ratio_val_over_fit"], ratio_test_over_fit=k["ratio_test_over_fit"]))
medians = {k: med([r[k] for r in rows]) for k in ["mae_fit", "mae_val", "mae_test", "ratio_test_over_val", "ratio_val_over_fit", "ratio_test_over_fit"]}
facts["kept_epoch_x00f_seed0"] = dict(per_subject=rows, median_over_subjects=medians,
                                      median_of_ratios_vs_ratio_of_medians=dict(median_of_ratios=medians["ratio_test_over_val"],
                                                                                ratio_of_medians=medians["mae_test"] / medians["mae_val"]),
                                      n_subjects_test_gt_val=int(sum(r["mae_test"] > r["mae_val"] for r in rows)),
                                      n_subjects_val_gt_fit=int(sum(r["mae_val"] > r["mae_fit"] for r in rows)),
                                      n_subjects_test_best_epoch_earlier_than_kept=int(sum(curves["x00f"][s]["epoch_min_mae_test"] < curves["x00f"][s]["best"] for s in SUBJ)),
                                      n_subjects_test_best_epoch_later_than_kept=int(sum(curves["x00f"][s]["epoch_min_mae_test"] > curves["x00f"][s]["best"] for s in SUBJ)),
                                      n_subjects_test_best_epoch_equal_kept=int(sum(curves["x00f"][s]["epoch_min_mae_test"] == curves["x00f"][s]["best"] for s in SUBJ)),
                                      test_best_vs_kept={s: dict(kept_epoch=curves["x00f"][s]["best"], test_best_epoch=curves["x00f"][s]["epoch_min_mae_test"],
                                                                 mae_test_kept=curves["x00f"][s]["kept"]["mae_test"], mae_test_best=float(curves["x00f"][s]["arr"]["mae_test"].min()))
                                                         for s in SUBJ})
rows_d = []
for s in SUBJ:
    if curves["d01"][s] is None:
        continue
    k = curves["d01"][s]["kept"]
    rows_d.append(dict(subject=s, best_epoch=curves["d01"][s]["best"], mae_fit=k["mae_fit"], mae_val=k["mae_val"], mae_test=k["mae_test"],
                       ratio_test_over_val=k["ratio_test_over_val"]))
if rows_d:
    facts["kept_epoch_d01_seed0"] = dict(per_subject=rows_d, median_over_subjects={k: med([r[k] for r in rows_d]) for k in ["mae_fit", "mae_val", "mae_test", "ratio_test_over_val"]})

arrays["kept_subjects"] = np.array(SUBJ)
arrays["kept_x00f_mae_fit_val_test"] = np.array([[r["mae_fit"], r["mae_val"], r["mae_test"]] for r in rows])
arrays["kept_x00f_ratio_test_over_val"] = np.array([r["ratio_test_over_val"] for r in rows])
arrays["kept_x00f_median_mae_fit_val_test"] = np.array([medians["mae_fit"], medians["mae_val"], medians["mae_test"]])

# cross-check vs facts_y01 (independent recomputation by another script)
fy = OUTDIR / "facts_y01.json"
if fy.exists():
    y1 = json.load(open(fy))
    e = y1["E_learning_curves"][FRAMEWORK]
    good = True
    for r in rows:
        v = e[r["subject"]]
        good &= abs(v["mae_fit"] - r["mae_fit"]) < 1e-4 and abs(v["mae_val"] - r["mae_val"]) < 1e-4 and abs(v["mae_test"] - r["mae_test"]) < 1e-4
        good &= v["best_epoch"] == r["best_epoch"] and v["epoch_of_min_mae_test"] == curves["x00f"][r["subject"]]["epoch_min_mae_test"]
    check("per-subject kept-epoch MAEs, best_epoch and test-best epoch agree with facts_y01 E", good)
    mm = e["_median_over_subjects"]
    check("median MAE fit/val/test agree with facts_y01 E", abs(mm["mae_fit"] - medians["mae_fit"]) < 1e-4 and abs(mm["mae_val"] - medians["mae_val"]) < 1e-4
          and abs(mm["mae_test"] - medians["mae_test"]) < 1e-4, f"{mm} vs {medians}")

# also verify the kept-epoch numbers against the independent per-trial summary stored at the end of each run: mean of trial mae (equal-size trials) = pooled MAE?
for s in SUBJ:
    c = curves["x00f"][s]
    for sp, key in [("fit", "mae_fit"), ("val", "mae_val"), ("test", "mae_test")]:
        tm = np.mean([t["mae"] for t in c["raw"]["final_per_trial"][sp]])
        check(f"x00f/{s}: mean of final per-trial {sp} MAE == mae_{sp} at the kept epoch", abs(tm - c["kept"][key]) < 1e-6 * max(1, c["kept"][key]) + 1e-6,
              f"{tm:.6f} vs {c['kept'][key]:.6f}")

# ------------------------------------------------------------------ recommendation for the "epochs" visual
cand = []
for s in SUBJ:
    c = curves["x00f"][s]
    v = c["arr"]["keras_val_loss"]
    cand.append(dict(subject=s, epochs_run=c["n"], best_epoch=c["best"], n_lr_cuts=c["n_cuts"], cuts_at_end_of_epoch=c["lr_cut_triggered"].tolist(),
                     val_first_over_best=c["val_first_over_best"], share_drop_first3=c["share_of_val_drop_in_first_3_epochs"],
                     val_loss_first=float(v[0]), val_loss_best=float(v[c["best"] - 1]),
                     cut_before_best=int(sum(t < c["best"] for t in c["lr_cut_triggered"])),
                     cut_after_best=int(sum(t >= c["best"] for t in c["lr_cut_triggered"])),
                     test_best_earlier=bool(c["epoch_min_mae_test"] < c["best"])))
facts["epochs_visual_candidates_x00f"] = cand


def rec_score(r):
    # visual wants: >=2 visible cuts, early drop (large first/best ratio, most of the drop in 3 epochs), a long enough run to see the plateau,
    # and a stop well before 200
    return (min(r["n_lr_cuts"], 3) * 1.0 + (1.0 if (r["share_drop_first3"] or 0) > 0.6 else 0.0) + (0.5 if r["val_first_over_best"] > 1.5 else 0.0)
            + (0.5 if 40 <= r["epochs_run"] <= 120 else 0.0))


for r in cand:
    r["score"] = rec_score(r)
best_c = max(cand, key=lambda r: (r["score"], r["n_lr_cuts"]))
facts["epochs_visual_recommendation"] = dict(subject=best_c["subject"], reason=(
    f"{best_c['subject']}: val loss falls {best_c['val_loss_first']:.2f} -> {best_c['val_loss_best']:.2f} (best epoch {best_c['best_epoch']}), "
    f"{best_c['n_lr_cuts']} lr cuts at the end of epochs {best_c['cuts_at_end_of_epoch']}, stops at epoch {best_c['epochs_run']} of 200"), scores={r['subject']: r['score'] for r in cand})

# ------------------------------------------------------------------ storyboard cross-check (c09, c13)
sb = {
    "c09_b03 steps per epoch": (7112, facts["steps_per_epoch"]["steps"]),
    "c09_b06 epochs_run min": (32, facts["x00_runs"]["epochs_run_min"]),
    "c09_b06 epochs_run max": (115, facts["x00_runs"]["epochs_run_max"]),
    "c09_b06 best_epoch min": (7, facts["x00_runs"]["best_epoch_min"]),
    "c09_b06 best_epoch max": (90, facts["x00_runs"]["best_epoch_max"]),
    "c13_b02 median MAE fit (Nm)": (0.58, round(medians["mae_fit"], 2)),
    "c13_b02 median MAE val (Nm)": (0.59, round(medians["mae_val"], 2)),
    "c13_b02 median MAE test (Nm)": (1.18, round(medians["mae_test"], 2)),
    "c13_b04 test/val CH": (1.2, round(rows[0]["ratio_test_over_val"], 1)),
    "c13_b04 test/val EG": (2.1, round(rows[1]["ratio_test_over_val"], 1)),
    "c13_b04 test/val HM": (1.7, round(rows[2]["ratio_test_over_val"], 1)),
    "c13_b04 test/val IES": (1.9, round(rows[3]["ratio_test_over_val"], 1)),
    "c13_b04 test/val JM": (6.0, round(rows[4]["ratio_test_over_val"], 1)),
    "c13_b04 test/val YES": (1.4, round(rows[5]["ratio_test_over_val"], 1)),
    "c13_b05 subjects with test-best epoch earlier than kept": (5, facts["kept_epoch_x00f_seed0"]["n_subjects_test_best_epoch_earlier_than_kept"]),
    "c13_b05 JM lowest test MAE (Nm)": (1.95, round(curves["x00f"]["JM"]["arr"]["mae_test"].min(), 2)),
    "c13_b05 JM test MAE at kept epoch (Nm)": (2.12, round(curves["x00f"]["JM"]["kept"]["mae_test"], 2)),
    "c09_b05 lr cut factor": (0.7, hp["LR_FACTOR"]),
    "c09_b06 epoch limit": (200, hp["EPOCHS"]),
    "c09_b06 patience": (25, hp["EARLY_PATIENCE"]),
}
facts["storyboard_cross_check"] = {k: dict(storyboard=a, computed=b, match=bool(a == b)) for k, (a, b) in sb.items()}
facts["storyboard_mismatches"] = [f"{k}: storyboard {a} vs computed {b}" for k, (a, b) in sb.items() if a != b]

# display values (rounded) kept separately
facts["display"] = dict(
    steps_per_epoch="7,112", x00_epochs_range=f"{facts['x00_runs']['epochs_run_min']}-{facts['x00_runs']['epochs_run_max']}",
    x00_best_range=f"{facts['x00_runs']['best_epoch_min']}-{facts['x00_runs']['best_epoch_max']}",
    median_mae_fit_val_test_nm=[round(medians[k], 2) for k in ["mae_fit", "mae_val", "mae_test"]],
    ratios_test_over_val={r["subject"]: round(r["ratio_test_over_val"], 2) for r in rows},
    recommended_epochs_visual=best_c["subject"],
)
facts["checks"] = checks
facts["n_checks"] = len(checks)
facts["all_checks_ok"] = all(c["ok"] for c in checks)

np.savez_compressed(OUTDIR / "y04.npz", **arrays)


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


with open(OUTDIR / "facts_y04.json", "w", encoding="utf-8") as f:
    json.dump({"y04": plain(facts)}, f, indent=1)

# ------------------------------------------------------------------ print
print(f"{len(checks)} checks passed")
print("steps/epoch", facts["steps_per_epoch"]["steps"])
print("X00: epochs_run", ep_run.min(), ep_run.max(), "best", ep_best.min(), ep_best.max(), "| runs at 200:", (ep_run >= 200).sum())
print("X00f kept epoch (seed 0):")
for r in rows:
    print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
print("  medians", {k: round(v, 3) for k, v in medians.items()})
print("D01:", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows_d])
print("candidates:")
for r in cand:
    print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
print("test-best vs kept:", {s: (v["kept_epoch"], v["test_best_epoch"]) for s, v in facts["kept_epoch_x00f_seed0"]["test_best_vs_kept"].items()})
print("recommend:", facts["epochs_visual_recommendation"])
print("MISMATCHES:", facts["storyboard_mismatches"])
print("wrote", OUTDIR / "y04.npz")
