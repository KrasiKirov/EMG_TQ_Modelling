"""
Registry of named configurations.  A configuration = passive model + EMG
normalisation + model + loss balancing (+ optional extra plantarflexion
weight).  Each entry says what question it answers.

Reference points
----------------
E01_old       the original pipeline
E02_minimal   bug-fixed passive map + passive gain correction + angle-weighted
              loss, original LSTM (the earlier "plan B")
E03_final     E02 + causal EMG auto-gain + mixture-of-experts head with
              per-angle scaled loss (the earlier `train.py --recommended`)
"""

CONFIGS = {
    # ── reference points ───────────────────────────────────────────────────
    'E01_old': dict(passive='legacy', emg='trainmax', model='lstm8', balance='none',
                    q='Original pipeline (baseline).'),
    'E02_minimal': dict(passive='gain', emg='trainmax', model='lstm8', balance='weights',
                        q='Previous minimal fix.'),
    'E03_final': dict(passive='gain', emg='agc', model='moe', balance='scale',
                      q='Previous final pipeline (--recommended).'),
    # ── passive torque: use the before-holds / model the relaxation ─────────
    'E04_final_holdgain': dict(passive='hold_gain', emg='agc', model='moe', balance='scale',
                               q='Each trial uses ITS OWN before-hold instead of the angle map.'),
    'E05_final_holdend': dict(passive='holdend_gain', emg='agc', model='moe', balance='scale',
                              q='Use the END of the before-hold (already relaxed) as passive level.'),
    'E06_final_drift': dict(passive='drift', emg='agc', model='moe', balance='scale',
                            q='Time-varying passive torque (stress relaxation during the trial), fitted on training rest periods.'),
    'E07_final_holdextrap': dict(passive='holdextrap', emg='agc', model='moe', balance='scale',
                                 q="Continue the before-hold's own relaxation curve through the trial."),
    'E08_minimal_drift': dict(passive='drift', emg='trainmax', model='lstm8', balance='weights',
                              q='Drift model on the simple model (no auto-gain, no MoE).'),
    # ── EMG normalisation ───────────────────────────────────────────────────
    'E09_final_agcsoft': dict(passive='gain', emg='agc_soft', model='moe', balance='scale',
                              q='Auto-gain that amplifies weak channels less (less noise at weak angles).'),
    'E10_final_mvcglobal': dict(passive='gain', emg='mvc_global', model='moe', balance='scale',
                                q='EMG as % of the MVC instead of auto-gain.'),
    'E11_final_mvcangle': dict(passive='gain', emg='mvc_angle', model='moe', balance='scale',
                               q="EMG as % of THIS angle's MVC."),
    'E12_final_noagc': dict(passive='gain', emg='trainmax', model='moe', balance='scale',
                            q='Final pipeline without auto-gain.'),
    # ── extra weight on the plantarflexed angles (user idea) ───────────────
    'E13_final_pf3': dict(passive='gain', emg='agc', model='moe', balance='scale', pf_extra=(2, 3.0),
                          q='x3 loss weight on the 2 most plantarflexed angles.'),
    'E14_final_pf10': dict(passive='gain', emg='agc', model='moe', balance='scale', pf_extra=(2, 10.0),
                           q='x10 loss weight on the 2 most plantarflexed angles.'),
    # ── EMG gain augmentation (models.RandomChannelGain) ────────────────────
    'E15_final_ga20': dict(passive='gain', emg='agc', model='moe+ga20', balance='scale',
                           q='Final pipeline + mild EMG gain augmentation (sigma 0.2).'),
    'E16_final_ga50': dict(passive='gain', emg='agc', model='moe+ga50', balance='scale',
                           q='Final pipeline + strong EMG gain augmentation (sigma 0.5).'),
    'E17_noagc_ga50': dict(passive='gain', emg='trainmax', model='moe+ga50', balance='scale',
                           q='Augmentation INSTEAD of auto-gain (simpler, no online normalisation).'),
    'E18_minimal_ga50': dict(passive='gain', emg='trainmax', model='lstm8+ga50', balance='weights',
                             q='Simplest fix (gain passive + weighted loss, original LSTM) + augmentation.'),
    # ── cross-subject pre-training (transfer.py) ───────────────────────────
    'E19_minimal_agc': dict(passive='gain', emg='agc', model='lstm8', balance='weights',
                            q='Control for E20: original LSTM + auto-gain, no pre-training.'),
    'E20_xsub_agc': dict(passive='gain', emg='agc', model='lstm8', balance='weights', pretrain=True,
                         q='Pre-train on the 5 other subjects, fine-tune on the target (transfer learning).'),
    'E21_xsub_trainmax': dict(passive='gain', emg='trainmax', model='lstm8', balance='weights', pretrain=True,
                              q='Same as E20 without auto-gain (control: E02_minimal).'),
    # ── training labels from the in-trial passive baseline (truth.py, M1) ──
    'E22_restanchor_final': dict(passive='restanchor', emg='agc', model='moe', balance='scale',
                                 q='Training labels = measured - baseline through the rest periods of the same trial.'),
    'E23_restanchor_minimal': dict(passive='restanchor', emg='trainmax', model='lstm8', balance='weights',
                                   q='Same labels with the simplest model (original LSTM, weighted loss, no auto-gain).'),
    # ── relaxed-window ablation (the user's question, M5) ──────────────────
    'E24_final_norest': dict(passive='gain', emg='agc', model='moe', balance='scale', rest_weight=0,
                             q='Relaxed (EMG-rest) windows REMOVED from training.'),
    'E25_final_rest02': dict(passive='gain', emg='agc', model='moe', balance='scale', rest_weight=0.2,
                             q='Relaxed windows down-weighted to 0.2.'),
    # ── per-session EMG normalisation without labels (M2-N4, transductive) ─
    'E26_final_sessp99': dict(passive='gain', emg='session_p99', model='moe', balance='scale',
                              q='Each session scaled by its own EMG 99th percentile (uses unlabelled test EMG).'),
    # ── classical baseline (M6) ──────────────────────────────────────────────
    # (balance='none': the ridge has separate coefficients per angle, so it is
    #  already balanced; per-angle weights would only change the effective
    #  regularisation of each angle.)
    'E27_ridge': dict(passive='gain', emg='trainmax', model='ridge', balance='none',
                      q='Ridge regression on lagged envelopes (+squares) per angle: is a deep model needed?'),
    'E28_ridge_agc': dict(passive='gain', emg='agc', model='ridge', balance='none',
                          q='Ridge with auto-gain inputs.'),
    # ── window length (M7) ──────────────────────────────────────────────────
    'E29_final_win40': dict(passive='gain', emg='agc', model='moe', balance='scale', window=40,
                            q='400 ms input window instead of 200 ms.'),
    'E30_final_win60': dict(passive='gain', emg='agc', model='moe', balance='scale', window=60,
                            q='600 ms input window.'),
    # ── round 2 (added after the first screening results were seen) ───────
    # completes the 2x2 design {passive: gain / restanchor} x {model: lstm8 /
    # moe} at emg='agc' (E19, E03, E22 are the other three cells)
    'E31_restanchor_agc_lstm8': dict(passive='restanchor', emg='agc', model='lstm8', balance='weights',
                                     q='Simplest candidate: rest-anchored labels + auto-gain + the ORIGINAL small LSTM.'),
    # ── post-hoc diagnostic (after the held-out check; NOT eligible for selection) ──
    # CH pushes toes-up at one angle in session 1 and toes-down in session 2.
    # Is the auto-gain or the per-angle MoE head responsible for the failure there?
    'E32_restanchor_noagc_moe': dict(passive='restanchor', emg='trainmax', model='moe', balance='scale',
                                     q='Diagnostic: E22 without auto-gain (does auto-gain break the direction flip?).'),
    # ── the recommended final pipeline (report 2, section 10) ───────────────
    # = E22 with report 1's auto-gain 'auto' rule: identical to E22 for every
    # subject who pushes one way, identical to E32 for one who pushes both
    # ways (only CH among the 6 subjects).
    'FINAL': dict(passive='restanchor', emg='agc_auto', model='moe', balance='scale',
                  q='Recommended pipeline: rest-anchored labels + auto-gain (auto rule) + mixture-of-experts LSTM.'),
    # ── leave-one-out ablations of FINAL (added for the presentation) ───────
    # Each removes ONE ingredient of FINAL and keeps everything else the same,
    # to show what that ingredient contributes.  Run in study 'ablation'.
    # FINAL without auto-gain is E32 (above).
    'E33_final_noscale': dict(passive='restanchor', emg='agc_auto', model='moe', balance='none',
                              q='Ablation: FINAL without the per-angle target scaling (plain MSE in Nm^2).'),
    'E34_final_oldpassive': dict(passive='legacy', emg='agc_auto', model='moe', balance='scale',
                                 q='Ablation: FINAL with the old passive table instead of the in-trial passive level.'),
    'E35_final_lstm8': dict(passive='restanchor', emg='agc_auto', model='lstm8', balance='scale',
                            q='Ablation: FINAL with the old small model (LSTM 8-8, no per-angle experts).'),
    # ── passive trials only (supervisor's choice, 2026-10-08) ───────────────
    # Training labels = measured − passive torque from the clean static holds
    # of the passive trials of the TRAINING session (no correction fitted on
    # in-trial rest).  Score with the passive trials of the TESTED session
    # (key 'passive_trial' in notes/final_pipeline/results_tables).
    'E36_static_old': dict(passive='static', emg='trainmax', model='lstm8', balance='none',
                           q='Old pipeline with the clean passive-trial table (JM bug removed), nothing else changed.'),
    'E37_static_full': dict(passive='static', emg='agc_auto', model='moe', balance='scale',
                            q='Passive trials only + auto-gain (auto rule) + per-angle corrections + divide by swing.'),
    'E38_static_full_win40': dict(passive='static', emg='agc_auto', model='moe', balance='scale', window=40,
                                  q='E37 with a 0.4 s window.'),
}


def get(name):
    cfg = dict(CONFIGS[name])
    cfg.setdefault('pf_extra', None)
    cfg.setdefault('pretrain', False)
    cfg.setdefault('rest_weight', None)
    cfg.setdefault('window', 20)
    return cfg
