# Code audit of the experiment pipeline (24 Oct 2026)

**Scope:** the EXPERIMENTS/ runner and scoring code, the ML/research modules it calls, the alternative models compared in rounds 1–2 (ridge, mixture of experts, other network types), and a read-only review of the teammate's work on origin/main.

**Method:** three independent reviewers, each followed by an adversarial verifier that re-derived every finding. Everything was read-only on the repo, and no model was trained. Scores were recomputed from the saved predictions (.npz).

## Main result

**No data leakage was found in the code.** Nothing from the tested session reaches training, target scaling, early stopping or model selection. The test session's torque is never used.

The EMG auto-gain does use the tested session's own EMG at prediction time. It looks only backwards in time and never at torque, and it starts from training-session values.

**Every score reproduces:**
- Framework runs recompute from the .npz files with a maximum relative difference of 2e-16.
- Original-code (X00) runs recompute to 3e-8.
- Data built separately by the frozen original code and by the framework are identical for JM (X equal; y within 1.9e-6 Nm).

## Findings (verified)

| severity | finding | status in the 24 Oct deck |
|---|---|---|
| major | Not an independent validation. Session-2 scores of YES and EG were used to choose the changes, and JM and CH influenced the auto-gain design. Only IES and HM were not used for any decision (report 1 had looked at them with a similar pipeline). | stated (limitations; IES + HM reported separately) |
| major | Part of the "other positions" gain at JM −0.08 rad comes from fixing a wrong training label (passive table), not from the model. | stated |
| minor | analyze_stage.py prints "/4" for rule 2 whatever the number of cells. Its status labels call some "noise-rule-only" failures "trade-off". Rule 3 averages unfloored R², so one extreme cell decides it. No decision changes. | known; the deck states that no change met all rules |
| minor | FINAL_REPORT.md Stage D table shows rule 4 as failed vs X00; the generated report says passed. FINAL_REPORT also calls JM, IES, CH and HM "never used for choosing", but only IES and HM qualify. | the deck supersedes FINAL_REPORT where they differ |
| minor | No train/validation curves were saved for framework runs. | added: code/learning_curves.py, results/learning_curves/ |
| info | EMG envelopes are zero-phase filtered (about 0.25 s look-ahead), for every model. Results are offline; a causal filter would add about 0.2 s delay. | stated (limitations) |
| info | Validation = last 20 % of the same trials, so early stopping sees only within-session fit. | stated |
| info | check_equivalence.py does not apply the last-validation-window drop. With it applied, the equivalence still holds. | — |

## Alternative models (rounds 1–2)

**Ridge regression is mechanically correct:**
- It uses the same 20-sample window.
- Features are standardised on the training part only.
- Alpha is chosen on the validation part.
- It uses the same split and the same answer key as the LSTM.

Its unbounded quadratic terms produce a few absurd predictions on out-of-range EMG. The LSTM's lead over ridge is therefore smaller than the old notes said, but it is consistent: ridge is worse at 42 of 48 positions on the passive-trial key. The notes also compared ridge with the wrong LSTM version.

**Network types (report 1):** the comparison was fair among the types (same data, targets, settings and seeds) but narrow:
- the alternatives were not tuned;
- it covered 3 people, learn 1 → 2 only;
- each type was scored on its own target.

**Mixture of experts:** the implementation is correct. Its own contribution was never isolated, because the comparison mixes changes of trunk size, dropout and loss scaling.

## Teammate's work (origin/main, read-only)

**The 500 ms window:**
- It lowered his mean RMSE by about 10 % over 5 people.
- It was chosen and confirmed on the retest session; he calls this retrospective.
- His metric is dominated by the strong positions.
- Our 0.5 s test on the final model, with the same seeds, shows the same pattern: the other positions improve a little and the toes-down positions stay about the same.

**Passive torque:**
- His session-specific passive targets are not leakage. They follow the same principle as our answer key.
- His YES −0.49 rad passive value (about 7.4 Nm) disagrees with the recording and our table (7.8–7.9 Nm).

**Worth adopting:** his test suite, the trial manifest, and the passive drift audit.
