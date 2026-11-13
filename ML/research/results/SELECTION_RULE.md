# Selection rule (written 2026-10-05 18:40, BEFORE any held-out-subject result was looked at)

* The recommended configuration is chosen ONLY from the screening study
  (subjects YES, JM, EG; protocols S1toS2 + S2toS1; 3 seeds):
  the causal configuration with the highest primary metric (mean per-angle
  R² vs the common reference truth, floored at -1, mean over subjects,
  folds and seeds).  Transductive and 1-seed configurations are not eligible.
* The honest performance estimate of this choice is the cross-fitted score
  (choose on one fold, score on the other) reported by analyze.py.
* The held-out subjects (IES, CH, HM; study "confirm") are used ONLY to check
  the result.  They never change the choice.  Configurations run there:
  E01_old and E03_final (references), E22_restanchor_final and
  E31_restanchor_agc_lstm8 (final candidates), seed 0, both folds.
* 2026-10-05 19:05: E29_final_win40 (400 ms) seed 0 was better than E03 seed 0 in 6/6 screening cells -> seeds 1, 2 added on the screening subjects so it becomes eligible under this rule, and E29 seed 0 added to the held-out check (decision made on screening data only; held-out results still not looked at).
* 2026-10-05 19:23 FINAL SELECTION (screening data only, before opening any held-out result):
  E29_final_win40 (E03 with a 400 ms window), mean primary 0.9321 (E03 0.9253, E22 0.9246);
  selected on both folds; cross-fitted score 0.932; leave-one-subject-out 0.925.
* 2026-10-05 19:28 POST-HOC DIAGNOSTIC (after unblinding, not used for selection): E23 and E32 on CH only, to find whether auto-gain or the MoE head fails at CH's direction-flip angle.
* 2026-10-05 19:37 AFTER UNBLINDING: the rule's pick E29 did not generalise best on the held-out subjects
  (IES has no passive recordings -> passive-map-based labels fail; CH pushes in both directions -> auto-gain fails at one angle).
  The report RECOMMENDS E22 (rest-anchored labels) with report 1's AGC 'auto' rule (off only for CH). This recommendation is
  post-hoc (informed by the held-out check), and is labelled as such in the report.
