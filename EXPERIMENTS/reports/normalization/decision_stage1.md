# Stage 1 decision (rule of PLAN.md applied mechanically)

Numbers: median over the 3 seeds for each person and position, then median over the cells of the group. MAE in Nm, total torque.

| Arm | toes-down MAE | vs N0 | people better at toes-down | other MAE | worst other person vs N0 | toes-down worst-seed MAE | passes |
|---|---|---|---|---|---|---|---|
| N0 control (session-1 max) | 0.853 | - | - | 1.137 | - | 0.938 | reference |
| N1 session-1 99th percentile | 0.991 | +0.138 (+16 %) | 1 of 3 (JM) | 1.016 | +33 % | 1.045 | no, fails 1 toes-down, 2 people, 3 elsewhere, 4 worst seed |
| N2 MVC | 0.969 | +0.116 (+14 %) | 1 of 3 (JM) | 1.280 | +43 % | 1.004 | no, fails 1 toes-down, 2 people, 3 elsewhere, 4 worst seed |
| N3 per-session 99th percentile | 0.709 | -0.144 (-17 %) | 2 of 3 (YES, EG) | 1.406 | +62 % | 0.759 | no, fails 3 elsewhere |

**No arm passes. N0 (control) stays.**
Stage 2 runs: N0 and N3 (lowest toes-down MAE among the others, for information only).
