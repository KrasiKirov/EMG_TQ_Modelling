# Presentation, 3 Nov 2026

Three slides. The live deck is the claude.ai artifact <https://claude.ai/artifact/RMp9vjNmHnDEgtvEAEfPq1>
(private until shared; export to PDF or PowerPoint from its Share menu). This folder is a copy.

| Slide | What it shows |
| --- | --- |
| `balance` | Balanced positions: share of the training loss per angle (YES), the same chart as the 27 Oct deck. Image asset `025e2ff1844f941400f738b4babf7e69` = `figures/fig_balance.png`. |
| `calibration` | The 10 s calibration, presented as an exploratory attempt, with three references. |
| `summary` | The final model (N0 + 10 s calibration): every design choice, and MAE, R², VAF for N0, A0 and both calibrated. |

Numbers: `EXPERIMENTS/normalization/calibration_summary.md` and `calibration_summary_A0_active_trainmax.md`
(6 people × 3 seeds, scored from 30 s to the end of each trial, median over seeds then angles).

References on the calibration slide:

- Lloyd DG, Besier TF (2003). An EMG-driven musculoskeletal model to estimate muscle forces and knee joint moments in vivo. J Biomech 36(6):765–776.
- Burden A (2010). How should we normalize electromyograms obtained from healthy participants? J Electromyogr Kinesiol 20(6):1023–1035. doi:10.1016/j.jelekin.2010.07.004
- Quesada L, Verdel D, Bruneau O, Berret B, Amorim M-A, Vignais N (2026). EMG-to-torque models for exoskeleton assistance: a framework for the evaluation of in situ calibration. Int J Robot Res. doi:10.1177/02783649251414884
