# Check-up with supervisor: where we are and proposed next steps

**Status in one sentence:** we did a first exploration of why the model fails at the toes-down angles and of possible fixes. The first results look promising but are **not validated yet**. The steps below are what we want to test properly, with the literature behind each idea.

Status labels:
- **Not started:** genuinely new.
- **First look:** tried once, quickly. The result is preliminary and must be confirmed.
- **First look, no gain:** tried quickly, did not help in our data. Worth discussing whether to drop it.

---

## A. Make sure our evaluation is trustworthy (top priority)

| # | Step | Why | Literature (paraphrased; check the paper for the exact wording) | Status |
|---|---|---|---|---|
| A1 | **Define voluntary torque carefully.** Measured torque = passive + voluntary, so the passive part must be estimated. Compare a passive value from separate relaxed recordings with one measured inside the pushing trial (relaxed moments found from EMG). | The passive estimate defines both the training answers and the grading answers. If it is wrong, the scores are wrong. | Passive ankle torque falls over time while held stretched (stress relaxation): Duong et al. 2001, *Clin Biomech* 16(7), doi:10.1016/S0268-0033(01)00043-2; McNair et al. 2001, *Med Sci Sports Exerc* 33(3), doi:10.1097/00005768-200103000-00003. It is also lower after contractions: Kay & Blazevich 2009, *J Appl Physiol* 107(4), doi:10.1152/japplphysiol.00281.2009. | First look |
| A2 | **Check that the conclusions do not depend on how "relaxed" is defined.** Re-score with other thresholds and smoothing. | Rules out the result being an artefact of our own choice. | General point: R²-type scores depend on how much the signal varies, so compare with care: Gupta et al. 2009, *J Hydrol* 377, doi:10.1016/j.jhydrol.2009.08.003. | Not started |
| A3 | **Score with an offset-free metric** (push size relative to the rest just before each push) as well as R² and the error in Nm. | Separates "wrong passive level" from "wrong push size". | Same as A2. Also Bland & Altman 1986, *Lancet*, doi:10.1016/S0140-6736(86)90837-8 (agreement depends on the range of values). | Not started |
| A4 | **Test both directions** (learn session 1 → test session 2, and the reverse), and **average 3 training runs** instead of using one. | One direction or one run can be lucky. | Standard practice; early stopping on held-out data: Prechelt 1998, *Neural Netw* 11(4), doi:10.1016/S0893-6080(98)00010-0. | First look |
| A5 | **Gravity correction:** separate the weight of the foot and footplate from the tissue's passive torque. | The "relaxed" reading includes both. | Anderson et al. 2010, *J Biomech* 43(6), doi:10.1016/j.jbiomech.2009.11.036. | Not started |

## B. Model and processing ideas to test properly

| # | Step | Why | Literature | Status |
|---|---|---|---|---|
| B1 | **Rescale the EMG automatically** as the signal changes: a running estimate of each muscle's quiet and strong levels ("auto-gain"). | For the same push, the EMG size differed between our two sessions (e.g. about half for JM at one angle). | EMG size depends on electrode contact, sweat and temperature, not only effort: De Luca 1997, *J Appl Biomech* 13(2), doi:10.1123/jab.13.2.135; Abdoli-Eramaki et al. 2012, doi:10.1016/j.jelekin.2012.04.009. Running or sliding rescaling has been used: Tanaka et al. 2022, *Sensors* 22(13), doi:10.3390/s22135005; Koller et al. 2015, *J NeuroEng Rehabil* 12:97, doi:10.1186/s12984-015-0086-5. | First look |
| B2 | **Give every angle the same weight in the fit** (divide by each angle's typical swing = weighted least squares). | The toes-down angles carry about 1 % of the fit otherwise. | Inverse-variance weighting of each task's error: Kendall et al. 2018, *CVPR*, doi:10.1109/CVPR.2018.00781. | First look |
| B3 | **Per-angle corrections on top of one shared model** ("mixture of experts"). | The EMG–torque link changes with muscle length. | Jacobs et al. 1991, *Neural Comput* 3(1), doi:10.1162/neco.1991.3.1.79. | First look |
| B4 | **Longer input window** (0.3–0.4 s instead of 0.2 s), combined with B1–B3. | More history may help. | Longer windows (~300–600 ms) used for EMG-based estimation: Wang H et al. 2024, *Sensors* 24:7301, doi:10.3390/s24227301. | Not started (in this combination) |
| B5 | **Short calibration at the start of each new session:** a few seconds of known pushes to correct the model. | Common fix for session-to-session drift. | Vidovic et al. 2016, *IEEE TNSRE*, doi:10.1109/tnsre.2015.2492619; Lloyd & Besier 2003, *J Biomech* 36(6), doi:10.1016/S0021-9290(03)00010-1. | First look (only helped with 20–30 s of calibration) |
| B6 | **One-directional (causal) EMG filters**, so the model could run live. | The current smoothing looks ~0.25 s into the future. | (General signal-processing point; no specific source needed.) | Not started |
| B7 | **Average the 3 trained models** into one (ensemble). | Small, free gain. | (Standard practice.) | First look |

## C. Ideas that did not help in our first look (discuss whether to drop them)

| Idea | Literature that motivated it | What we saw |
|---|---|---|
| Divide EMG by the max push (MVC) instead of B1 | Burden 2010, doi:10.1016/j.jelekin.2010.07.004 | Slightly worse. Our MVC was recorded once at the start, but the EMG size changed later. |
| Pre-train on other people, then refit on this person | Zhang L et al. 2022, *IEEE TNSRE* 30, doi:10.1109/TNSRE.2022.3156786 | Worse: our 6 people differ a lot (angles, strength, push direction). |
| Random EMG-size changes during training (augmentation) | Tsinganos et al. 2020, *Sensors* 20:4892, doi:10.3390/s20174892 | Helped only without B1; nothing extra with it. |
| Re-scale each session using its whole EMG recording | Du et al. 2017, *Sensors* 17:458, doi:10.3390/s17030458 | Mixed (good for 2 people, bad for JM), and it needs the whole session in advance. |
| Extra weight (×3, ×10) on the toes-down angles | — | No gain on top of B2. |
| Straight-line (ridge) model instead of a network | Ziai & Menon 2011, doi:10.1186/1743-0003-8-56; Clancy et al. 2012, doi:10.1109/TBME.2011.2170423 | Clearly worse, but a useful baseline. |

## D. Recording protocol (needs the lab; probably the most valuable)

| # | Step | Why | Literature |
|---|---|---|---|
| D1 | **Record a session on a different day** (electrodes re-applied), and ideally new people. | Our two sessions were in the same visit with the electrodes left on, the easiest case. Published EMG models degrade a lot on later days. | Ziai & Menon 2011 (wrist R² ~0.96 same session → ~0.46–0.59 the next day); Schulte et al. 2022, *Front Robot AI* 9:869476, doi:10.3389/frobt.2022.869476 (error ~6 % → ~10–11 % on later days). |
| D2 | **A short relaxed hold before AND after each pushing trial.** | Passive torque drifts and changes after contractions. One hold before is not enough. | Duong 2001; McNair 2001; Kay & Blazevich 2009 (as in A1). |
| D3 | **Keep the push direction the same at each angle in both sessions** (one participant pulled at one angle and pushed at it in the other session). | Otherwise the task itself changes between sessions. | — |
| D4 | **Ultrasound of the calf during pushes** (if available). | Passive torque during a contraction may differ from rest, and this is the only way to check. | MacIntosh & MacNaughton 2005, *J Appl Physiol* 98; Hoffman et al. 2012, *J Appl Physiol* 113. |

---

**Questions for the supervisor:**
1. Is defining voluntary torque from the relaxed moments inside the same trial (A1) acceptable to you?
2. Can we get a different-day session (D1)? And can we change the protocol (D2, D3)?
3. Priority: validation first (A2, A3, D1), or more modelling (B4–B6)?
