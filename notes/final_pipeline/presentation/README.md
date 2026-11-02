# Presentation files

## Supervisor version (v2, use this one)

| file | what it is |
|---|---|
| `EMG_torque_supervisor_v2.pptx` | The deck in plain English. Speaker notes on every slide. Slides 45–50 are backup slides for questions. |
| `questions_and_answers_v2.md` | The questions the supervisor may ask, with the answer and the slide that answers each one. |
| `sources_v2.md` | Every source used, with its DOI or PubMed link and one sentence on what it shows. All checked by a second reviewer. |
| `figures_v2/` | Every figure in the deck, made from the real data. Also contains the full-trial inspection plots for the other direction (`inspect_<person>_S2toS1_old/new.png`). |
| `src_v2/` | The scripts that made the figures (`F1_…`–`F10_…`) and the deck (`build_deck_v2.js` + text files). |

### Where the numbers come from

| What | Source |
|---|---|
| Main 3 (YES, JM, EG) | `ML/research/results/screen`: 3 training runs, averaged. |
| Extra 3 (IES, CH, HM) | `ML/research/results/confirm`: 1 training run. |
| "New" | config `E22_restanchor_final` (CH: `E32_restanchor_noagc_moe`). This is identical to `FINAL`. |
| "Old" | config `E01_old`. |
| "What each change contributes" slide | `ML/research/results/ablation` (new configs `E33_final_noscale`, `E34_final_oldpassive`, `E35_final_lstm8`, plus `E32`), 1 run, all 6 people, both directions. |

All of them are graded with the shared answer key (`ML/research/truth.py`).

## Older version (v1)

`EMG_torque_old_vs_new.pptx`, `questions_and_answers.md`, `figures/` and `src/` are the first, more technical deck.
