# Explainer video v4: windows, memory and shuffling (a check of our method)

About 8 minutes. Same format as v2/v3: Kokoro `af_sarah`, speed 1.0, 0.6 s gap, 1080p30, the v2 palette, no code on
screen, every number tagged in `SRC:`. Written for the supervisors' question: "torque at time t needs information
from earlier moments, so why are the windows cut up and shuffled?" Plain claims only, backed by checks on real data.

Model shown: the final model before calibration (N0: balanced LSTM, total torque, EMG ÷ session-1 maximum, 0.2 s).
Real examples: EG, session 2, −0.038 rad (V0 = N0 retrained for EG seed 0 with its weights saved). Facts:
`data4/facts_w01.json` (errors by push phase, lag), `facts_w02.json` (inside the LSTM, order test),
`facts_w03.json` (shuffling on vs off), `facts_v01.json` (windows, counts; from v3).

---

## c01 | The question

### c01_b01
VIS: Title card "Windows, memory and shuffling: a check of our method". Then a quote card: "Torque at time t needs information from earlier moments. So why cut the recording into windows, and shuffle them?"
SAY: Here is the question. The torque at any moment depends on what the muscles did a little earlier. So why do we cut the recordings into short windows, and then shuffle them during training? Doesn't that throw the timing away?
SRC: supervisors' question (user, 3 Nov 2026).

### c01_b02
VIS: Three short answers appear one under the other: "1 · The history is inside each window." "2 · Shuffling changes the order of the examples, not what is in them." "3 · We checked it on real data, four ways." A footer: "and one question that is still open: is 0.2 s of history enough?"
SAY: The short answer has three parts. The history is inside each window. Shuffling only changes the order of the examples, never what is inside them. And we checked it on real data, four different ways. One question is still open, and we will end on it: is 0.2 seconds of history enough?
SRC: —

## c02 | Two ways to give a model memory

### c02_b01
VIS: Left panel "a muscle model (CEINMS, the earlier ankle project)": a chain of boxes for the activation equation, u(t) = α·e(t − d) − β₁·u(t − 1) − β₂·u(t − 2), with arrows feeding u(t − 1) and u(t − 2) back in. A long timeline below with one state passed from sample to sample, start to end of the recording. Label: "the state is carried through the whole recording → it must run in time order".
SAY: There are two ways to give a model memory. The first is the way muscle models do it, like the CEINMS model from the earlier ankle project. Each new value of muscle activation is computed from the previous ones. So the model carries a state from sample to sample, through the whole recording. A model like that must run in time order, and shuffling its samples would make no sense.
SRC: Team 11 BIEN470 D2 report, activation dynamics u(t) = αe(t − d) − β₁u(t − 1) − β₂u(t − 2) (p. 6), CEINMS (pp. 8–10).

### c02_b02
VIS: Right panel "our LSTM": the recording with a 20-sample box; an arrow from the box to "torque at its last sample". Under it: "each prediction uses only the last 0.2 s · the memory starts empty for every window". Two tags under the panels: "carried state (like an IIR filter)" vs "fixed look-back (like an FIR filter / impulse response)".
SAY: The second way is ours. Each prediction looks back over a fixed stretch: the last twenty samples, 0.2 seconds, and nothing before that. The LSTM does have a memory, but only inside that window. It starts empty for every window. In signal-processing terms, the first kind is like an I I R filter, with a state carried forward. Ours is like an F I R filter, an impulse response with a fixed length.
SRC: stateless Keras LSTM (memory reset per sample); window 20 samples (splits.windows).

### c02_b03
VIS: A regression table builds up: each row = "20 past samples of the 5 signals" → "torque at t". Rows are then shuffled on screen; a note: "least squares on these rows gives the same answer in any row order". Then the same rows labelled "training examples for the LSTM".
SAY: An impulse-response model is fitted the same way. Each row of its data holds the last few samples of input, and the torque at that moment. Shuffle the rows, and least squares gives exactly the same answer, because the history lives inside each row, not in the order of the rows. Our training examples are those rows.
SRC: ordinary least squares is invariant to row order.

## c03 | Inside the LSTM

### c03_b01
VIS: One real window from EG's session-2 trial, at the start of a push: 20 columns × 5 rows (GM, GL, soleus, TA, angle) as a coloured grid, oldest on the left. Above it, the torque around that moment, with the window's 0.2 s highlighted and a dot at its last sample: "[[w02_torque]] Nm".
SAY: Let's open the trained model, and follow one real window through it. This one is from person EG, in session 2, right at the start of a push. Twenty time steps, five signals each, oldest on the left. The question it answers: what is the torque at the last step?
SRC: facts_w02 (window at t = [[w02_time]] s, torque [[w02_torque]] Nm); w02 win.

### c03_b02
VIS: The LSTM drawn as a small cell: a box with "memory (8 numbers)" and three gates: "forget", "write", "read". The 20 columns enter one at a time. For each step, the real values light up: the forget-gate average, and the 8 memory numbers as bars that change height (from w02 l1_c). A small step counter "step 1 … 20".
SAY: The LSTM reads the window one step at a time, oldest first. It keeps a memory of eight numbers. At every step, three gates decide what happens to it. A forget gate decides how much of the old memory to keep. A write gate decides what to add from the new step. And a read gate decides what to pass on. These bars are the real memory values of our model, step by step through this window.
SRC: Keras LSTM equations; facts_w02 units 8; w02 l1_c, l1_f, l1_i, l1_o.

### c03_b03
VIS: After step 20, the memory of the second layer becomes one number through the output: "× the angle's swing" → "prediction: [[w02_pred]] Nm" next to the measured "[[w02_torque]] Nm". Then all memory bars drop to zero: "next window: memory starts empty".
SAY: After the twentieth step, the memory is boiled down into one number, the prediction, here [[w02_pred]] newton metres, against [[w02_torque]] measured. Then the memory is wiped. The next window starts from an empty memory. So everything the model uses about the past, for this prediction, comes from inside this one window.
SRC: facts_w02 window_prediction_Nm, window_torque_Nm; stateless LSTM.

### c03_b04
VIS: A check card: "We recomputed the network by hand, with its saved weights, for [[w02_checked]] windows: largest difference from the model's own output: [[w02_hand]]". Subtitle: "nothing hidden: these are the only steps".
SAY: To be sure nothing else is going on, we recomputed the network by hand, from its saved weights, for [[w02_checked]] windows. The largest difference from the model's own output was [[w02_hand]]. Those twenty steps are all there is.
SRC: facts_w02 hand_vs_model_max_abs, hand_checked_windows.

## c04 | Check 1: the order of the windows

### c04_b01
VIS: EG's session-2 trial: the measured torque (white) and the model's prediction made window by window in time order (green). Then the windows are drawn as cards, shuffled, predicted in that random order, and put back by their time stamps (dots landing on the curve). The two predictions overlap exactly. Badge: "largest difference over [[w02_windows]] windows: [[w02_order]] Nm".
SAY: First check: does the order of the windows matter to the model? We predicted this whole trial window by window, in time order. Then we shuffled the windows, predicted them in that random order, and put each prediction back at its own time. The two curves are identical. Over [[w02_windows]] windows, the largest difference is [[w02_order]] newton metres. Each window is a complete example on its own.
SRC: facts_w02 order_test_windows, order_test_max_abs_Nm.

## c05 | Check 2: training with and without shuffling

### c05_b01
VIS: Two strips of batches. "Shuffled": each batch of 8 holds windows from different angles and moments (colours mixed). "Not shuffled": the batches walk through trial 1, then trial 2, … (long runs of one colour). A drifting arrow under the second: "the model keeps leaning towards the last trial it saw".
SAY: Shuffling matters during training, but for a different reason. Each batch of eight windows nudges the model a little. Shuffled, every batch mixes angles and moments. Not shuffled, training walks through one trial after another, and the model keeps leaning towards the last one it saw. That is about balanced learning, not about time.
SRC: model.fit(shuffle=True) vs S0_noshuffle (shuffle=False, windows in stored order).

### c05_b02
VIS: Results table, the final model trained twice, everything else identical (6 people × 3 seeds): rows "shuffled" and "not shuffled"; columns "toes-down MAE", "R²", "other MAE", "R²", and the same after the 10 s calibration. Values [[w03_*]]. Footer: epochs run and kept.
SAY: [[w03_say]]
SRC: facts_w03 shuffle_on, shuffle_off.

## c06 | Check 3: timing and the end of a push

### c06_b01
VIS: EG's trial zoomed on two pushes: measured torque and prediction. Phases coloured: onset (first 0.3 s of a push), plateau, release (0.5 s after it ends), rest. A lag meter: "best alignment of prediction and torque: median [[w01_lag]] s · [[w01_within]] % of trials within 50 ms".
SAY: Second check: if the window were too short to see the cause of the torque, the prediction would come out late, or it would miss the end of each push, when the muscle relaxes but the torque is still fading. So we measured both. Across [[w01_trials]] trials, the best alignment between prediction and torque is a lag of [[w01_lag]] seconds, and [[w01_within]] percent of trials are within fifty milliseconds. The prediction is not late.
SRC: facts_w01 lag_s, trials.

### c06_b02
VIS: Four bars: error by phase after removing each trial's offset: rest [[w01_rest]], onset [[w01_onset]], plateau [[w01_plateau]], release [[w01_release]] Nm. The release bar is not the tallest; the onset bar is.
SAY: And here is the error in each phase of a push, with each trial's offset removed. The release is not where the model struggles: [[w01_release]] newton metres, against [[w01_plateau]] on the plateau. The largest error is at the onset, [[w01_onset]], where the torque rises fastest, and a small timing slip costs the most.
SRC: facts_w01 shape_mae_median.

### c06_b03
VIS: Caveat card: "The EMG envelope is smoothed with a zero-phase filter: it spreads each moment over about ±0.4 s, past and future. Fine offline; a live device would need a causal filter."
SAY: One honest caveat. Before the windows are cut, the EMG is smoothed with a zero-phase filter. It spreads each moment over about 0.4 seconds, both before and after. So each window already carries a little more history than 0.2 seconds, and a little of the future. That is fine offline. A live device would need a filter that only looks back.
SRC: v2 facts_x02 imp90 ±0.376 s (2 Hz zero-phase envelope).

## c07 | What is still open

### c07_b01
VIS: Two cards. Left "settled": "the history is inside each window" · "window order does not change predictions" · "shuffling = balanced training" · "no lag". Right "open": "is 0.2 s the best look-back?" with "literature: EMG-to-force dependence levels off after about 0.3–0.4 s (Wang et al. 2024)" and "next: the same model with 0.5 s and 1 s windows".
SAY: So what is settled, and what is not? Settled: the history is inside each window, the order of the windows does not change the predictions, shuffling only balances the training, and the predictions are not late. Open: whether 0.2 seconds is the best look-back. In a hand-wrist study, the error kept falling up to about 0.3 to 0.4 seconds of history. So the fair next step is to train the same model with 0.5 and 1 second windows, and compare.
SRC: Wang H et al. 2024, Sensors 24(22):7301 (error levelled off after ~293–391 ms; LSTM best 488 ms).

## c08 | Recap

### c08_b01
VIS: Four recap lines with icons.
SAY: To recap. Each window is a complete example: the last 0.2 seconds of muscle activity, and the torque at its end. The memory lives inside the window. Shuffling changes the order of the examples, not their contents, and only keeps the training balanced. And the open question is the length of that look-back, which we can test directly.
SRC: —
