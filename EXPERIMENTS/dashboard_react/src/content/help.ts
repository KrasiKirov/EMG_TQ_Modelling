// Text of the "How to use this page" panel on the results page (study v2). The pre-registered limitations are added
// below it from confirm.json.

export interface HelpSection { title: string; items: [string, string][] }

export const HELP_SECTIONS: HelpSection[] = [
  {
    title: 'Finding your way',
    items: [
      ['Study v2', 'The study pre-registered on 18 Nov 2026 (EXPERIMENTS/protocol/study_v2.json). Each history length and overlap (a "cell") was tuned on session 1 only; its winner, A0 (the setting chosen across cells) and three fixed reference settings were then run on session 2 once.'],
      ['Settings', 'A0: the chosen setting. Cell winners: the tuned network of each window length and overlap, or of each carried-state chunk length. A1: lagged ridge regression. A2: last semester\'s network (8 units × 2 layers) trained with study v2\'s pipeline. A3: A0 trained on total torque.'],
      ['Filters', 'Each filter value jumps to a setting that was run. If none has exactly that combination, the nearest one is shown and a line under the filters says what else changed.'],
      ['Link', 'The address bar keeps the selected setting (#setting=<id>). Copy it to send someone the same view; the id is also shown after the title.'],
      ['Sections', 'Summary (one row per person) → worst positions (measured vs predicted torque) → MAE and R² at every position (a box over seeds) → how the setting was chosen (session-1 validation) → peak torque → reproduce and run details (collapsed: training time, how to re-run the setting, the computer the runs used).'],
    ],
  },
  {
    title: 'What is measured',
    items: [
      ['Session 1 → session 2', 'Each person\'s model learns from session 1 (the lab\'s "test") and predicts all of session 2 ("retest"), which it never saw and which was not used to choose anything.'],
      ['Validation', 'The last 20 % of each session-1 trial, after a gap. Used to tune and choose the settings; labelled "validation (session 1)" wherever it is shown.'],
      ['Positions', 'Fixed ankle angles per person, in radians. Negative = plantarflexed.'],
      ['Active torque', 'Measured torque minus the passive torque from the tested session\'s own relaxed holds (YES: lab worksheet; IES has no relaxed holds and uses the experiment-sheet values). Total torque = the measured torque.'],
      ['Scoring start', 'Every run is scored from 1 s after its trial\'s start, so a long window and a short one are scored on the same samples. The waveform curves show the whole trial.'],
    ],
  },
  {
    title: 'Reading the numbers',
    items: [
      ['nMAE', 'The pre-registered metric: at each position the mean absolute error divided by the 99th percentile of the person\'s measured active torque on session 2 (A3, which predicts total torque, uses the same scale), then the mean over positions. Lets people with small and large torques be compared.'],
      ['MAE (Nm)', 'The same mean absolute error without the division.'],
      ['R² per position', 'Within one trial (one angle): how well the torque changes are followed. 1 = perfect, 0 = no better than the trial\'s mean, below 0 = worse than that.'],
      ['Seeds', 'A seed sets the network\'s random starting weights. Cell winners have seeds 10–12; A0, A2 and A3 seeds 10–19; A1 has one run (seed 10). Each person\'s value is the median over seeds.'],
      ['[lo, hi]', 'An interval for that median over seeds (about 95 %; none with 1 seed). It shows training randomness only, not how much results would change with other trials or other people.'],
      ['Combined', 'Under the per-person rows: exp of the mean over people of log nMAE, the study\'s own way of combining people (the same as its selection scores). The Median row is the plain median of the 6 people.'],
    ],
  },
];
