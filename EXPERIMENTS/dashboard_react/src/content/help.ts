// Text of the "How to use this page" panel on the Results page.

export interface HelpSection { title: string; items: [string, string][] }

export const HELP_SECTIONS: HelpSection[] = [
  {
    title: 'Finding your way',
    items: [
      ['Study', '"Experiments" is the window study (window length, overlap, torque, LSTM state). "Hyperparameter tuning" lists the 6 finalist networks and how one was chosen.'],
      ['Filters', 'Each filter value jumps to a setting that was actually run. If no run has exactly that combination, the nearest one is shown and a line under the filters says what else changed.'],
      ['Steps', 'Only settings that feed the network every 10 ms sample are shown. Settings that used one sample every 50 ms are left out.'],
      ['Link', 'The address bar keeps the selected setting (#setting=<id>). Copy it to send someone the same view; the id is also shown after the title.'],
      ['Sections', 'Summary (one row per person) → worst positions (measured vs predicted torque) → MAE and R² at every position (a box over seeds) → peak torque → training time → how to re-run the setting → the computer the runs used.'],
    ],
  },
  {
    title: 'What is measured',
    items: [
      ['Session 1 → session 2', 'Each person\'s model learns from session 1 (the lab\'s "test") and predicts all of session 2 ("retest"), which it never saw.'],
      ['Positions', '8 fixed ankle angles per person, in radians. Negative = plantarflexed. "One of the 2 most plantarflexed" marks each person\'s 2 most negative angles, where the model struggles most.'],
      ['Active torque', 'Measured torque minus the passive torque from the tested session\'s own relaxed holds (YES: lab worksheet; IES has no relaxed holds and uses the experiment-sheet values). Total torque = the measured torque.'],
    ],
  },
  {
    title: 'Reading the numbers',
    items: [
      ['MAE (Nm)', 'Mean absolute error at each position, then the mean over the 8 positions.'],
      ['MAE ÷ peak', 'MAE divided by the person\'s largest measured active torque. Needed to compare people: their peak torque runs from about 9 to 67 Nm.'],
      ['MAE vs reference', 'This setting\'s MAE ÷ the reference\'s MAE for the same person, on the seeds both have. Below 1 = lower error. The reference is the 200 ms, full-overlap setting of the same torque; in the tuning study, last semester\'s network size retrained on this pipeline. The median row also says how many people have a lower error.'],
      ['R² per position', 'Within one trial (one angle): how well the torque changes are followed. 1 = perfect, 0 = no better than the trial\'s mean, below 0 = worse than that. The summary shows the median of the 8 positions and, separately, the worst one.'],
      ['Seeds', 'A seed sets the network\'s random starting weights. Each person\'s value is the median over seeds. The same seed with the same software reproduces a run exactly.'],
      ['[lo, hi]', 'An interval for that median over seeds (about 95 %; 94 % with 5 seeds; none with 1 seed). It shows training randomness only, not how much results would change with other trials or other people.'],
      ['Caution', 'These settings were compared on session 2, the data they are tested on. Treat differences between settings as leads to confirm, not conclusions.'],
    ],
  },
];
