// All slide text for the v2 deck, plain English, "we" voice. Figure numbers: see FIGNUM (from the figure agents).
const { N, CP, b, t, f } = require('./numbers.js');
const FN = require('./fignum.js');

N.slideRef = { passive: '9–11, 13', drift: '12–13', table: '9–10, 13', agc: '14–15', scale: '16–17' };

// ── problem slide
N.problemTitle = 'Last semester: the scores dropped at the toes-down angles';
N.problemNotes = 'Each bar is one ankle angle, most toes-down on the left. This is the old pipeline, learned from session 1 and tested on session 2, scored the way we scored it last semester. For YES, JM and EG the most toes-down angles drop sharply, YES even below zero. IES and HM drop less and CH hardly at all. JM also has a bad bar at −0.08 rad; that one turned out to be a bug in the passive table. These are re-runs of the old code (3 runs averaged for YES, JM, EG; 1 run for the others), so the numbers differ slightly from last semester’s single run.';

// ── real data slides
N.strongBullets = FN.strongBullets;
N.strongNotes = FN.strongNotes;
N.toesBullets = FN.toesBullets;
N.toesNotes = FN.toesNotes;

// ── old table
N.tableBullets = FN.tableBullets;
N.tableNotes = FN.tableNotes;
N.wrongBullets = FN.wrongBullets;
N.wrongNotes = FN.wrongNotes;
N.jmBullets = FN.jmBullets;
N.jmNotes = FN.jmNotes;
N.driftBullets = FN.driftBullets;
N.driftNotes = FN.driftNotes;
N.fixBullets = [
  [b('What “inside the trial” means: '), t('we do not use any separate recording. We look inside the pushing trial itself for the moments when the person is relaxed.')],
  [b('Step 1: '), t('the muscle signals tell us when all 4 muscles are switched off (average activity below 3 % of each muscle’s range, for at least 0.5 s). We never use the torque to decide this.')],
  [b('Step 2: '), t('in those moments nobody is pushing, so the sensor reads only passive torque. We take the middle value (median) of each relaxed stretch and draw a smooth line through them (smoothed over about 8 s, so it cannot follow a 1–2 s push).')],
  [b('Step 3: '), t('voluntary torque = measured torque − that line, sample by sample.')],
  [b('Fixes all three passive problems: '), t('right level (measured in the same trial), follows the drift, and no table, so no bad entries. It also works for IES, who has no separate relaxed recordings at all.')],
  [b('Used for: '), t('the training answers (session used for learning only) and the shared answer key for grading. The model itself never sees this line; it only gets EMG + angle.')],
];
N.fixNotes = 'This is the fix for the first three problems. Instead of a separate recording, we measure the passive level inside the same trial. The muscle signals tell us when the person is relaxed. At those moments the sensor reads only passive torque, so we take the middle value of each relaxed stretch and draw a smooth line through them. Voluntary torque is the measured torque minus that line. The line is very smooth, about 8 seconds, so it cannot copy the pushes, which last 1 to 2 seconds. This gives the right level, follows the drift, needs no table, and works for IES, who has no separate relaxed recordings.';

// ── EMG
N.emgWhyBullets = FN.emgWhyBullets;
N.emgCauseBullets = FN.emgCauseBullets;
N.emgNotes = FN.emgNotes;
N.agcBullets = FN.agcBullets;
N.agcNotes = FN.agcNotes;

// ── loss / scaling
N.lossBullets = FN.lossBullets;
N.lossNotes = FN.lossNotes;
N.scaleBullets = FN.scaleBullets;
N.scaleNotes = FN.scaleNotes;

// ── model
N.lstmBullets = FN.lstmBullets;
N.lstmNotes = FN.lstmNotes;
N.expertsAchieve = FN.expertsAchieve;

// ── ablation
Object.assign(N, FN.ablation);
N.ablationBullets = require('./ablation_text.js').bullets;
N.ablationNotes = require('./ablation_text.js').notes;
N.scaleBullets[3] = [b('What it achieves: '), t(require('./ablation_text.js').scale)];

// ── checking
N.keyBullets = FN.keyBullets;
N.keyNotes = FN.keyNotes;
N.phantomBullets = FN.phantomBullets;
N.phantomNotes = FN.phantomNotes;
N.runsBullets = FN.runsBullets;
N.runsNotes = FN.runsNotes;

// ── results
N.summaryBullets = FN.summaryBullets;
N.summaryNotes = FN.summaryNotes;
N.errorNotes = FN.errorNotes;
N.susBullets = FN.susBullets;
N.susBox = FN.susBox;
N.susNotes = FN.susNotes;
N.inspectFoot = FN.inspectFoot;
N.inspectNotes = FN.inspectNotes;
N.hardestNotes1 = FN.hardestNotes1;
N.hardestNotes2 = FN.hardestNotes2;

// ── wrap-up
N.limitBullets = [
  'Only 6 people, and both sessions were in the same visit with the electrodes left on. A session on another day would be a harder test.',
  'YES at −0.49 rad is still the weakest angle (R² 0.34 learned from session 1, 0.70 learned from session 2): YES only pushes about 1 Nm there, so an error of 0.3 Nm is already a third of the push.',
  'The 3 extra people are not completely fresh: our first report this semester already looked at them, and CH’s push-direction flip shaped the auto-gain on/off rule.',
  'We picked the final version after seeing the 3 extra people, so for that choice they are not a fully independent check.',
  'The answer key uses the same “relaxed moments” idea as the new training answers. Check: the version from our first report, whose answers are built differently, scores the same 0.925 on this key, so the key is not just rewarding the new pipeline.',
  'The key assumes that when the EMG is quiet the person really is not pushing. When someone is almost never relaxed in a trial, the line is less certain (JM session 2 at −0.50 rad: relaxed only 0.9 % of the time).',
  'The EMG smoothing looks about 0.25 s into the future (it runs forwards and backwards). Fine for analysis afterwards, not yet for real-time use.',
];
N.nextBullets = [
  'Combine the new pipeline with a longer window (0.3–0.4 s): the 0.4 s window looked promising (0.932 vs 0.925) but we have not tested it together with the new passive method yet.',
  'Record a session on a different day (or a new person) to confirm everything independently.',
  'If possible, record a short relaxed moment before and after each pushing trial, and keep the push direction the same at each angle in both sessions.',
  'Make the EMG smoothing one-directional so the model could run live.',
  'Average the 3 training runs into one model: a small free gain (+0.006 R²).',
  'Move the new pipeline into the main training script and save the trained models.',
];
N.limitNotes = 'The limits: only six people and both sessions in the same visit; YES at the most toes-down angle is still the weakest; the final choice was made after seeing the extra people; the answer key relies on the relaxed moments; and the EMG smoothing is not real-time yet. Next steps: combine the new pipeline with a 0.3 to 0.4 second window, which looked promising; record a session on another day; and move this into the main training script.';

N.sources = FN.sources;
N.takeaways = [
  'The toes-down failure came from the data (passive torque, EMG size, how the fit weights the angles), not from the network.',
  'Each problem has a concrete fix: passive level measured inside the trial, auto-gain on the EMG, every angle weighted equally in the fit.',
  `Results, all 6 people: toes-down R² ${f(N.mainOld.pf2)} → ${f(N.mainNew.pf2)} on the main 3 and ${f(N.extraOld.pf2)} → ${f(N.extraNew.pf2)} on the extra 3; error at the toes-down angles ${f(N.mainOld.err)} → ${f(N.mainNew.err)} Nm (main 3) and ${f(N.extraOld.err)} → ${f(N.extraNew.err)} Nm (extra 3).`,
  'Checked fairly: one shared answer key, both directions, 3 runs, 3 extra people, code review.',
];

// ── backup: what else we tried (plain English)
const TH = { bold: true, color: 'FFFFFF', fill: { color: '1F2A44' }, valign: 'middle' };
const row = (a, bb, c, keep) => [
  { text: a, options: { bold: true, color: '1F2A44', fill: { color: 'F2F1EE' } } },
  { text: bb, options: { fill: { color: 'FFFFFF' } } },
  { text: c, options: { bold: true, color: keep ? 'EB6834' : '52514E', fill: { color: 'FFFFFF' } } },
];
const head = [{ text: 'What we tried', options: TH }, { text: 'What happened (average R² on YES, JM, EG, both directions, 3 runs; our pipeline: 0.925)', options: TH }, { text: 'Verdict', options: TH }];
N.tried = [
  [head,
    row('Longer window: the model looks at the last 0.4 s instead of 0.2 s', 'Slightly better overall (0.932 vs 0.925), but only at the strong angles; the toes-down angles stayed the same (0.84).', 'Promising: next we combine it with the new pipeline (0.3–0.4 s)', true),
    row('Passive value from the relaxed hold recorded just before each trial', 'Tie (0.921). The hold is still a separate moment, before any pushing.', 'Not kept: the in-trial level is simpler', false),
    row('A passive value that slowly relaxes during the trial (fitted drift curve)', 'Tie (0.921).', 'Not kept: the in-trial line already follows the drift', false),
    row('EMG divided by the strongest push (MVC) instead of auto-gain', 'Worse (0.887; per angle 0.904). The MVC was recorded once at the start of the visit, but the EMG size changes later.', 'Not kept', false),
    row('Extra weight on the 2 most toes-down angles (×3 and ×10)', '×3: tie (0.922). ×10: worse (0.916) and the toes-down angles did not improve (0.83).', 'Not kept: dividing by the swing already balances the angles', false),
    row('Randomly changing the EMG size during fitting, so the model gets used to size changes', 'Helps when there is no auto-gain (0.826 → 0.857), adds nothing on top of auto-gain (0.922).', 'Not kept: auto-gain does the same job', false),
    row('Fit on the other 5 people first, then refit on this person’s data', 'Worse (0.885; YES drops to 0.71 in one direction). People differ too much (angles, strength, push direction).', 'Not kept', false),
  ],
  [head,
    row('Removing the relaxed periods from the fit (or giving them ×0.2 weight)', 'Same R² (0.924) but the model invents more torque while the person is relaxed (phantom 0.25 → 0.36 Nm; tested on the earlier version from our first report). Relaxed periods are useful, not noise.', 'Not kept', false),
    row('Scaling each session by its own overall EMG level', '0.775 and breaks JM. It also needs the whole test session in advance, which is not allowed.', 'Not allowed', false),
    row('A straight-line model (ridge regression) instead of the network', '0.624; with auto-gain 0.863. The network adds something, but auto-gain matters more.', 'Not kept', false),
    row('The old small network inside the new pipeline', '0.921 vs 0.930 (1 run each), more phantom torque (0.32 vs 0.21 Nm) and occasional overshoot spikes.', 'Simpler fallback', true),
    row('Other common network designs (GRU, CNN, plain network, bigger LSTM) on the same fixed data — our first report', 'None better everywhere; some much worse at single angles. The network type was not the problem.', 'Not kept', false),
    row('Leak test: replace the measured angle by the set angle (the footplate bends slightly when pushing) — our first report', 'No change (e.g. YES +0.03 rad: 0.99 → 0.99). The model does not cheat through the angle sensor.', 'Check passed', true),
    row('Using session 2’s own torque to re-zero each test trial — our first report', 'Not better overall, and it uses the answer we are trying to predict.', 'Rejected (cheating)', false),
  ],
];

module.exports = N;
