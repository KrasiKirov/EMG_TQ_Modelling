// Builds notes/final_pipeline/presentation/EMG_torque_supervisor_v2.pptx
// Plain-English deck for the supervisor: what we did, why, results for all 6 people, sources.
const path = require('path');
const pptxgen = require('pptxgenjs');
const { applyTheme } = require(process.env.SKILL_DIR + '/scripts/apply_theme.js');
const { HEX, make } = require('./deck_helpers.js');
const N = require('./text.js');               // all text + numbers (text.js -> numbers.js, fignum.js)

const WT = process.env.WT;
const PRES = path.join(WT, 'notes', 'final_pipeline', 'presentation');
const F2 = (n) => path.join(PRES, 'figures_v2', n);
const F1 = (n) => path.join(PRES, 'figures', n);
const OUTF = path.join(PRES, 'EMG_torque_supervisor_v2.pptx');

const THEME = {
  name: 'EMG Torque', headFontFace: 'Calibri', bodyFontFace: 'Calibri',
  colors: { dk1: '1B1B1B', lt1: 'FFFFFF', dk2: '1F2A44', lt2: 'F2F1EE', accent1: 'EB6834', accent2: '2A78D6',
            accent3: '1BAF7A', accent4: 'C0392B', accent5: '8A8984', accent6: 'EDA100', hlink: '2A78D6', folHlink: '6D2E46' },
};
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.theme = { headFontFace: 'Calibri', bodyFontFace: 'Calibri' };
pres.title = 'Predicting voluntary ankle torque from muscle signals';
const C = pres.SchemeColor;
const H = make(pres);
const { txt, card, box, pill, bullets, b, t, img, circleNum } = H;

pres.defineSlideMaster({
  title: 'TITLE_DARK', background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 1.9, w: 11.7, h: 1.9, fontSize: 38, bold: true, color: C.background1, valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.05, w: 11.7, h: 2.6, fontSize: 19, color: 'DDE3EE', valign: 'top', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'SECTION', background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 36, bold: true, color: C.background1, valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 3.95, w: 11.7, h: 1.6, fontSize: 18, color: 'DDE3EE', valign: 'top', margin: 0 }, text: '' } },
  ],
  slideNumber: { x: 12.45, y: 7.0, w: 0.6, h: 0.3, fontSize: 11, color: 'AAB4C6', align: 'right' },
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: 'FFFFFF' },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.5, y: 0.25, w: 12.3, h: 0.8, fontSize: 28, bold: true, color: C.text2, valign: 'middle', margin: 0 }, text: '' } },
  ],
  slideNumber: { x: 12.45, y: 7.05, w: 0.6, h: 0.3, fontSize: 11, color: '8A8984', align: 'right' },
});

let nSlides = 0;
let section = 'Intro';
function slide(title, master = 'CONTENT') {
  nSlides += 1;
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  s.addText(title, { placeholder: 'title' });
  return s;
}
function newSection(name, title, sub) {
  section = name;
  pres.addSection({ title: name });
  if (title) {
    const s = slide(title, 'SECTION');
    if (sub) s.addText(sub, { placeholder: 'body' });
    return s;
  }
  return null;
}
const foot = (s, text) => txt(s, text, { x: 0.5, y: 6.95, w: 11.8, h: 0.4, fontSize: 11, color: HEX.ink2 });
const R2 = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

// ════════════════════════════ INTRO ════════════════════════════
pres.addSection({ title: 'Intro' });
let s = slide('Predicting the ankle torque made by the muscles, from muscle signals', 'TITLE_DARK');
s.addText([
  { text: 'What went wrong at the toes-down angles last semester, what we changed and why, how we check the model now, and the results for all 6 people', options: { breakLine: true } },
  { text: 'Old pipeline (last semester) vs new pipeline · October 2026', options: { fontSize: 15, color: 'AAB4C6' } },
], { placeholder: 'body' });
s.addNotes('In this presentation we go through what went wrong last semester at the toes-down angles, the five problems we found, how we fixed each one, how we now check the model, and the results for all six people. We finish with the limits and the sources.');

// ── the experiment
s = slide('The experiment');
img(s, F1('setup_angles.png'), 0.4, 1.2, 6.3, 4.4);
txt(s, 'Negative angle = toes down. The calf is short there, so people can only push weakly (about 17–30 % of their strongest push).', { x: 0.55, y: 5.7, w: 6.0, h: 0.8, fontSize: 13, color: HEX.ink2, italic: true });
[
  ['Setup', 'The foot is strapped to a footplate that is locked at one angle, so the ankle cannot move. A sensor measures the torque (twist) at the ankle.'],
  ['Task', 'Push the toes down, relax, repeat, for about 90 s. That is one trial. One trial per angle, 8 angles, from toes down (about −45°) to toes up (about +14°).'],
  ['Recorded', '4 muscle signals (EMG): 3 calf muscles (GM, GL, SOL) that push the toes down and 1 shin muscle (TA) that pulls them up. Plus the ankle angle and the torque, 1000 times per second.'],
  ['People, sessions', '6 people. Each did the 8 trials twice in the same visit: session 1 (“test”) and session 2 (“retest”). The electrodes stayed on.'],
  ['Push or pull?', '5 people only pushed the toes down. CH pulled the toes up at its most toes-down angles, and at one angle pulled in session 1 but pushed in session 2.'],
].forEach((r, i) => {
  const y = 1.2 + i * 1.12;
  card(s, 7.0, y, 5.85, 1.02);
  txt(s, r[0], { x: 7.15, y: y + 0.1, w: 1.45, h: 0.8, fontSize: 14, bold: true, color: C.accent1 });
  txt(s, r[1], { x: 8.6, y: y + 0.08, w: 4.15, h: 0.9, fontSize: 12.5, valign: 'middle' });
});
s.addNotes('The foot is locked at one angle and the person pushes the toes down and relaxes for about 90 seconds; that is one trial. There are 8 angles, and each person did all 8 twice in the same visit, with the electrodes left on. We record 4 muscle signals, the angle and the torque. Negative angles are toes down; there the calf is short and weak. Five people only pushed; CH pulled the toes up at its most toes-down angles, which the torque data shows clearly.');

// ── what the model does
s = slide('What the model does, and how we score it');
box(s, 0.6, 1.35, 3.6, 1.55, HEX.paleBlue, HEX.blue);
txt(s, [b('Input'), t('\nthe last 0.2 s of the 4 muscle signals + the ankle angle (20 time points × 5 numbers)')], { x: 0.75, y: 1.45, w: 3.3, h: 1.35, fontSize: 13.5, align: 'center', valign: 'middle' });
H.arrow(s, 4.3, 2.12, 4.95, 2.12);
box(s, 5.05, 1.35, 3.2, 1.55, HEX.light, HEX.navy);
txt(s, [b('Model'), t('\na small neural network with a few thousand adjustable numbers (old: 1,001, new: 3,665)')], { x: 5.2, y: 1.45, w: 2.9, h: 1.35, fontSize: 13.5, align: 'center', valign: 'middle' });
H.arrow(s, 8.35, 2.12, 9.0, 2.12);
box(s, 9.1, 1.35, 3.6, 1.55, HEX.paleOrange, HEX.orange);
txt(s, [b('Output'), t('\nthe voluntary torque right now (Nm). Slide the 0.2 s window forward 10 ms at a time → a whole torque curve')], { x: 9.25, y: 1.45, w: 3.3, h: 1.35, fontSize: 13.5, align: 'center', valign: 'middle' });
[
  ['Learning = fitting', 'The model’s numbers are adjusted so its output matches the voluntary torque of one session, exactly like a least-squares curve fit, just with many more parameters. We fit on the first 80 % of each trial and use the last 20 % to decide when to stop fitting (so it does not over-fit).'],
  ['Testing', 'The fitted model then predicts the other session, which it never saw. One model per person (we do not mix people).'],
  ['Score (per angle)', 'R², as in curve fitting: 1 = perfect, 0 = no better than a flat line at the average, below 0 = worse than that flat line. Plus the error in Nm (typical size of the miss), because R² alone can mislead.'],
].forEach((r, i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 3.3, 3.9, 3.55);
  txt(s, r[0], { x: x + 0.2, y: 3.45, w: 3.5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  txt(s, r[1], { x: x + 0.2, y: 3.92, w: 3.5, h: 2.85, fontSize: 13.5 });
});
s.addNotes('The model looks at the last 0.2 seconds of the four muscle signals plus the ankle angle and outputs the torque the muscles are making right now. Learning is just fitting: like a least-squares fit in MATLAB, but with a few thousand parameters. We fit on one session and test on the other session, which the model has never seen. We score each angle with R², the same R² as in curve fitting, and always with the error in Newton-metres next to it.');

// ════════════════════════════ PROBLEM ════════════════════════════
newSection('The problem');
s = slide(N.problemTitle);
img(s, F2('problem_all6.png'), 0.35, 1.15, 12.6, 6.05);
s.addNotes(N.problemNotes);

// ── what the sensor measures
s = slide('What the torque sensor measures');
img(s, F2('vt_concept.png'), 0.35, 1.15, 8.2, 5.9, 'left');
card(s, 8.75, 1.2, 4.1, 5.85, HEX.light);
bullets(s, [
  [b('measured torque = passive torque + voluntary torque')],
  [b('Passive: '), t('what the sensor reads when the muscles are switched off. The stretched tissues around the ankle pull back like a rubber band (plus the weight of the foot). It depends on the angle.')],
  [b('Voluntary: '), t('the extra twist the muscles add when the person pushes. This is what we want the model to predict.')],
  [b('So: '), t('voluntary = measured − passive. To get the right answer for the model we must know the passive level. Everything on the next slides is about getting that number right.')],
], { x: 8.95, y: 1.4, w: 3.75, h: 5.5, fontSize: 14.5, gap: 10 });
s.addNotes('The torque sensor does not know who is making the torque. When the person is fully relaxed it still reads something: that is the passive torque, the stretched tissues pulling back plus the weight of the foot. When the person pushes, the reading changes by the voluntary torque. We want the model to predict only the voluntary part, so we subtract the passive part. That means we need to know the passive level, and that is exactly where things went wrong.');

// ── real data, strong angle
s = slide('Real data: a strong angle (easy case)');
img(s, F2('vt_real_strong.png'), 0.35, 1.15, 8.3, 5.95, 'left');
card(s, 8.85, 1.2, 4.0, 5.85, HEX.light);
txt(s, 'How to read it', { x: 9.05, y: 1.35, w: 3.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
bullets(s, N.strongBullets, { x: 9.05, y: 1.85, w: 3.65, h: 5.1, fontSize: 13.5, gap: 8 });
s.addNotes(N.strongNotes);

// ── real data, toes-down angle
s = slide('Real data: the most toes-down angle (where it broke)');
img(s, F2('vt_real_toesdown.png'), 0.35, 1.15, 8.3, 5.95, 'left');
card(s, 8.85, 1.2, 4.0, 5.85, HEX.paleOrange);
txt(s, 'Why the passive value matters here', { x: 9.05, y: 1.35, w: 3.6, h: 0.7, fontSize: 16, bold: true, color: C.text2 });
bullets(s, N.toesBullets, { x: 9.05, y: 2.1, w: 3.65, h: 4.9, fontSize: 13.5, gap: 8 });
s.addNotes(N.toesNotes);

// ── five problems roadmap
s = slide('Five things broke the toes-down angles');
[
  ['Wrong passive level', 'The passive value came from a separate relaxed recording. Inside the pushing trial the relaxed level is different.', 'Measure the passive level inside the same trial', N.slideRef.passive],
  ['Passive drift', 'Even inside one 90 s trial, the relaxed level slowly slides.', 'The in-trial line follows the slide', N.slideRef.drift],
  ['Bad table entry (JM)', 'Pieces of a recording that were not relaxed got into the table: 12 Nm error at one angle.', 'No table needed any more', N.slideRef.table],
  ['EMG size changes between sessions', 'Same push, different EMG size in session 2 (JM: about half).', 'Auto-gain: each muscle rescaled by its own recent range', N.slideRef.agc],
  ['The fit ignored the small angles', 'Least squares in Nm: errors at strong angles are hundreds of times bigger, so they dominate.', 'Divide each angle’s torque by its typical swing', N.slideRef.scale],
].forEach((c, i) => {
  const y = 1.2 + i * 1.15;
  circleNum(s, 0.5, y + 0.22, i + 1, HEX.navy);
  txt(s, c[0], { x: 1.15, y: y + 0.05, w: 2.9, h: 0.95, fontSize: 15, bold: true, valign: 'middle' });
  txt(s, c[1], { x: 4.1, y: y + 0.05, w: 4.35, h: 0.95, fontSize: 13, valign: 'middle' });
  card(s, 8.6, y + 0.05, 3.55, 0.95, HEX.paleOrange);
  txt(s, '→ ' + c[2], { x: 8.75, y: y + 0.05, w: 3.3, h: 0.95, fontSize: 13, bold: true, color: C.accent1, valign: 'middle' });
  txt(s, 'slide ' + c[3], { x: 12.2, y: y + 0.05, w: 0.75, h: 0.95, fontSize: 11, color: HEX.ink2, valign: 'middle' });
});
s.addNotes('We traced the failure to five concrete problems. None of them is in the neural network itself: three are about the passive torque, one is about the EMG signal size, and one is about how the fit weights the angles. The next slides take them one by one: what the problem is, how we found it, and the fix.');

// ── old table, how built
newSection('Problems 1–3: the passive torque');
s = slide('Problems 1 & 3: the old passive table — what it was and how it was built');
img(s, F2('oldtable_passive_trial.png'), 0.35, 1.2, 7.9, 5.9, 'left', 'middle');
card(s, 8.45, 1.2, 4.4, 5.85, HEX.light);
bullets(s, N.tableBullets, { x: 8.65, y: 1.35, w: 4.05, h: 5.6, fontSize: 12.5, gap: 6 });
s.addNotes(N.tableNotes);

// ── the JM table and the bug
s = slide('The old table for JM, and the bug in it');
img(s, F2('oldtable_map_JM.png'), 0.35, 1.15, 8.3, 5.95, 'left', 'middle');
card(s, 8.85, 1.2, 4.0, 5.85, HEX.light);
bullets(s, N.jmBullets, { x: 9.05, y: 1.35, w: 3.65, h: 5.6, fontSize: 13, gap: 8 });
s.addNotes(N.jmNotes);

// ── how we saw it was wrong
s = slide('How we saw the old passive values were wrong');
img(s, F2('oldtable_vs_intrial_all.png'), 0.35, 1.15, 8.5, 5.95, 'left');
card(s, 9.05, 1.2, 3.8, 5.85, HEX.light);
bullets(s, N.wrongBullets, { x: 9.25, y: 1.35, w: 3.45, h: 5.6, fontSize: 13, gap: 8 });
s.addNotes(N.wrongNotes);

// ── drift
s = slide('Problem 2: passive drift — the relaxed level slides during a trial');
img(s, F2('drift_examples.png'), 0.35, 1.15, 8.4, 5.95, 'left', 'middle');
card(s, 8.95, 1.2, 3.9, 5.85, HEX.light);
bullets(s, N.driftBullets, { x: 9.15, y: 1.35, w: 3.55, h: 5.6, fontSize: 13, gap: 8 });
s.addNotes(N.driftNotes);

// ── the fix: in-trial passive
s = slide('The fix for 1–3: measure the passive level inside the same trial');
img(s, F2('vt_newmethod_steps.png'), 0.35, 1.15, 8.2, 5.95, 'left');
card(s, 8.75, 1.2, 4.1, 5.85, HEX.paleOrange);
bullets(s, N.fixBullets, { x: 8.95, y: 1.32, w: 3.78, h: 5.68, fontSize: 12.3, gap: 6 });
s.addNotes(N.fixNotes);

// ── EMG size
newSection('Problem 4: EMG size');
s = slide('Problem 4: the EMG size changes between session 1 and session 2');
img(s, F2('emg_size_change.png'), 0.35, 1.15, 12.6, 3.3);
card(s, 0.5, 4.6, 6.1, 2.45, HEX.light);
txt(s, 'Why it matters', { x: 0.7, y: 4.7, w: 5.7, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
bullets(s, N.emgWhyBullets, { x: 0.7, y: 5.1, w: 5.75, h: 1.9, fontSize: 12.5, gap: 5 });
card(s, 6.75, 4.6, 6.1, 2.45, HEX.light);
txt(s, 'What could explain it (electrodes stayed on)', { x: 6.95, y: 4.7, w: 5.7, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
bullets(s, N.emgCauseBullets, { x: 6.95, y: 5.1, w: 5.75, h: 1.9, fontSize: 12, gap: 4 });
s.addNotes(N.emgNotes);

// ── auto-gain
s = slide('The fix for 4: auto-gain, an automatic volume control for each muscle');
img(s, F2('agc_timeseries.png'), 0.35, 1.15, 7.9, 5.95, 'left');
card(s, 8.45, 1.2, 4.4, 5.85, HEX.paleOrange);
bullets(s, N.agcBullets, { x: 8.65, y: 1.35, w: 4.05, h: 5.6, fontSize: 12.5, gap: 6 });
s.addNotes(N.agcNotes);

// ── small angles
newSection('Problem 5: the fit');
s = slide('Problem 5: the fit ignored the small angles');
img(s, F2('swing_per_angle.png'), 0.35, 1.15, 12.6, 3.65);
card(s, 0.5, 4.95, 12.35, 2.1, HEX.light);
bullets(s, N.lossBullets, { x: 0.7, y: 5.08, w: 12.0, h: 1.9, fontSize: 13.5, gap: 6 });
s.addNotes(N.lossNotes);

s = slide('The fix for 5: divide each angle’s torque by its typical swing');
img(s, F2('swing_divide_example.png'), 0.35, 1.15, 7.9, 5.95, 'left');
card(s, 8.45, 1.2, 4.4, 5.85, HEX.paleOrange);
bullets(s, N.scaleBullets, { x: 8.65, y: 1.35, w: 4.05, h: 5.6, fontSize: 13, gap: 7 });
s.addNotes(N.scaleNotes);

// ════════════════════════════ NEW PIPELINE & MODEL ════════════════════════════
newSection('Pipeline and model');
s = slide('Old vs new pipeline, step by step');
const steps = [
  ['1. Clean the EMG', 'filter (30–300 Hz), absolute value, smooth (2 Hz)', 'same'],
  ['2. Scale the EMG', '÷ the biggest value seen in session 1', '÷ biggest value in session 1, then auto-gain: each muscle follows its own recent range (switched off for people who push both ways)'],
  ['3. Make the “right answer”', 'measured − one passive number per trial, from the separate relaxed recording', 'measured − passive level measured inside the same trial (smooth line through the relaxed moments)'],
  ['4. The model', 'LSTM network, 8 memory cells per layer, one output (1,001 numbers)', 'same kind of network, 16 memory cells per layer + one small correction per angle (3,665 numbers)'],
  ['5. The fit', 'least squares on torque in Nm: the strong angles dominate', 'least squares on torque ÷ that angle’s typical swing: every angle counts the same (1/8 each)'],
  ['6. Checking', 'learn session 1 → test session 2; 1 run; R² against its own answer key', 'both directions; 3 runs averaged; one shared answer key for every model; R² + error in Nm + phantom torque; 3 extra people'],
];
txt(s, 'OLD (last semester)', { x: 0.5, y: 1.1, w: 4, h: 0.35, fontSize: 15, bold: true, color: C.accent2 });
txt(s, 'NEW', { x: 0.5, y: 4.0, w: 4, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
steps.forEach((st, i) => {
  const x = 0.5 + i * 2.07, w = 1.92;
  box(s, x, 1.5, w, 2.3, HEX.paleBlue, HEX.blue);
  txt(s, [b(st[0], { fontSize: 14 }), t('\n' + st[1])], { x: x + 0.08, y: 1.58, w: w - 0.16, h: 2.15, fontSize: 13, align: 'center', valign: 'middle' });
  const same = st[2] === 'same';
  box(s, x, 4.4, w, 2.5, same ? HEX.light : HEX.paleOrange, same ? HEX.muted : HEX.orange);
  txt(s, [b(st[0], { fontSize: 14 }), t('\n' + st[2])], { x: x + 0.08, y: 4.48, w: w - 0.16, h: 2.35, fontSize: 13, align: 'center', valign: 'middle' });
  if (i < 5) { H.arrow(s, x + w + 0.01, 2.65, x + 2.06, 2.65); H.arrow(s, x + w + 0.01, 5.65, x + 2.06, 5.65); }
});
txt(s, 'orange = changed. Only step 5 adjusts the model’s numbers by fitting; the other steps are fixed processing (using values from the learning session) or checking.', { x: 0.5, y: 6.98, w: 11.8, h: 0.35, fontSize: 12, color: HEX.ink2 });
s.addNotes('Here is the whole pipeline side by side. The EMG cleaning is unchanged. The EMG scaling now has auto-gain. The right answer uses the passive level measured inside the same trial. The network is the same kind as before, a bit wider, with one small correction per angle. The fit gives every angle the same weight. And the checking is much stricter; we come back to that in a few slides.');

// ── the LSTM
s = slide('The model: the same kind of network as before (LSTM), a bit wider');
card(s, 0.5, 1.2, 6.15, 5.85, HEX.light);
txt(s, 'What an LSTM does, in plain words', { x: 0.7, y: 1.35, w: 5.8, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
bullets(s, [
  'It reads the 0.2 s window one time point at a time (20 points, 10 ms apart), like reading a sentence word by word.',
  'It keeps a small running memory while it reads, so it can use how the muscle signals were changing, not just their value right now.',
  'At the end of the window its memory is summarised in a few numbers (“memory cells”). A final weighted sum of those numbers gives the torque.',
  'Two such layers are stacked: the second reads what the first produced.',
  'Same network type as last semester; we kept it because it worked well everywhere except the toes-down angles, and the problem turned out to be in the data.',
], { x: 0.7, y: 1.85, w: 5.8, h: 5.1, fontSize: 13.5, gap: 8 });
card(s, 6.85, 1.2, 6.0, 5.85, HEX.paleOrange);
txt(s, 'Why 16 memory cells instead of 8?', { x: 7.05, y: 1.35, w: 5.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
bullets(s, N.lstmBullets, { x: 7.05, y: 1.85, w: 5.65, h: 5.1, fontSize: 13.5, gap: 8 });
s.addNotes(N.lstmNotes);

// ── the experts, concretely
s = slide('The per-angle corrections (“experts”), concretely');
box(s, 0.5, 1.35, 2.55, 1.6, HEX.light, HEX.navy);
txt(s, [b('LSTM part'), t('\nreads the 0.2 s window → 16 summary numbers h1 … h16')], { x: 0.6, y: 1.42, w: 2.35, h: 1.45, fontSize: 12.5, align: 'center', valign: 'middle' });
box(s, 3.6, 1.2, 3.55, 1.25, HEX.light, HEX.navy);
txt(s, [b('Shared read-out'), t('\nw1·h1 + … + w16·h16 + offset\nsame for every angle (17 numbers)')], { x: 3.68, y: 1.25, w: 3.4, h: 1.15, fontSize: 12, align: 'center', valign: 'middle' });
box(s, 3.6, 2.65, 3.55, 1.5, HEX.paleOrange, HEX.orange);
txt(s, [b('Correction of this angle'), t('\nc1·h1 + … + c16·h16\none set of 16 multipliers per angle (8 angles → 128 numbers), start at 0, kept small')], { x: 3.68, y: 2.7, w: 3.4, h: 1.4, fontSize: 12, align: 'center', valign: 'middle' });
box(s, 3.6, 4.4, 3.55, 1.05, HEX.paleBlue, HEX.blue);
txt(s, [b('Angle selector'), t('\nat a trained angle: use its correction; between two angles: blend the two')], { x: 3.68, y: 4.43, w: 3.4, h: 1.0, fontSize: 12, align: 'center', valign: 'middle' });
txt(s, 'ankle angle →', { x: 1.6, y: 4.75, w: 1.9, h: 0.4, fontSize: 12.5, align: 'right', color: HEX.ink2 });
H.arrow(s, 3.1, 1.8, 3.55, 1.8); H.arrow(s, 3.1, 2.6, 3.55, 3.3);
H.arrow(s, 5.38, 4.38, 5.38, 4.18);
s.addShape(pres.shapes.OVAL, { x: 7.55, y: 2.2, w: 0.55, h: 0.55, fill: { color: 'FFFFFF' }, line: { color: HEX.navy, width: 1.5 } });
txt(s, '+', { x: 7.55, y: 2.2, w: 0.55, h: 0.55, fontSize: 20, bold: true, align: 'center', valign: 'middle', color: HEX.navy });
H.arrow(s, 7.18, 1.8, 7.6, 2.28); H.arrow(s, 7.18, 3.3, 7.6, 2.68);
box(s, 8.45, 1.95, 2.0, 1.05, HEX.light, HEX.navy);
txt(s, [b('× typical swing'), t('\nof this angle')], { x: 8.5, y: 2.0, w: 1.9, h: 0.95, fontSize: 12, align: 'center', valign: 'middle' });
H.arrow(s, 8.12, 2.47, 8.42, 2.47);
H.arrow(s, 10.47, 2.47, 10.85, 2.47);
box(s, 10.9, 1.95, 1.95, 1.05, HEX.paleOrange, HEX.orange);
txt(s, [b('torque'), t('\n(Nm)')], { x: 10.95, y: 2.0, w: 1.85, h: 0.95, fontSize: 12.5, align: 'center', valign: 'middle' });
card(s, 7.55, 3.4, 5.3, 3.65, HEX.light);
bullets(s, [
  [b('Not a separate mini-network: '), t('just one extra set of 16 multipliers per angle, added to the shared read-out. In effect each angle gets its own adjusted read-out weights (w + c).')],
  [b('In fitting terms: '), t('a shared slope plus a small per-angle deviation that is pulled toward zero, similar in spirit to random slopes in a mixed-effects model.')],
  [b('Why: '), t('the link between EMG size and torque changes with muscle length (short calf at toes down → less torque for the same EMG). One shared read-out gets pulled toward the strong angles.')],
  [b('No per-angle offset on purpose: '), t('CH pushed one way in session 1 and the other way in session 2 at one angle; an offset would learn “the usual torque here” and be wrong.')],
], { x: 7.75, y: 3.52, w: 4.95, h: 3.45, fontSize: 12, gap: 5 });
card(s, 0.5, 5.65, 6.85, 1.4, HEX.paleBlue);
txt(s, [b('What it achieves: '), t(N.expertsAchieve)], { x: 0.7, y: 5.75, w: 6.5, h: 1.2, fontSize: 12.5 });
s.addNotes('Concretely: the LSTM part reads the window and ends with 16 summary numbers. The old model turned them into torque with one weighted sum, the same for every angle. The new model adds, for the current angle, a second weighted sum with that angle’s own 16 multipliers. They start at zero and are kept small, so an angle only gets its own correction if the data asks for it. Between two trained angles the two corrections are blended. Then we multiply by the typical swing of that angle to get back to Newton-metres. It is not a separate network, just 16 extra numbers per angle; think of a shared slope plus a per-angle deviation, like random slopes in a mixed-effects model.');

// ── what each change contributes (ablation)
s = slide('What each change contributes (remove one change at a time)');
s.addTable(N.ablationRows, { x: 0.5, y: 1.2, w: 12.35, colW: N.ablationColW, rowH: 0.4, fontFace: 'Calibri', fontSize: 12.5,
  border: { type: 'solid', pt: 0.75, color: 'D9D7D2' }, margin: 0.05 });
card(s, 0.5, N.ablationNoteY, 12.35, 7.05 - N.ablationNoteY, HEX.light);
bullets(s, N.ablationBullets, { x: 0.7, y: N.ablationNoteY + 0.1, w: 12.0, h: 7.0 - N.ablationNoteY - 0.15, fontSize: 12, gap: 3 });
s.addNotes(N.ablationNotes);

// ════════════════════════════ CHECKING ════════════════════════════
newSection('How we check');
s = slide('How the model is checked: before vs now');
const chk = [
  ['Which way', 'learned from session 1, tested on session 2', 'both ways: session 1 → 2 and session 2 → 1 (two independent tests per person)'],
  ['Training runs', '1 run', '3 runs per test, averaged (never the best one). Each run starts from different random numbers'],
  ['Answer key', 'each pipeline graded against its own “right answer” (its own passive estimate)', 'one shared answer key for every model: measured torque − passive level measured inside the test trial'],
  ['What we report', 'R² per angle', 'R² per angle + error in Nm + phantom torque (torque predicted while the person is relaxed)'],
  ['Who', 'the people we were fixing', '3 main people + 3 extra people checked afterwards (IES, CH, HM)'],
  ['Picking the final version', '—', 'a rule written down before looking at the extra people (next slides)'],
  ['Code', '—', 'an independent review of the code for leaks (test data accidentally used in fitting) and wrong scores: 16 issues found and fixed before the final runs'],
];
const chkRows = [[
  { text: '', options: { fill: { color: HEX.navy } } },
  { text: 'Before (last semester)', options: { bold: true, color: 'FFFFFF', fill: { color: HEX.blue }, align: 'center' } },
  { text: 'Now', options: { bold: true, color: 'FFFFFF', fill: { color: HEX.orange }, align: 'center' } },
]];
chk.forEach(r => chkRows.push([
  { text: r[0], options: { bold: true, color: HEX.navy, fill: { color: HEX.light } } },
  { text: r[1], options: { fill: { color: 'FFFFFF' } } },
  { text: r[2], options: { fill: { color: 'FFFFFF' } } },
]));
s.addTable(chkRows, { x: 0.5, y: 1.2, w: 12.35, colW: [2.3, 4.3, 5.75], rowH: 0.62, fontFace: 'Calibri', fontSize: 13, valign: 'middle',
  border: { type: 'solid', pt: 0.75, color: 'D9D7D2' }, margin: 0.06 });
card(s, 0.5, 6.35, 12.35, 0.75, HEX.navy);
txt(s, 'Same old model, fair answer key: its average R² drops from 0.68 to 0.54. Its own answer key was hiding failures.', { x: 0.7, y: 6.4, w: 12.0, h: 0.65, fontSize: 15, bold: true, color: 'FFFFFF', valign: 'middle' });
s.addNotes('This table is the checklist of how we check the model now compared with last semester. We test both ways, we average three training runs, we grade every model with the same answer key, we report the error in Newton-metres and the phantom torque, we check on three extra people, we picked the final version with a written rule, and the code was reviewed for leaks. With the fair answer key, the old model actually looks worse than we thought: 0.54 instead of 0.68.');

// ── both directions + split
s = slide('“Both directions”: two independent tests per person');
const rowY = [1.5, 4.0];
[['Session 1', 'Session 2', 'Model A: learned from session 1 → tested on session 2 (what we did last semester)'],
 ['Session 2', 'Session 1', 'Model B: learned from session 2 → tested on session 1 (new)']].forEach((r, i) => {
  const y = rowY[i];
  txt(s, r[2], { x: 0.5, y: y - 0.45, w: 12, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  txt(s, r[0] + ' (8 trials)', { x: 0.5, y: y + 0.05, w: 2.2, h: 0.4, fontSize: 13, bold: true });
  for (let k = 0; k < 8; k++) {
    const yy = y + 0.5 + k * 0.17;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: yy, w: 2.8, h: 0.13, fill: { color: HEX.blue }, line: { color: HEX.blue } });
    s.addShape(pres.shapes.RECTANGLE, { x: 3.3, y: yy, w: 0.7, h: 0.13, fill: { color: HEX.green }, line: { color: HEX.green } });
  }
  txt(s, 'fit on the first 80 % of each trial (blue); last 20 % (green) only used to decide when to stop', { x: 4.1, y: y + 0.6, w: 3.0, h: 1.0, fontSize: 12, color: HEX.ink2 });
  H.arrow(s, 7.15, y + 1.15, 8.0, y + 1.15);
  txt(s, r[1] + ' (8 trials)', { x: 8.1, y: y + 0.05, w: 2.6, h: 0.4, fontSize: 13, bold: true });
  for (let k = 0; k < 8; k++) {
    const yy = y + 0.5 + k * 0.17;
    s.addShape(pres.shapes.RECTANGLE, { x: 8.1, y: yy, w: 3.5, h: 0.13, fill: { color: HEX.orange }, line: { color: HEX.orange } });
  }
  txt(s, 'whole session predicted, never seen during fitting', { x: 11.7, y: y + 0.55, w: 1.3, h: 1.2, fontSize: 11.5, color: HEX.ink2 });
});
card(s, 0.5, 6.25, 12.35, 0.85, HEX.light);
txt(s, 'Why both: if a fix only worked one way (for example because session 2 happened to be cleaner), it would show up here. Every number in the results is the average of both directions unless the slide says otherwise.', { x: 0.7, y: 6.3, w: 12.0, h: 0.75, fontSize: 13.5, valign: 'middle' });
s.addNotes('Both directions means two separate models per person: one learns from session 1 and is tested on session 2, which is what we did last semester, and one learns from session 2 and is tested on session 1. Inside the learning session we fit on the first 80 percent of each trial and use the last 20 percent only to decide when to stop fitting. The tested session is never used for fitting.');

// ── one shared answer key
s = slide('One shared answer key for every model');
img(s, F2('answer_key.png'), 0.35, 1.15, 8.3, 5.95, 'left');
card(s, 8.85, 1.2, 4.0, 5.85, HEX.light);
bullets(s, N.keyBullets, { x: 9.05, y: 1.35, w: 3.65, h: 5.6, fontSize: 13, gap: 8 });
s.addNotes(N.keyNotes);

// ── phantom torque
s = slide('Phantom torque: torque predicted while the person is relaxed');
img(s, F2('phantom.png'), 0.35, 1.15, 12.6, 4.6);
card(s, 0.5, 5.85, 12.35, 1.2, HEX.light);
bullets(s, N.phantomBullets, { x: 0.7, y: 5.95, w: 12.0, h: 1.05, fontSize: 13.5, gap: 4 });
s.addNotes(N.phantomNotes);

// ── 3 runs
s = slide('Randomness: why we do 3 training runs');
img(s, F2('three_runs.png'), 0.35, 1.15, 12.6, 4.6);
card(s, 0.5, 5.85, 12.35, 1.2, HEX.light);
bullets(s, N.runsBullets, { x: 0.7, y: 5.95, w: 12.0, h: 1.05, fontSize: 13.5, gap: 4 });
s.addNotes(N.runsNotes);

// ── how the final version was picked
s = slide('How we picked the final version (the “written rule”)');
const tl = [
  ['1. Before looking', 'We wrote down a rule (in a dated file): “the version with the highest average R² on YES, JM and EG, both directions, 3 runs each, wins”.'],
  ['2. The rule picked', 'A version that still used the separate relaxed recordings (with a correction) and a longer 0.4 s window: average R² 0.932, vs 0.925 for the in-trial version. Basically a tie.'],
  ['3. Check on 3 extra people', 'It failed for IES, who has no separate relaxed recordings at all (its passive value could not be made). At CH’s flipped angle (pulled in session 1, pushed in session 2) both versions failed until auto-gain was switched off for CH.'],
  ['4. Final choice', 'The in-trial passive version (now “new”) works for IES and HM, and for CH with auto-gain off. It tied on the main 3, so we recommend it.'],
];
tl.forEach((r, i) => {
  const x = 0.5 + i * 3.1;
  box(s, x, 1.3, 2.9, 3.6, i === 3 ? HEX.paleOrange : HEX.light, i === 3 ? HEX.orange : HEX.navy);
  txt(s, r[0], { x: x + 0.15, y: 1.4, w: 2.6, h: 0.5, fontSize: 15, bold: true, color: C.text2 });
  txt(s, r[1], { x: x + 0.15, y: 1.95, w: 2.6, h: 2.85, fontSize: 13 });
  if (i < 3) H.arrow(s, x + 2.92, 3.1, x + 3.08, 3.1);
});
card(s, 0.5, 5.15, 12.35, 1.9, HEX.light);
bullets(s, [
  [b('Honest part: '), t('the final choice was made after seeing the 3 extra people, so for that choice they are not a fully independent test. A new person, or a session on another day, would be.')],
  [b('Why a written rule at all: '), t('we tried 23 versions. If you pick the best of many on the same data, the winner looks better than it really is (like picking the best of 23 fits). Writing the rule first keeps us honest.')],
], { x: 0.7, y: 5.25, w: 12.0, h: 1.75, fontSize: 13.5, gap: 6 });
s.addNotes('Because we tried 23 versions, we wrote down before looking at the three extra people how we would pick the winner: the highest average R² on the three main people. That rule picked a version that still used the separate relaxed recordings plus a longer window; it was basically tied with the in-trial version. On the extra people it broke for IES, who has no separate relaxed recordings. At CH’s one flipped angle both versions failed until we switched auto-gain off for CH, which is now an automatic rule. The in-trial version works for IES and HM, and for CH with auto-gain off, so we recommend it. To be honest, that last choice was made after seeing the extra people.');

// ════════════════════════════ RESULTS ════════════════════════════
newSection('Results');
s = slide('Results: every person, old vs new');
img(s, F2('results_summary.png'), 0.35, 1.15, 12.6, 4.4);
card(s, 0.5, 5.6, 12.35, 1.45, HEX.light);
bullets(s, N.summaryBullets, { x: 0.7, y: 5.7, w: 12.0, h: 1.3, fontSize: 13, gap: 4 });
s.addNotes(N.summaryNotes);

s = slide('Results per angle: learned from session 1 → tested on session 2');
img(s, F2('results_r2_S1toS2.png'), 0.35, 1.1, 12.6, 6.0);
s.addNotes('These are the same per-angle graphs as the problem slide, now with old and new side by side, both graded with the shared answer key. The red-zone toes-down bars of the old pipeline are gone.');

s = slide('Results per angle: learned from session 2 → tested on session 1');
img(s, F2('results_r2_S2toS1.png'), 0.35, 1.1, 12.6, 6.0);
s.addNotes('The other direction tells the same story.');

s = slide('Error in Nm per angle (learned from session 1 → tested on session 2)');
img(s, F2('results_error_S1toS2.png'), 0.35, 1.1, 12.6, 6.0);
s.addNotes(N.errorNotes);

s = slide('Is R² = 0.99 too good to be true?');
img(s, F2('sus_r2_vs_swing.png'), 0.35, 1.15, 7.9, 4.4, 'left');
card(s, 8.45, 1.2, 4.4, 5.85, HEX.light);
bullets(s, N.susBullets, { x: 8.65, y: 1.35, w: 4.05, h: 5.6, fontSize: 12.5, gap: 6 });
card(s, 0.5, 5.7, 7.75, 1.35, HEX.paleBlue);
txt(s, N.susBox, { x: 0.7, y: 5.78, w: 7.4, h: 1.2, fontSize: 12.5 });
s.addNotes(N.susNotes);

// full trial inspections
newSection('Full trial inspection');
['YES', 'JM', 'EG', 'IES', 'CH', 'HM'].forEach(subj => {
  s = slide(`Full trial inspection · ${subj} · learned from session 1 → tested on session 2`);
  pill(s, 0.5, 1.1, 'OLD', HEX.blue);
  pill(s, 6.95, 1.1, 'NEW', HEX.orange);
  img(s, F2(`inspect_${subj}_S1toS2_old.png`), 0.35, 1.5, 6.35, 5.45, 'center');
  img(s, F2(`inspect_${subj}_S1toS2_new.png`), 6.75, 1.5, 6.35, 5.45, 'center');
  foot(s, N.inspectFoot[subj]);
  s.addNotes(N.inspectNotes[subj]);
});

// hardest angles
s = slide('The hardest angle of each person, up close (1/2)');
img(s, F2('hardest_1.png'), 0.35, 1.1, 12.6, 5.85);
foot(s, '“learned from session 1, tested on session 2” = the model was fitted on session 1 and then predicted session 2, which it never saw. Black = truth, blue = old, orange = new.');
s.addNotes(N.hardestNotes1);
s = slide('The hardest angle of each person, up close (2/2)');
img(s, F2('hardest_2.png'), 0.35, 1.1, 12.6, 5.85);
foot(s, 'Same layout for the 3 extra people (1 run each). CH pulls the toes up at this angle, so its torque is positive.');
s.addNotes(N.hardestNotes2);

// ════════════════════════════ WRAP-UP ════════════════════════════
newSection('Wrap-up');
s = slide('Limits and next steps');
card(s, 0.5, 1.2, 7.3, 5.85, HEX.light);
txt(s, 'Limits', { x: 0.7, y: 1.32, w: 6.9, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
bullets(s, N.limitBullets, { x: 0.7, y: 1.8, w: 6.95, h: 5.2, fontSize: 13, gap: 6 });
card(s, 8.0, 1.2, 4.85, 5.85, HEX.paleOrange);
txt(s, 'Next steps', { x: 8.2, y: 1.32, w: 4.5, h: 0.4, fontSize: 17, bold: true, color: C.accent1 });
bullets(s, N.nextBullets, { x: 8.2, y: 1.8, w: 4.5, h: 5.2, fontSize: 13, gap: 6 });
s.addNotes(N.limitNotes);

N.sources.forEach((page, i) => {
  s = slide(N.sources.length > 1 ? `Sources (${i + 1}/${N.sources.length})` : 'Sources');
  bullets(s, page, { x: 0.5, y: 1.1, w: 12.35, h: 6.1, fontSize: 13, gap: 4 });
  s.addNotes('The sources behind each idea. None of them tested exactly our setup; they justify trying an idea, and our own runs decide whether it works here.');
});

s = slide('Takeaways', 'TITLE_DARK');
s.addText(N.takeaways.map((x, i) => ({ text: x, options: { bullet: true, breakLine: i < N.takeaways.length - 1 } })), { placeholder: 'body' });
s.addNotes('Three things to remember: the toes-down failure came from the data, not from the network; each of the five problems has a concrete fix we can explain; and we now check the model fairly, with one answer key, both directions, three runs and extra people.');

// ════════════════════════════ BACKUP ════════════════════════════
newSection('Backup', 'Backup slides', 'For questions: what else we tried, the other direction, numbers per person, words used');
N.tried.forEach((page, i) => {
  s = slide(`Backup · What else we tried (${i + 1}/${N.tried.length})`);
  s.addTable(page, { x: 0.5, y: 1.15, w: 12.35, colW: [2.6, 6.95, 2.8], fontFace: 'Calibri', fontSize: 11.5, valign: 'middle',
    border: { type: 'solid', pt: 0.75, color: 'D9D7D2' }, margin: 0.05, autoPage: false });
  s.addNotes('Backup: everything else we tried, in plain words, with what happened and whether we kept it.');
});
s = slide('Backup · Error per angle (learned from session 2 → tested on session 1)');
img(s, F2('results_error_S2toS1.png'), 0.35, 1.1, 12.6, 6.0);
s.addNotes('Backup: the error in Newton-metres for the other direction.');
s = slide('Backup · Numbers per person');
s.addTable(N.perSubjectRows, { x: 0.5, y: 1.15, w: 12.35, colW: [1.4, 1.8, 1.5, 1.5, 1.5, 1.5, 1.55, 1.6], rowH: 0.33, fontFace: 'Calibri', fontSize: 12,
  border: { type: 'solid', pt: 0.75, color: 'D9D7D2' }, margin: 0.04 });
foot(s, N.perSubjectFoot);
s.addNotes('Backup: every person, both directions, old vs new.');
s = slide('Backup · Words we use');
N.glossary.forEach((g, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = 0.5 + col * 6.2, y = 1.15 + row * 0.98;
  card(s, x, y, 6.05, 0.88);
  txt(s, g[0], { x: x + 0.15, y: y + 0.06, w: 1.85, h: 0.76, fontSize: 13, bold: true, color: C.text2, valign: 'middle' });
  txt(s, g[1], { x: x + 2.05, y: y + 0.06, w: 3.9, h: 0.76, fontSize: 11.5, valign: 'middle' });
});
s.addNotes('Backup: the words used in the talk.');

(async () => {
  await pres.writeFile({ fileName: OUTF });
  await applyTheme(OUTF, THEME);
  console.log('wrote', OUTF, 'slides', nSlides);
})();
