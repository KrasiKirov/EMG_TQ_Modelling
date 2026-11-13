// Builds notes/final_pipeline/presentation/EMG_torque_old_vs_new.pptx
const path = require('path');
const pptxgen = require('pptxgenjs');
const { applyTheme } = require(process.env.SKILL_DIR + '/scripts/apply_theme.js');

const WT = process.env.WT;
const FIG = path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures');
const OUTF = path.join(WT, 'notes', 'final_pipeline', 'presentation', 'EMG_torque_old_vs_new.pptx');
const fig = (n) => path.join(FIG, n);

const THEME = {
  name: 'EMG Torque',
  headFontFace: 'Calibri',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '1B1B1B', lt1: 'FFFFFF', dk2: '1F2A44', lt2: 'F2F1EE',
    accent1: 'EB6834', accent2: '2A78D6', accent3: '1BAF7A', accent4: 'C0392B',
    accent5: '8A8984', accent6: 'EDA100', hlink: '2A78D6', folHlink: '6D2E46',
  },
};
const HEX = { ink: '1B1B1B', navy: '1F2A44', light: 'F2F1EE', orange: 'EB6834', blue: '2A78D6', green: '1BAF7A',
              red: 'C0392B', muted: '8A8984', grid: 'E6E5E1', ink2: '52514E' };

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';            // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = 'Predicting voluntary ankle torque from muscle signals';
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

// ── layouts ──────────────────────────────────────────────────────────────────
pres.defineSlideMaster({
  title: 'TITLE_DARK',
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.0, w: 11.7, h: 1.9, fontSize: 40, bold: true,
                                color: C.background1, valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.15, w: 11.7, h: 1.6, fontSize: 20,
                                color: 'DDE3EE', valign: 'top', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: 'FFFFFF' },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.55, y: 0.3, w: 12.2, h: 0.85, fontSize: 32, bold: true,
                                color: C.text2, valign: 'middle', margin: 0 }, text: '' } },
  ],
  slideNumber: { x: 12.45, y: 7.0, w: 0.6, h: 0.3, fontSize: 11, color: '8A8984', align: 'right' },
});

let slideIdx = 0;
function newSlide(master, section) {
  slideIdx += 1;
  return pres.addSlide({ masterName: master, sectionTitle: section });
}
function txt(slide, text, opts) {
  slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: 'top' }, opts));
}
function pill(slide, x, y, label, color, w = 1.15) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.36, rectRadius: 0.18, fill: { color }, line: { color } });
  txt(slide, label, { x, y, w, h: 0.36, fontSize: 13, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
}
function card(slide, x, y, w, h, fill = HEX.light) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill } });
}
function circleNum(slide, x, y, n, color) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color }, line: { color } });
  txt(slide, String(n), { x, y, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
}
function bullets(slide, items, opts) {
  const runs = items.map((it, i) => {
    const r = typeof it === 'string' ? { text: it, options: {} } : it;
    r.options = Object.assign({ bullet: { indent: 16 }, paraSpaceAfter: 6 }, r.options);
    if (i < items.length - 1) r.options.breakLine = true;
    return r;
  });
  slide.addText(runs, Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: 'top' }, opts));
}

function labelText(slide, label, text, opts) {
  slide.addText([{ text: label, options: { bold: true, color: HEX.navy } }, { text }],
    Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: 'top' }, opts));
}

// ── 1. title ─────────────────────────────────────────────────────────────────
pres.addSection({ title: 'Intro' });
let s = newSlide('TITLE_DARK', 'Intro');
s.addText('Predicting voluntary ankle torque from muscle signals', { placeholder: 'title' });
s.addText([
  { text: 'What was wrong at the toes-down angles, what we changed, and how we check it now', options: { breakLine: true } },
  { text: 'Old pipeline (last semester) vs new pipeline  ·  October 2026', options: { fontSize: 16, color: 'AAB4C6' } },
], { placeholder: 'body' });
s.addNotes('Today: what the model does, why it failed last semester at the toes-down angles, what we changed, and how we check it now. About 10 minutes.');

// ── 2. the task ──────────────────────────────────────────────────────────────
s = newSlide('CONTENT', 'Intro');
s.addText('The task: muscle signals in, voluntary torque out', { placeholder: 'title' });
s.addImage({ path: fig('setup_angles.png'), x: 0.45, y: 1.45, w: 6.3, h: 4.17 });
txt(s, 'Angle convention from the lab code: negative = toes down (plantarflexion)', { x: 0.55, y: 5.75, w: 6.1, h: 0.5, fontSize: 12, color: HEX.ink2, italic: true });
const rows2 = [
  ['In', '4 EMG signals (calf: GM, GL, SOL; shin: TA) + the ankle angle, every 10 ms'],
  ['Out', 'Voluntary torque (Nm): the twist the muscles make, without the ankle’s passive “rubber band” torque'],
  ['Data', '6 people · 2 sessions in one visit (“test”, then “retest”) · 8 trials per session, one per angle, ~90 s of push / relax · recorded at 1000 Hz'],
  ['Task', 'Everyone pushed the toes down. Exception: CH pulled the toes UP at its 3–4 most toes-down angles (+10 to +30 Nm, shin muscle)'],
  ['Model', 'One model per person: learn on one session, predict the other session it never saw'],
];
rows2.forEach((r, i) => {
  const y = 1.45 + i * 1.08;
  card(s, 7.05, y, 5.75, 0.95);
  txt(s, r[0], { x: 7.25, y: y + 0.12, w: 0.9, h: 0.7, fontSize: 15, bold: true, color: C.accent1 });
  txt(s, r[1], { x: 8.15, y: y + 0.1, w: 4.5, h: 0.78, fontSize: 13.5 });
});
s.addNotes('The input is 4 muscle signals plus the ankle angle; the output is the torque the muscles produce voluntarily. Each person did two sessions in the same visit, 8 angles each, pushing and relaxing for 90 seconds. Negative angles are toes down: that is where the calf is short and weak. Everyone pushed the toes down, except CH, who pulled the toes up at its most toes-down angles; the torque data shows that clearly. Each person gets their own model: we train on one session and test on the other.');

// ── 3. last semester's problem ───────────────────────────────────────────────
pres.addSection({ title: 'The problem' });
s = newSlide('CONTENT', 'The problem');
s.addText('Last semester: broken only at the toes-down angles', { placeholder: 'title' });
s.addImage({ path: fig('problem_old_r2.png'), x: 1.2, y: 1.25, w: 10.94, h: 4.2 });
txt(s, [
  { text: 'R² = 1 − (squared error ÷ how much the torque varies).  ', options: { bold: true, color: HEX.navy } },
  { text: '1 = perfect · 0 = no better than always guessing the average · below 0 = worse than that' },
], { x: 0.7, y: 5.5, w: 11.9, h: 0.4, fontSize: 13.5, align: 'center' });
const stats3 = [
  ['0.97', 'strong angle: torque swings ±15 Nm, error 2.5 Nm'],
  ['0.44', 'toes-down angle: swings only ±0.4 Nm, error 0.3 Nm'],
  ['−13', 'old model there: error ~1.5 Nm on that ±0.4 Nm swing'],
];
stats3.forEach((st, i) => {
  const x = 0.7 + i * 4.05;
  card(s, x, 6.0, 3.85, 1.05);
  txt(s, 'R² ' + st[0], { x: x + 0.2, y: 6.07, w: 1.6, h: 0.9, fontSize: 24, bold: true, color: i === 2 ? C.accent4 : C.text2, valign: 'middle' });
  txt(s, st[1], { x: x + 1.75, y: 6.1, w: 1.95, h: 0.85, fontSize: 12, valign: 'middle' });
});
s.addNotes('Last semester the model was excellent at most angles but collapsed at the toes-down angles: negative R2 means worse than just guessing the average. Why so harsh? R2 compares the error with how much the torque moves. At toes-down angles the person can only produce about 1 Nm, so a tiny error destroys R2. That is why from now on we always also report the error in Nm.');

// ── 4. passive vs voluntary torque ───────────────────────────────────────────
s = newSlide('CONTENT', 'The problem');
s.addText('Voluntary torque = measured torque − passive torque', { placeholder: 'title' });
s.addImage({ path: fig('torque_split.png'), x: 0.4, y: 1.3, w: 7.95, h: 5.64 });
card(s, 8.6, 1.35, 4.25, 1.95, 'E3EDFB');
pill(s, 8.8, 1.48, 'OLD', HEX.blue);
txt(s, 'Separate relaxed recordings (passive trials): mean torque of each steady part → one number per angle, interpolated between angles, subtracted from every sample.', { x: 8.8, y: 1.93, w: 3.9, h: 1.3, fontSize: 13 });
card(s, 8.6, 3.42, 4.25, 2.5, 'FCE6DB');
pill(s, 8.8, 3.55, 'NEW', HEX.orange);
bullets(s, [
  'EMG says “relaxed”: the 4 muscles together below 3 % of their range for ≥ 0.5 s',
  'median torque of each relaxed stretch',
  'smooth line through them (8 s) = passive level inside this trial',
], { x: 8.8, y: 4.0, w: 3.9, h: 1.85, fontSize: 13 });
txt(s, 'Here: separate recording 10.6 Nm, real relaxed level 8.4–9.0 Nm and drifting. Passive torque relaxes over time and depends on what the ankle just did (Duong 2001; Kay & Blazevich 2009), so it is measured inside the trial. Torque is never used to decide “relaxed”; the model never sees this line.',
  { x: 8.6, y: 6.0, w: 4.25, h: 1.3, fontSize: 11.5, color: HEX.ink2 });
s.addNotes('The sensor measures everything, including the passive torque: the stretched ankle pulling back like a rubber band. We only want the muscle part, so voluntary torque equals measured minus passive. Last semester the passive value came from separate relaxed recordings: here 10.6 Nm. But inside the pushing trial, the relaxed level is only about 8.5 Nm and slowly drifts. So the old answer key was off by 2 Nm at an angle where the whole signal is 1 Nm. The new way measures the passive level inside the same trial: the EMG tells us when the person is relaxed, and at those moments the torque is purely passive; we draw a smooth line through them.');

// ── 5. five causes ───────────────────────────────────────────────────────────
s = newSlide('CONTENT', 'The problem');
s.addText('Five things broke the toes-down angles, and the fix for each', { placeholder: 'title' });
const causes = [
  ['Wrong passive level', 'separate relaxed recording ≠ relaxed level inside the trial (YES: 10.6 vs ~8.5 Nm)', 'measure it inside the trial'],
  ['Passive drift', 'the tissue slowly relaxes during the 90 s trial', 'the in-trial line follows the drift'],
  ['Bad passive table entry', 'JM: a movement segment got into the table (+11.7 Nm) → 12 Nm error at −0.078 rad', 'no table needed anymore'],
  ['EMG size changes between sessions', 'same push, smaller signal (JM calf ≈ 0.4× in session 2)', 'auto-gain: rescale each muscle by its recent range'],
  ['Training ignored small angles', 'plain MSE in Nm²: a strong angle weighs ~550× more than the weakest', 'divide each angle’s target by its typical swing'],
];
causes.forEach((c, i) => {
  const y = 1.4 + i * 1.12;
  circleNum(s, 0.6, y + 0.22, i + 1, HEX.navy);
  txt(s, c[0], { x: 1.3, y: y + 0.08, w: 3.4, h: 0.85, fontSize: 16, bold: true, valign: 'middle' });
  txt(s, c[1], { x: 4.75, y: y + 0.08, w: 4.6, h: 0.85, fontSize: 13.5, valign: 'middle' });
  card(s, 9.55, y + 0.08, 3.25, 0.85, 'FCE6DB');
  txt(s, '→ ' + c[2], { x: 9.7, y: y + 0.08, w: 3.0, h: 0.85, fontSize: 13.5, bold: true, color: C.accent1, valign: 'middle' });
});
s.addNotes('We traced the failure to five concrete causes, none of them in the network itself. One: the passive level was taken from the wrong recording. Two: it drifts during the trial. Three: for JM, a movement segment was accidentally accepted into the passive table, which shifted one angle by 12 Nm. Four: the EMG signal size changes between sessions for the same push. Five: the training loss was dominated by the strong angles. Each has a fix, shown in orange.');

// ── 6. old vs new pipeline ───────────────────────────────────────────────────
pres.addSection({ title: 'New pipeline' });
s = newSlide('CONTENT', 'New pipeline');
s.addText('Old vs new pipeline, step by step', { placeholder: 'title' });
s.addImage({ path: fig('pipeline_old_vs_new.png'), x: 0.45, y: 1.35, w: 12.45, h: 5.23 });
txt(s, [
  { text: 'Auto-gain uses only the past EMG of the session being predicted: no torque, nothing from the future.', options: { bold: true, color: HEX.navy, breakLine: true } },
  { text: 'Only the model step learns anything; every other box is fixed processing or checking.  Code: ML/research, run with --config FINAL' },
], { x: 0.55, y: 6.62, w: 11.7, h: 0.65, fontSize: 12.5, color: HEX.ink2 });
s.addNotes('Here is the whole pipeline. The EMG cleaning is unchanged: filter, absolute value, smoothing. What changed, in orange: the EMG scaling now has auto-gain, which rescales each muscle using only its recent past, and is switched off for a person who pushes both ways, like CH. The target uses the in-trial passive level. The model gets one small expert per angle. The loss gives every angle the same weight. And the checking is much stricter, which I will show in a moment.');

// ── 7. the model ─────────────────────────────────────────────────────────────
s = newSlide('CONTENT', 'New pipeline');
s.addText('The model: still an LSTM, plus one small expert per angle', { placeholder: 'title' });
s.addImage({ path: fig('fig05_arch_old.png'), x: 0.45, y: 1.3, w: 6.45, h: 2.49 });
s.addImage({ path: fig('fig06_arch_new.png'), x: 0.45, y: 3.95, w: 6.45, h: 3.04 });
card(s, 7.2, 1.35, 5.6, 5.9);
labelText(s, 'LSTM: ', 'reads the last 200 ms (20 steps \u00D7 5 numbers) one step at a time with a small memory, and sums it up in 16 numbers', { x: 7.4, y: 1.55, w: 5.2, h: 0.8 });
labelText(s, 'Experts: ', 'one extra output per training angle; the ankle angle picks which correction is added (and blends two between angles). Your \u201Cone model per angle\u201D idea, but sharing what the LSTM learns', { x: 7.4, y: 2.45, w: 5.2, h: 1.25 });
labelText(s, 'Training: ', '~57,000 windows (first 80 % of each training trial). Guess, measure the error (MSE), nudge the numbers (Nadam, batch 64); stop when the error on the last 20 % (validation) stops improving. ~5 min per model on a laptop CPU', { x: 7.4, y: 3.85, w: 5.2, h: 1.3 });
labelText(s, 'No memorising: ', '3,665 numbers vs ~57,000 windows, 20 % dropout, early stopping, and it is never trained on the tested session', { x: 7.4, y: 5.3, w: 5.2, h: 0.85 });
labelText(s, 'Most of the gain is the data, not the network: ', 'the original tiny LSTM with the new data steps reaches 0.917 (vs 0.925)', { x: 7.4, y: 6.3, w: 5.2, h: 0.8 });
s.addNotes('The model is still an LSTM: a network that reads the last 200 ms step by step and keeps a small memory. It is a bit wider now, and on top of it each training angle has a small expert: an extra output that adds a correction for that angle, chosen by the ankle angle. That is basically the idea of one model per angle, but all angles share the same LSTM. Training is the usual loop: guess, measure the error, adjust, and stop when the held-back validation part stops improving. Important: most of the improvement comes from the data processing; the old tiny LSTM with the new data steps is almost as good.');

// ── 8. how it is checked ─────────────────────────────────────────────────────
pres.addSection({ title: 'Checking' });
s = newSlide('CONTENT', 'Checking');
s.addText('How the model is checked: before vs now', { placeholder: 'title' });
card(s, 0.55, 1.35, 5.6, 4.4, 'E3EDFB');
pill(s, 0.75, 1.55, 'BEFORE', HEX.blue, 1.4);
bullets(s, [
  'Trained on session 1, tested on session 2 only',
  '1 training run',
  'R² against its own target: each pipeline graded with its own answer key',
  'R² only',
  'A wrong passive value shifted the answer key AND the prediction together, so the error was invisible',
], { x: 0.8, y: 2.15, w: 5.15, h: 3.4, fontSize: 15 });
card(s, 6.45, 1.35, 6.35, 4.4, 'FCE6DB');
pill(s, 6.65, 1.55, 'NOW', HEX.orange, 1.4);
bullets(s, [
  'Both directions: session 1 → 2 and session 2 → 1',
  '3 training runs, averaged (never the best one)',
  'One shared answer key for every model: the in-trial voluntary torque',
  'R² + typical error in Nm + “phantom” torque while relaxed',
  'Nothing from the tested session is used for training or choosing (auto-gain only reads its EMG as it arrives)',
  'Winner chosen by a written rule, then checked on 3 extra people (IES, CH, HM)',
  'Independent code review: 16 issues found and fixed before the final runs',
], { x: 6.7, y: 2.1, w: 5.9, h: 3.6, fontSize: 14.5 });
card(s, 0.55, 5.95, 12.25, 1.2, HEX.navy);
txt(s, 'Same old model, fair answer key: its average R² drops from 0.68 to 0.54. The old score was hiding failures (JM at −0.078 rad was 12 Nm off).',
  { x: 0.8, y: 6.05, w: 11.8, h: 1.0, fontSize: 16, bold: true, color: 'FFFFFF', valign: 'middle' });
s.addNotes('This is the second big change. Before, each pipeline was graded against its own estimate of the passive torque, so a pipeline with a wrong passive estimate was graded with a wrong answer key, and one direction, one run. Now every model is graded against the same answer key, in both directions, averaged over 3 training runs, and we also report the error in Nm and the fake torque predicted while the person is relaxed. The winner was chosen by a written rule and then checked on three people not used for choosing. An independent code review found 16 bugs, all fixed. With the fair answer key, the old model actually looks worse than we thought: it was hiding a 12 Nm error.');

// ── 9. results ───────────────────────────────────────────────────────────────
pres.addSection({ title: 'Results' });
s = newSlide('CONTENT', 'Results');
s.addText('Results: the toes-down angles are fixed', { placeholder: 'title' });
const chartOpts = (title) => ({
  barDir: 'col', barGrouping: 'clustered', showTitle: true, title, titleFontSize: 14, titleColor: HEX.navy, titleFontFace: '+mn-lt',
  showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 12, dataLabelFormatCode: '0.00', dataLabelColor: HEX.ink, dataLabelFontFace: '+mn-lt',
  chartColors: [HEX.navy, HEX.orange], valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisMajorUnit: 0.25, valAxisLabelFontSize: 11, valAxisLabelColor: HEX.ink2,
  valAxisLabelFontFace: '+mn-lt', catAxisLabelFontSize: 12, catAxisLabelColor: HEX.ink, catAxisLabelFontFace: '+mn-lt', valGridLine: { color: HEX.grid, size: 0.75 },
  catGridLine: { style: 'none' }, showLegend: true, legendPos: 'b', legendFontSize: 12, legendFontFace: '+mn-lt', legendColor: HEX.ink,
});
s.addChart(pres.charts.BAR, [
  { name: 'all angles (mean R²)', labels: ['Old', 'Report-1 final', 'FINAL'], values: [0.540, 0.925, 0.925] },
  { name: '2 most toes-down angles', labels: ['Old', 'Report-1 final', 'FINAL'], values: [0.04, 0.85, 0.84] },
], Object.assign({ x: 0.5, y: 1.25, w: 6.1, h: 4.1 }, chartOpts('Main subjects YES, JM, EG (3 runs × 2 directions)')));
s.addChart(pres.charts.BAR, [
  { name: 'all angles (mean R²)', labels: ['Old', 'Report-1 final', 'FINAL'], values: [0.878, 0.870, 0.967] },
  { name: '2 most toes-down angles', labels: ['Old', 'Report-1 final', 'FINAL'], values: [0.70, 0.66, 0.97] },
], Object.assign({ x: 6.75, y: 1.25, w: 6.1, h: 4.1 }, chartOpts('3 extra people IES, CH, HM (1 run each)')));
const stats9 = [
  ['−14.3 → 0.34', 'YES −0.486 rad (error 1.52 → 0.31 Nm)'],
  ['−0.99 → 0.90', 'JM −0.787 rad, trained S2 → tested S1'],
  ['8 → 0', 'angles with negative R² (out of 48)'],
  ['1.25 → 0.17 Nm', 'phantom torque while relaxed'],
];
stats9.forEach((st, i) => {
  const x = 0.55 + i * 3.08;
  card(s, x, 5.5, 2.92, 1.08);
  txt(s, st[0], { x: x + 0.18, y: 5.56, w: 2.65, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  txt(s, st[1], { x: x + 0.18, y: 6.07, w: 2.65, h: 0.45, fontSize: 12 });
});
txt(s, [
  { text: 'FINAL beats the old pipeline at 46 / 48 single-angle results (main) and 42 / 48 (extra people); the others are small drops at middle / toes-up angles, never toes-down (largest 0.92 → 0.83).', options: { breakLine: true } },
  { text: 'Extra people: 1 run each; CH’s one angle where the task flipped between sessions is left out for every pipeline. FINAL on all angles (auto-gain off for CH): 0.966.' },
], { x: 0.55, y: 6.66, w: 11.75, h: 0.6, fontSize: 11, color: HEX.ink2 });
s.addNotes('On the three main subjects, the new pipeline reaches a mean R2 of 0.925, and at the two most toes-down angles it goes from about 0 to 0.84. On the three extra people, checked afterwards with one run each, the toes-down angles reach 0.97. Report 1’s version fails there for subject IES, who has no passive recordings at all, because its answer key needs one; the new in-trial passive level does not. Concrete examples: YES at minus 0.486 goes from minus 14 to 0.34, and no angle has a negative R2 anymore. The model also stops inventing torque while the person is relaxed: 1.25 Nm down to 0.17 Nm. And nothing got meaningfully worse: 46 of 48 single-angle results improved, the other two dropped by less than 0.01.');

// ── 10. full trial inspection before / after ─────────────────────────────────
s = newSlide('CONTENT', 'Results');
s.addText('Before / after: full trial inspection, YES (trained S1 → tested S2)', { placeholder: 'title' });
pill(s, 0.6, 1.3, 'OLD', HEX.blue);
s.addImage({ path: fig('trial_YES_S1toS2_OLD.png'), x: 0.55, y: 1.75, w: 3.25, h: 5.62 });
pill(s, 4.15, 1.3, 'FINAL', HEX.orange);
s.addImage({ path: fig('trial_YES_S1toS2_FINAL.png'), x: 4.1, y: 1.75, w: 3.25, h: 5.62 });
card(s, 7.75, 1.3, 5.05, 5.95);
txt(s, 'How to read it', { x: 7.95, y: 1.45, w: 4.7, h: 0.45, fontSize: 17, bold: true, color: C.text2 });
bullets(s, [
  '8 panels = the 8 trials (angles) of the tested session; the top panel is the most toes-down angle',
  'Blue = truth (voluntary torque), red = the model’s prediction; 0 = relaxed, negative = pushing',
  'Panel title: angle, R², error in Nm',
  'Old, top panel: right timing but the whole trace sits ~1.5 Nm too low (wrong passive level) → R² −14',
  'FINAL: follows the pushes and the relaxed periods at every angle',
  'All 6 people × both directions, old and FINAL: notes/final_pipeline/trial_plots/',
], { x: 7.95, y: 2.0, w: 4.7, h: 5.15, fontSize: 14 });
s.addNotes('These are the same full trial inspection plots as last semester, for YES, scored against the fair answer key. Each panel is one angle; the top one is the hardest, toes-down angle. On the left, the old pipeline: at the top the prediction has the right timing but sits about 1.5 Nm too low the whole time; that is the wrong passive level. On the right, the new pipeline follows the pushes and the relaxed periods. The plots for all six people and both directions are in the trial plots folder.');

// ── 11. hardest angles up close ──────────────────────────────────────────────
s = newSlide('CONTENT', 'Results');
s.addText('The hardest angles up close: old vs FINAL', { placeholder: 'title' });
s.addImage({ path: fig('overlays_hard_angles.png'), x: 0.45, y: 1.3, w: 6.72, h: 6.02 });
card(s, 7.5, 1.3, 5.3, 5.95);
bullets(s, [
  { text: 'Black = truth, blue = old, orange = FINAL', options: { bold: true } },
  'Old: the right shape but the wrong level (passive torque) and often the wrong size (EMG signal size changed between sessions)',
  'FINAL: tracks both the pushes and the relaxed periods',
  'HM is one of the 3 extra people, not part of the main comparison',
  'Still weakest: YES −0.486 (top). The person only pushes ~1 Nm there, so an error of ~0.3 Nm still limits R² (0.34 averaged over 3 runs)',
], { x: 7.7, y: 1.5, w: 4.9, h: 5.6, fontSize: 14.5 });
s.addNotes('Zooming in on the hardest angle of four people. Black is the truth, blue the old pipeline, orange the new one. The old pipeline gets the shape but the level and often the size wrong. The new one follows the pushes. HM is one of the three extra people. Honestly, YES at minus 0.486 is still the weakest: the person only pushes about 1 Nm there, so even a 0.3 Nm error limits R2.');

// ── 12. what did not help ────────────────────────────────────────────────────
pres.addSection({ title: 'Wrap-up' });
s = newSlide('CONTENT', 'Wrap-up');
s.addText('What we tried that did not beat it', { placeholder: 'title' });
const tried = [
  ['Ridge regression (simple linear)', 0.624], ['Minimal fix, no auto-gain', 0.826], ['Ridge + auto-gain', 0.863],
  ['Pre-train on the 5 other people', 0.885], ['EMG ÷ max effort (MVC)', 0.887], ['EMG ÷ MVC at that angle', 0.904],
  ['×10 weight on toes-down angles', 0.916], ['Passive from the hold before each trial', 0.921], ['Passive drift model', 0.921],
  ['EMG gain augmentation', 0.922], ['×3 weight on toes-down angles', 0.922], ['Relaxed periods removed from training', 0.924],
  ['FINAL', 0.925], ['400 ms window (+0.007, strong angles only)', 0.932],
];
s.addChart(pres.charts.BAR, [{ name: 'mean R²', labels: tried.map(t => t[0]), values: tried.map(t => t[1]) }], {
  x: 0.45, y: 1.25, w: 7.6, h: 6.0, barDir: 'bar', chartColors: [HEX.navy], showValue: true, dataLabelPosition: 'outEnd',
  dataLabelFormatCode: '0.000', dataLabelFontSize: 11, dataLabelColor: HEX.ink, dataLabelFontFace: '+mn-lt',
  valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, valAxisLabelFontSize: 11, valAxisLabelColor: HEX.ink2, valAxisLabelFontFace: '+mn-lt',
  catAxisLabelFontSize: 11.5, catAxisLabelColor: HEX.ink, catAxisLabelFontFace: '+mn-lt', valGridLine: { color: HEX.grid, size: 0.75 },
  catGridLine: { style: 'none' }, showLegend: false, showTitle: true, title: 'Mean R², main subjects (3 runs × 2 directions)', titleFontSize: 13,
  titleColor: HEX.navy, titleFontFace: '+mn-lt',
});
card(s, 8.35, 1.3, 4.45, 5.95);
bullets(s, [
  { text: 'Auto-gain is the biggest single fix (+0.09)', options: { bold: true } },
  'Extra tricks on top add ≤ 0.01',
  '×3 extra weight on toes-down angles ties; ×10 is worse',
  'MVC scaling is worse: the MVC was recorded once, the EMG size changes between sessions',
  'Pre-training on other people is worse: people differ too much',
  'Removing the relaxed periods gives the same R² but 45 % more phantom torque: relaxed periods are useful, not noise',
], { x: 8.55, y: 1.5, w: 4.05, h: 5.6, fontSize: 14 });
s.addNotes('We tested 23 configurations, each with 3 runs in both directions, and these are the main ones. The simple linear model is far behind. Auto-gain is the single biggest fix. Using the passive holds, MVC scaling, extra weight on the toes-down angles, augmentation, or pre-training on other people did not beat it. The 400 millisecond window gave a tiny gain on the strong angles only. And removing the relaxed periods from training does not help: the model then predicts fake torque while you relax, so those periods are useful.');

// ── 13. limits and next steps ────────────────────────────────────────────────
s = newSlide('CONTENT', 'Wrap-up');
s.addText('Honest limits and next steps', { placeholder: 'title' });
card(s, 0.55, 1.35, 7.3, 4.4);
txt(s, 'Limits', { x: 0.8, y: 1.5, w: 6.8, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
bullets(s, [
  '6 people; the two sessions are in the same visit, so a different-day test would probably be harder',
  'YES −0.486 is still the weakest cell (R² 0.34, ~0.3 Nm error on a ~1 Nm push)',
  'FINAL was picked after seeing the 3 extra people (the written rule had picked the 400 ms variant), so they are not fully independent proof',
  'The answer key and FINAL’s labels use the same relaxed-moment idea. Check: report 1’s pipeline, whose labels are built differently, scores the same 0.925 on it',
  'The answer key assumes torque is purely passive while the EMG is quiet; with few relaxed moments it is less certain (JM session 2, −0.50 rad: relaxed 0.9 % of the time)',
  'The EMG filters use ~0.25 s of future signal: fine offline, not yet real-time',
], { x: 0.8, y: 2.0, w: 6.85, h: 3.7, fontSize: 14 });
card(s, 8.15, 1.35, 4.65, 4.4, 'FCE6DB');
txt(s, 'Next steps', { x: 8.4, y: 1.5, w: 4.2, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
bullets(s, [
  'Record a session on a different day, or a new person, to confirm cleanly',
  'One-directional (causal) filters for real-time use',
  'Try FINAL with a 400 ms window',
  'Re-check with a different rest detector and smoothing',
  'Average the 3 runs’ models: +0.006 R² for free',
  'Move FINAL into the main training script and save the trained model',
], { x: 8.4, y: 2.0, w: 4.2, h: 3.7, fontSize: 14 });
card(s, 0.55, 5.95, 12.25, 1.2, HEX.navy);
txt(s, 'Bottom line: for these 6 people, the toes-down problem is fixed when the model is tested on the other session of the same visit. A different-day recording is the one test still missing.',
  { x: 0.8, y: 6.05, w: 11.8, h: 1.0, fontSize: 16, bold: true, color: 'FFFFFF', valign: 'middle' });
s.addNotes('To be honest about the limits: six people, and both sessions were in the same visit. YES at minus 0.486 is improved but still the weakest angle. The final choice was made after looking at the three extra people. The answer key uses the same relaxed-moment idea as the new labels; as a check, report 1’s pipeline, which builds its labels differently, scores exactly the same on it, so the key is not simply rewarding FINAL. The key also assumes that quiet EMG means zero voluntary torque, which is less certain when someone is almost never relaxed. And the filters look slightly into the future, so this is offline for now. Next steps: a different-day session, causal filters, and moving this into the main training script.');

// ── 14. closing ──────────────────────────────────────────────────────────────
s = newSlide('TITLE_DARK', 'Wrap-up');
s.addText('Takeaways', { placeholder: 'title' });
s.addText([
  { text: 'The toes-down failure came from the data (passive level, EMG size, loss weighting), not the network', options: { bullet: true, breakLine: true } },
  { text: 'New pipeline: in-trial passive level + auto-gain + one expert per angle → toes-down R² ≈ 0 → 0.84 (0.97 on 3 extra people)', options: { bullet: true, breakLine: true } },
  { text: 'Checked fairly: one shared answer key, both directions, 3 runs, extra people, code review', options: { bullet: true, breakLine: true } },
  { text: 'Details: notes/report2/round2_report.md · notes/final_pipeline/final_code_pipeline.md · code ML/research (--config FINAL)', options: { fontSize: 14, color: 'AAB4C6' } },
], { placeholder: 'body' });
s.addNotes('Three things to remember: the failure was in the data, not the network; the new pipeline fixes the toes-down angles; and we checked it fairly. Everything, including the full report, the step-by-step document and the code, is in the repository.');

// ── backup slides (not presented; for questions) ────────────────────────────
pres.addSection({ title: 'Backup' });

// B1. rest detection step by step
s = newSlide('CONTENT', 'Backup');
s.addText('Backup · How the passive line is found, step by step', { placeholder: 'title' });
s.addImage({ path: fig('rest_detection_steps.png'), x: 0.35, y: 1.3, w: 7.95, h: 6.0 });
card(s, 8.5, 1.3, 4.3, 5.95);
bullets(s, [
  'Activity = average over the 4 muscles of (EMG − its quietest level in this trial) ÷ (its range over the whole session)',
  'Relaxed starts below 0.03 and ends above 0.06 (two thresholds, so it does not flicker)',
  'Keep stretches ≥ 0.5 s; skip their first 0.3 s (torque settles after the muscle switches off) and last 0.1 s (the smoothed EMG rises just before a push)',
  'Median torque per stretch; outlier stretches dropped; smooth line (σ = 8 s), flat before the first and after the last stretch',
  'Every trial had ≥ 2 stretches (median 15, ~24 % of the time relaxed). Fewest: JM session 2, −0.50 rad (0.9 %)',
  'Training labels: training session only. Test answer key: same recipe on the tested session, never given to the model',
  'Basis: passive torque relaxes and depends on recent history (Duong 2001; McNair 2001; Kay & Blazevich 2009). Anchoring it on relaxed moments is our own correction, not a published method',
], { x: 8.7, y: 1.45, w: 3.95, h: 5.7, fontSize: 11.5 });
s.addNotes('Backup. The exact recipe behind the new passive level, shown on the same YES trial as slide 4. Step 1 uses EMG only: when the average activity of the 4 muscles drops below 3 % of the session range, the person is relaxed, until it rises above 6 %. Step 2: in each relaxed stretch the torque is purely passive, so we take its median. Step 3: a smooth line through those dots is the passive level at every moment. Step 4: voluntary torque is measured minus that line, so it is 0 when relaxed.');

// B2. auto-gain
s = newSlide('CONTENT', 'Backup');
s.addText('Backup · Auto-gain, and why it is off for CH', { placeholder: 'title' });
s.addImage({ path: fig('agc_before_after.png'), x: 0.45, y: 1.25, w: 9.1, h: 3.38 });
card(s, 9.8, 1.3, 3.0, 3.3, HEX.navy);
txt(s, '0.40× → 0.74×', { x: 9.95, y: 1.45, w: 2.75, h: 0.55, fontSize: 22, bold: true, color: 'FFFFFF' });
txt(s, 'JM, session 2 vs 1, same push: auto-gain closes most of the gap, not all.', { x: 9.95, y: 2.05, w: 2.7, h: 1.0, fontSize: 12, color: 'DDE3EE' });
txt(s, '+0.09 R²', { x: 9.95, y: 3.05, w: 2.75, h: 0.5, fontSize: 22, bold: true, color: C.accent1 });
txt(s, 'biggest single fix (0.826 → 0.917, same small LSTM)', { x: 9.95, y: 3.6, w: 2.7, h: 0.8, fontSize: 12, color: 'DDE3EE' });
card(s, 0.55, 4.85, 6.0, 2.4);
txt(s, 'How it works (per muscle)', { x: 0.75, y: 4.95, w: 5.6, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
bullets(s, [
  'Every 0.5 s: floor = 10th percentile of the past 20 s, top = 95th percentile of the past 45 s',
  'Output = (EMG − floor) ÷ (top − floor), kept between 0 and 2',
  'Starts from the training session’s values; the range never shrinks below 0.35× of them; electrode spikes are held out',
  'No torque and no future samples: it can run live',
], { x: 0.75, y: 5.33, w: 5.65, h: 1.85, fontSize: 12 });
card(s, 6.75, 4.85, 6.05, 2.4, 'FCE6DB');
txt(s, 'Switched off when someone pushes both ways', { x: 6.95, y: 4.95, w: 5.7, h: 0.35, fontSize: 14, bold: true, color: C.accent1 });
bullets(s, [
  'Each training trial: torque when the EMG is high (top 20 %) − torque when it is low (bottom 20 %)',
  'Some trials above +2 Nm and others below −2 Nm → auto-gain OFF. Only CH',
  'Why: rescaling each muscle to its own range hides which muscle (calf or shin) is working. CH’s flipped angle: R² −3.3 / −5.5 with auto-gain, 0.98 / 0.95 without',
], { x: 6.95, y: 5.33, w: 5.7, h: 1.85, fontSize: 12 });
s.addNotes('Backup. Auto-gain fixes the fact that the same push gives a different EMG size in the second session. For JM at minus 0.64, the calf signal rises only 0.4 times as much in session 2; after auto-gain, 0.74 times. It rescales each muscle by its recent floor and top, from past EMG only, and starts from the training values. It is switched off when the training data has pushes in both directions, which only happens for CH: there it hid which muscle was working.');

// B3. every subject
s = newSlide('CONTENT', 'Backup');
s.addText('Backup · Every subject: old vs FINAL', { placeholder: 'title' });
const H1 = { bold: true, color: 'FFFFFF', fill: { color: HEX.navy }, align: 'center', valign: 'middle', fontSize: 12 };
const tRows = [
  [{ text: 'Subject', options: Object.assign({ rowspan: 2 }, H1) }, { text: 'Trained → tested', options: Object.assign({ rowspan: 2 }, H1) },
   { text: 'R², all 8 angles', options: Object.assign({ colspan: 2 }, H1) }, { text: 'R², 2 most toes-down', options: Object.assign({ colspan: 2 }, H1) },
   { text: 'Error at 2 most toes-down (Nm)', options: Object.assign({ colspan: 2 }, H1) }],
  ['old', 'FINAL', 'old', 'FINAL', 'old', 'FINAL'].map(t => ({ text: t, options: Object.assign({}, H1, { fill: { color: t === 'old' ? HEX.blue : HEX.orange } }) })),
];
const data = [
  ['YES', 'S1 → S2', '0.605', '0.885', '−0.43', '0.58', '1.79', '0.63'],
  ['YES', 'S2 → S1', '0.697', '0.942', '−0.08', '0.81', '1.41', '0.56'],
  ['JM', 'S1 → S2', '0.336', '0.894', '0.34', '0.91', '0.54', '0.21'],
  ['JM', 'S2 → S1', '−0.064', '0.885', '−0.72', '0.85', '0.94', '0.29'],
  ['EG', 'S1 → S2', '0.789', '0.972', '0.33', '0.97', '1.97', '0.50'],
  ['EG', 'S2 → S1', '0.880', '0.969', '0.82', '0.95', '0.97', '0.49'],
  ['Main mean', '3 runs each', '0.540', '0.925', '0.04', '0.84', '1.27', '0.45'],
  ['IES', 'S1 → S2', '0.912', '0.984', '0.92', '0.98', '0.89', '0.45'],
  ['IES', 'S2 → S1', '0.730', '0.982', '0.24', '0.98', '2.52', '0.46'],
  ['CH', 'S1 → S2', '0.976', '0.988', '0.96', '0.99', '1.18', '0.69'],
  ['CH', 'S2 → S1', '0.945', '0.983', '0.94', '0.98', '1.53', '0.75'],
  ['HM', 'S1 → S2', '0.850', '0.950', '0.52', '0.94', '0.58', '0.29'],
  ['HM', 'S2 → S1', '0.830', '0.911', '0.62', '0.95', '0.71', '0.26'],
  ['Extra mean', '1 run each', '0.874', '0.966', '0.70', '0.97', '1.23', '0.48'],
];
data.forEach(r => {
  const mean = r[0].includes('mean');
  tRows.push(r.map((t, j) => ({ text: t, options: {
    align: j < 2 ? 'left' : 'center', valign: 'middle', fontSize: 12, bold: mean || (j >= 2 && j % 2 === 1),
    color: j >= 2 && j % 2 === 1 ? HEX.orange : (t.startsWith('−') ? HEX.red : HEX.ink),
    fill: { color: mean ? 'E9E7E2' : 'FFFFFF' } } })));
});
s.addTable(tRows, { x: 0.55, y: 1.3, w: 12.25, colW: [1.55, 1.6, 1.5, 1.5, 1.5, 1.5, 1.55, 1.55], rowH: 0.33,
  border: { type: 'solid', pt: 0.75, color: 'D9D7D2' }, fontFace: THEME.bodyFontFace, margin: 0.04 });
txt(s, [
  { text: 'Paired Wilcoxon test, main subjects: FINAL beats old in 12 / 12 toes-down results (mean +0.80 R², p = 0.0005) and 46 / 48 overall (p < 10⁻¹²), uncorrected. 3 people: this describes them, not people in general.', options: { bold: true, color: HEX.navy, breakLine: true } },
  { text: 'Scored against the shared answer key. Per-angle R² averaged over the runs, floored at −1, then over the angles. Error = RMSE. FINAL = rest-anchored labels + auto-gain (off for CH) + experts + per-angle scaled loss. Red = negative R².' },
], { x: 0.55, y: 6.66, w: 11.75, h: 0.66, fontSize: 10.5, color: HEX.ink2 });
s.addNotes('Backup. Every subject in both directions. The old pipeline has negative toes-down R² for YES and JM; FINAL is at least 0.58 everywhere at the toes-down angles, and its toes-down error is 0.2 to 0.75 Nm instead of up to 2.5 Nm.');

// B4. glossary
s = newSlide('CONTENT', 'Backup');
s.addText('Backup · Words and numbers', { placeholder: 'title' });
const gloss = [
  ['R²', '1 − squared error ÷ variance of the truth. 1 = perfect, 0 = always guessing the average, < 0 = worse'],
  ['Error (RMSE)', 'typical size of the miss, in Nm (square root of the mean squared error)'],
  ['Phantom torque', 'average size of the prediction while the person is relaxed; should be 0'],
  ['MSE (the loss)', 'mean squared error: what training makes small. FINAL first divides each angle’s target by its typical swing'],
  ['LSTM', 'a layer that reads the 20 time steps in order and keeps a small memory between steps'],
  ['Mixture of experts', 'one shared network + one small correction per training angle, chosen by the ankle angle'],
  ['Dropout (20 %)', 'randomly hides some numbers during training so the model cannot memorise'],
  ['Validation, early stopping', 'last 20 % of each training trial held back; stop 15 epochs after it last improved; keep the best'],
  ['Epoch', 'one pass over all training windows (FINAL: 17–73, median ~28)'],
  ['Run / seed', 'the same training from a different random start; we average 3, never pick the best'],
  ['Auto-gain (AGC)', 'rescales each muscle by its recent range, using only past EMG'],
  ['MVC', 'maximum voluntary contraction: the strongest push, recorded once at the start of the visit'],
];
gloss.forEach((g, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = 0.55 + col * 6.2, y = 1.35 + row * 0.97;
  card(s, x, y, 6.0, 0.86);
  txt(s, g[0], { x: x + 0.18, y: y + 0.08, w: 1.75, h: 0.7, fontSize: 13, bold: true, color: C.text2, valign: 'middle' });
  txt(s, g[1], { x: x + 1.98, y: y + 0.08, w: 3.88, h: 0.7, fontSize: 12, valign: 'middle' });
});
s.addNotes('Backup. Definitions of every term used in the talk.');

(async () => {
  await pres.writeFile({ fileName: OUTF });
  await applyTheme(OUTF, THEME);
  console.log('wrote', OUTF, 'slides', slideIdx);
})();
