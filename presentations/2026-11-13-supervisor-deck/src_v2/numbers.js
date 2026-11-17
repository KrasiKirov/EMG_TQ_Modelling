// Text and numbers for the v2 deck. Numbers come from computed.json (result files) or from the
// figure/research agents' reports (marked with the figure they come from).
const CP = require('./computed.json');
const HEX = { navy: '1F2A44', orange: 'EB6834', blue: '2A78D6', red: 'C0392B', light: 'F2F1EE', ink: '1B1B1B' };
const b = (text, extra) => ({ text, options: Object.assign({ bold: true, color: HEX.navy }, extra || {}) });
const t = (text) => ({ text, options: {} });
const f = (v, d = 2) => (v < 0 && Math.abs(v).toFixed(d) !== (0).toFixed(d) ? '−' : '') + Math.abs(v).toFixed(d);
const N = {};
const MAIN = ['YES', 'JM', 'EG'], EXTRA = ['IES', 'CH', 'HM'], PROTS = ['S1toS2', 'S2toS1'];
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const grp = (subs, who, key) => mean(subs.flatMap(sj => PROTS.map(p => CP.per_subject[`${sj}|${p}`][who][key])));

// ── per-subject backup table
const H1 = { bold: true, color: 'FFFFFF', fill: { color: HEX.navy }, align: 'center', valign: 'middle', fontSize: 12 };
N.perSubjectRows = [
  [{ text: 'Person', options: Object.assign({ rowspan: 2 }, H1) }, { text: 'Learned → tested', options: Object.assign({ rowspan: 2 }, H1) },
   { text: 'R², all 8 angles', options: Object.assign({ colspan: 2 }, H1) }, { text: 'R², 2 most toes-down', options: Object.assign({ colspan: 2 }, H1) },
   { text: 'Error, 2 most toes-down (Nm)', options: Object.assign({ colspan: 2 }, H1) }],
  ['old', 'new', 'old', 'new', 'old', 'new'].map(x => ({ text: x, options: Object.assign({}, H1, { fill: { color: x === 'old' ? HEX.blue : HEX.orange } }) })),
];
const cell = (x, j, bold) => ({ text: x, options: { align: j < 2 ? 'left' : 'center', valign: 'middle', fontSize: 12, bold: bold || (j >= 2 && j % 2 === 1),
  color: j >= 2 && j % 2 === 1 ? HEX.orange : (x.startsWith('−') ? HEX.red : HEX.ink), fill: { color: bold ? 'E9E7E2' : 'FFFFFF' } } });
[[MAIN, 'Main 3 · average', '3 runs each'], [EXTRA, 'Extra 3 · average', '1 run each']].forEach(([subs, lab, runs]) => {
  subs.forEach(sj => PROTS.forEach(p => {
    const r = CP.per_subject[`${sj}|${p}`];
    const row = [sj, p === 'S1toS2' ? 'session 1 → 2' : 'session 2 → 1', f(r.old.r2_all, 3), f(r.new.r2_all, 3), f(r.old.r2_pf2), f(r.new.r2_pf2), f(r.old.err_pf2), f(r.new.err_pf2)];
    N.perSubjectRows.push(row.map((x, j) => cell(x, j, false)));
  }));
  const row = [lab, runs, f(grp(subs, 'old', 'r2_all'), 3), f(grp(subs, 'new', 'r2_all'), 3), f(grp(subs, 'old', 'r2_pf2')), f(grp(subs, 'new', 'r2_pf2')),
               f(grp(subs, 'old', 'err_pf2')), f(grp(subs, 'new', 'err_pf2'))];
  N.perSubjectRows.push(row.map((x, j) => cell(x, j, true)));
});
N.perSubjectFoot = 'Graded with the shared answer key. Per-angle R² averaged over the runs, floored at −1, then averaged over the angles. Error = typical size of the miss (RMSE). Red = negative R².';
N.mainOld = { r2: grp(MAIN, 'old', 'r2_all'), pf2: grp(MAIN, 'old', 'r2_pf2'), err: grp(MAIN, 'old', 'err_pf2'), ph: grp(MAIN, 'old', 'phantom') };
N.mainNew = { r2: grp(MAIN, 'new', 'r2_all'), pf2: grp(MAIN, 'new', 'r2_pf2'), err: grp(MAIN, 'new', 'err_pf2'), ph: grp(MAIN, 'new', 'phantom') };
N.extraOld = { r2: grp(EXTRA, 'old', 'r2_all'), pf2: grp(EXTRA, 'old', 'r2_pf2'), err: grp(EXTRA, 'old', 'err_pf2'), ph: grp(EXTRA, 'old', 'phantom') };
N.extraNew = { r2: grp(EXTRA, 'new', 'r2_all'), pf2: grp(EXTRA, 'new', 'r2_pf2'), err: grp(EXTRA, 'new', 'err_pf2'), ph: grp(EXTRA, 'new', 'phantom') };

N.glossary = [
  ['EMG', 'electrical activity of a muscle, picked up by electrodes on the skin. We use its smoothed size (“envelope”).'],
  ['Passive torque', 'what the torque sensor reads when the muscles are switched off (stretched tissues + weight of the foot).'],
  ['Voluntary torque', 'measured torque − passive torque: the part the muscles make. This is what the model predicts.'],
  ['R²', 'as in curve fitting: 1 = perfect, 0 = no better than a flat line at the average, below 0 = worse than that.'],
  ['Error (RMSE)', 'typical size of the miss, in Nm (square root of the mean squared error).'],
  ['Phantom torque', 'average size of the torque the model predicts while the person is relaxed. Should be 0.'],
  ['Fitting / training', 'adjusting the model’s numbers so its output matches the data (least squares).'],
  ['Training run', 'one complete fit, started from random numbers. We do 3 and average them.'],
  ['LSTM', 'a network that reads the 0.2 s window step by step and keeps a small running memory.'],
  ['Per-angle correction', '16 extra multipliers per angle added to the shared read-out (“experts”).'],
  ['Auto-gain', 'rescales each muscle signal by its own recent quiet level and strong-push level, using only the past.'],
  ['Typical swing', 'the standard deviation of the voluntary torque at one angle in the training data.'],
];

module.exports = { N, CP, b, t, f, mean, grp };
