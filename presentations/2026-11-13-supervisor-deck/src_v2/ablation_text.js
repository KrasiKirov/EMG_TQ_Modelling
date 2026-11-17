// Ablation text (numbers from computed.json; 1 training run per test, all 6 people, both directions).
const { f, CP } = require('./numbers.js');
const FN = require('./fignum.js');
const V = FN.ablation.vals;
const M = ['YES', 'JM', 'EG'], X = ['IES', 'CH', 'HM'];
const g = (k, s, key) => V(k, s, key);
const fmt = (v, d = 2) => (v === null || v === undefined ? '…' : f(v, d));
const one = (k, sp) => CP.ablation[k][sp] ? CP.ablation[k][sp].r2_all : null;
const b = (text) => ({ text, options: { bold: true, color: '1F2A44' } });
const t = (text) => ({ text, options: {} });
module.exports = {
  bullets: [
    [b('How to read: '), t('each row takes ONE change out of the new pipeline and keeps everything else. The drop compared with the top row is what that change contributes. 1 training run each (so the top row differs slightly from the 3-run averages elsewhere); differences under about 0.01–0.02 are noise.')],
    [b('Passive level measured inside the trial = the biggest contributor: '), t(`with the old table instead, the main 3 fall from ${fmt(g('new', M, 'r2_all'), 2)} to ${fmt(g('E34_final_oldpassive', M, 'r2_all'), 2)} and the toes-down angles from ${fmt(g('new', M, 'r2_pf2'))} to ${fmt(g('E34_final_oldpassive', M, 'r2_pf2'))}; phantom torque goes back up to ${fmt(g('E34_final_oldpassive', M, 'phantom'))} Nm.`)],
    [b('Auto-gain: '), t(`without it the main 3 fall to ${fmt(g('E32_restanchor_noagc_moe', M, 'r2_all'))}, mostly JM (learned from session 1: ${fmt(one('new', 'JM|S1toS2'))} → ${fmt(one('E32_restanchor_noagc_moe', 'JM|S1toS2'))}) and EG (learned from session 2: ${fmt(one('new', 'EG|S2toS1'))} → ${fmt(one('E32_restanchor_noagc_moe', 'EG|S2toS1'))}). For YES and HM it is slightly better without it, so it is not free.`)],
    [b('Dividing by the typical swing: '), t(`without it the toes-down angles of the main 3 fall from ${fmt(g('new', M, 'r2_pf2'))} to ${fmt(g('E33_final_noscale', M, 'r2_pf2'))} (YES learned from session 1: average R² ${fmt(one('new', 'YES|S1toS2'))} → ${fmt(one('E33_final_noscale', 'YES|S1toS2'))}).`)],
    [b('New network: '), t(`the old small network inside the new pipeline gives ${fmt(g('E35_final_lstm8', M, 'r2_all'))} vs ${fmt(g('new', M, 'r2_all'))} on the main 3, with more phantom torque (${fmt(g('E35_final_lstm8', M, 'phantom'))} vs ${fmt(g('new', M, 'phantom'))} Nm): a small gain, as said before.`)],
  ],
  notes: 'To show what each change really contributes, we took one change out of the new pipeline at a time and kept everything else the same, one training run each. The in-trial passive level is by far the most important: with the old table the main three drop to about 0.72 and the toes-down angles to about 0.48. Auto-gain matters mostly for JM and EG, whose EMG size changed a lot; for YES and HM it is slightly better without it, so it is not free. Dividing by the typical swing mainly protects the toes-down angles. The bigger network adds a little.',
  scale: `taking only this step out of the new pipeline, the toes-down R² of the main 3 falls from ${fmt(g('new', M, 'r2_pf2'))} to ${fmt(g('E33_final_noscale', M, 'r2_pf2'))} (1 run; see the “what each change contributes” slide).`,
};
