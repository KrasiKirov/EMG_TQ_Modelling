// Plain-language takeaways: short sentences built from the numbers the page shows for the selected setting.
// Rules (design 22 Nov 2026): only what the numbers show; each names its session; no percentages comparing settings;
// never "best" for a "no clear difference"; no claim the study did not test.
import { CONFIRM, kindOf, sizeOf, trainOf, tuningRuns, type OverfitPoint, type PersonRuns } from './v2';
import { personScore, worstR2 } from './score';
import { worstPositions } from './studyWaves';
import { median } from './stats';

const list = (xs: string[]) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const n2 = (v: number) => v.toFixed(2), n3 = (v: number) => v.toFixed(3);
const medOver = (row: number[]) => median(row.filter(Number.isFinite));
const follow = (r2: number) => (r2 >= 0.95 ? 'closely' : r2 >= 0.85 ? 'fairly well' : 'only partly');

interface Pair { arm: string; comparator: string; effects: Record<string, number>; decision: string }
const isPair = (c: unknown): c is Pair => typeof c === 'object' && c !== null && 'arm' in c;
const lowerFor = (c: Pair) => Object.values(c.effects).filter((x) => x < 0).length;

/** the study's answer, in words (session 2, pre-registered decisions from confirm.json) */
export function headline(): string[] {
  const f3 = Object.values(CONFIRM.families.F3).filter(isPair);
  const vs = (kind: string) => f3.find((c) => c.arm === 'FE_A0' && c.comparator.endsWith(kind));
  const out: string[] = [];
  const a1 = vs('A1'), a2 = vs('A2');
  if (a1 && a2) {
    out.push(`The chosen tuned LSTM (A0) had lower session-2 error than the lagged ridge baseline for ${lowerFor(a1)} of 6 people, `
      + `and lower than last semester's 8×2 network for ${lowerFor(a2)} of 6 (decisions: ${a1.decision}, ${a2.decision}).`);
  }
  const f1 = Object.values(CONFIRM.families.F1).filter(isPair), f2 = Object.values(CONFIRM.families.F2);
  const none = (xs: { decision: string }[]) => xs.length > 0 && xs.every((c) => c.decision === 'no clear difference');
  const same: string[] = [];
  if (none(f1.filter((c) => c.arm.includes('_w')))) same.push('a longer window (0.5 s or 1 s instead of 0.2 s)');
  if (none(f1.filter((c) => c.arm.includes('carry')))) same.push('carrying the memory across windows (2, 5 or 10 s)');
  if (none(f2)) same.push('less window overlap');
  if (same.length) out.push(`On session 2, ${list(same)} made no clear difference: the pre-registered rule needs all 6 people to agree and a difference of at least 5 %.`);
  const a3 = f3.find((c) => c.arm === 'FE_A3');
  if (a3) out.push(`Training on total instead of active torque (A3): ${a3.decision}.`);
  const notRun = Object.entries(CONFIRM.families.F1).filter(([, c]) => /budget/.test(c.decision)).map(([k]) => k);
  if (notRun.length) out.push('Windows of 2 s were screened on validation but not tuned, and 5 s and 10 s were not run, within the compute budget.');
  return out;
}

/** under the summary table */
export function summaryTakeaway(id: string, people: PersonRuns[], total: boolean): string[] {
  if (!people.length) return [];
  const s = people.map((p) => ({ subj: p.subj, nmae: personScore(p, 'nmae'), r2: personScore(p, 'r2'), worst: worstR2(p) }));
  const byN = [...s].sort((a, b) => a.nmae - b.nmae);
  const comb = Math.exp(s.reduce((a, x) => a + Math.log(x.nmae), 0) / s.length);
  const out = [
    `nMAE ${n3(comb)} (combined) means the typical error on session 2 is about ${(comb * 100).toFixed(1)} % of the person's near-peak active torque`
      + `${total ? ' (this setting predicts total torque but is scaled the same way)' : ''}; it ranges from ${n3(byN[0].nmae)} (${byN[0].subj}) to ${n3(byN[byN.length - 1].nmae)} (${byN[byN.length - 1].subj}).`,
    `At a typical position the torque's rise and fall is followed ${follow(median(s.map((x) => x.r2)))} (median R² over people ${n2(median(s.map((x) => x.r2)))}; 1 = perfect, 0 = no better than the trial's mean).`,
  ];
  const weak = s.filter((x) => x.worst < 0.5);
  if (weak.length) {
    const pf = weak.filter((x) => (worstPositions(id, x.subj)[0]?.k ?? 99) < 2).map((x) => x.subj);
    out.push(`The worst position is much weaker for ${list(weak.map((x) => x.subj))} (R² below 0.5)`
      + (pf.length === weak.length ? `; for ${weak.length === 1 ? 'that person' : `all ${weak.length}`} it is one of the 2 most plantarflexed positions.`
        : pf.length ? `; for ${list(pf)} it is one of the 2 most plantarflexed positions.` : '.'));
  } else out.push('Every person\'s worst position still has R² of at least 0.5.');
  return out;
}

/** under the waveforms: why the worst positions score low */
export function wavesTakeaway(id: string, people: PersonRuns[]): string[] {
  const worstPeak: number[] = [], allPeak: number[] = [];
  for (const p of people) {
    const w = worstPositions(id, p.subj)[0];
    if (w && Number.isFinite(p.peaks[w.k])) worstPeak.push(p.peaks[w.k]);
    allPeak.push(...p.peaks.filter(Number.isFinite));
  }
  if (!worstPeak.length) return [];
  const a = median(worstPeak), b = median(allPeak);
  return a < b
    ? [`At each person's worst position the measured torque is small (median peak ${a.toFixed(1)} Nm, against ${b.toFixed(1)} Nm over all positions): `
      + 'a small error in Nm is a large share of the signal there, so R² drops even where the curve shape looks close.']
    : [`The worst positions are not the low-torque ones (median peak ${a.toFixed(1)} Nm, against ${b.toFixed(1)} Nm over all positions).`];
}

/** under a per-position chart */
export function positionsTakeaway(people: PersonRuns[], m: 'mae' | 'r2'): string[] {
  if (!people.length) return [];
  const at = people.map((p) => {
    const v = (m === 'mae' ? p.mae : p.r2).map(medOver);
    const k = m === 'mae' ? v.indexOf(Math.max(...v.filter(Number.isFinite))) : v.indexOf(Math.min(...v.filter(Number.isFinite)));
    return { subj: p.subj, k, dorsi: p.positions[k] > 0 };
  });
  if (m === 'mae') {
    const d = at.filter((x) => x.dorsi).length;
    return [`The largest error in Nm is at a dorsiflexed position (positive angle) for ${d} of ${people.length} people. `
      + 'Error in Nm grows with the torque itself, which is why the summary also shows nMAE.'];
  }
  const pf = at.filter((x) => x.k < 2).length;
  return [`The lowest R² is at one of the 2 most plantarflexed positions for ${pf} of ${people.length} people.`];
}

/** under the peak-torque chart */
export function peakTakeaway(people: PersonRuns[]): string[] {
  if (!people.length) return [];
  const pk = people.map((p) => ({ subj: p.subj, max: Math.max(...p.peaks.filter(Number.isFinite)), last: p.peaks.indexOf(Math.max(...p.peaks.filter(Number.isFinite))) === p.peaks.length - 1 }));
  const s = [...pk].sort((a, b) => a.max - b.max);
  return [`Peak torque differs about ${Math.round(s[s.length - 1].max / s[0].max)}-fold between people (${s[0].max.toFixed(1)} Nm for ${s[0].subj} to ${s[s.length - 1].max.toFixed(1)} Nm for ${s[s.length - 1].subj}), `
    + `and is highest at the most dorsiflexed position for ${pk.filter((x) => x.last).length} of ${people.length}. This is why errors are compared as nMAE.`];
}

/** the tuning experiment: study-wide, then this setting */
export function tuningTakeaway(id: string): string[] {
  const runs = tuningRuns();
  const by = (u: number, l: number) => runs.filter((r) => r.units === u && r.layers === l).map((r) => r.score);
  const m = (u: number, l: number) => median(by(u, l));
  const winners = Object.values(CONFIRM.choice.cell_winners).map((w) => runs.find((x) => x.id === w)).filter((r): r is NonNullable<typeof r> => !!r);
  const lowDrop = winners.filter((r) => r.dropout <= 0.1).length;
  const out = [
    `On validation (session 1), 8-unit networks scored clearly worse (median nMAE ${n3(m(8, 1))} at 8×1, ${n3(m(8, 2))} at 8×2); from 16 units up the one-layer medians are close (${n3(m(16, 1))} at 16×1, ${n3(m(32, 1))} at 32×1, ${n3(m(64, 1))} at 64×1), with the lowest single runs at 64×1.`,
    `${Object.values(CONFIRM.choice.cell_winners).filter((w) => { const r = runs.find((x) => x.id === w); return r && r.units === 64 && r.layers === 1; }).length} of ${Object.keys(CONFIRM.choice.cell_winners).length} cell winners are 64×1. `
      + '64 units was the largest size searched and 64×2 was never drawn, so whether a larger network would do better was not tested.',
    'The coarse round\'s small networks also had the lowest learning rates (8–16 units at 0.0004–0.0018), so size and learning rate are partly mixed up in the coarse draws; the fine round varies them separately around each cell\'s best.',
    `Across all runs, within a cell, dropout had the clearest link to validation error (higher dropout, higher error; ${lowDrop} of ${winners.length} cell winners have dropout of 0.1 or less); size (more units, lower error) and learning rate (higher, a little lower error) had smaller links of about equal size. This is a description of the search, not a controlled experiment.`,
  ];
  const k = kindOf(id), size = sizeOf(id);
  if (k === 'A1') out.push('This setting (lagged ridge) has no network and was not tuned beyond its α.');
  else if (k === 'A2') out.push('This setting is last semester\'s 8×2 network with dropout 0.3, fixed and not tuned; it is the anchor of every coarse round.');
  else if (size) out.push(`This setting's network is ${size}.`);
  return out;
}

/** the overfitting check */
export function overfitTakeaway(id: string, pts: OverfitPoint[]): string[] {
  const out: string[] = [];
  const t = trainOf(id);
  if (t && kindOf(id) !== 'A1') {
    out.push(`This setting's runs stopped training at a median of ${Math.round(t.epochs)} epochs and kept the weights of epoch ${Math.round(t.bestEpoch)}: `
      + `the point where the session-1 validation error was lowest (${Math.round(t.nParams).toLocaleString('en')} weights, over ${t.runs} runs).`);
  }
  const g = (p: OverfitPoint) => p.test / p.val;
  const by = (s: string) => pts.filter((p) => p.size === s).map(g);
  const r64 = by('64×1'), r32 = by('32×1'), r8 = by('8×2');
  if (r64.length && r32.length) {
    out.push(`Every network's error roughly doubles from session-1 validation to session 2 (64×1 winners ×${Math.min(...r64).toFixed(2)}–${Math.max(...r64).toFixed(2)}, `
      + `32×1 winners ×${Math.min(...r32).toFixed(2)}–${Math.max(...r32).toFixed(2)}${r8.length ? `, last semester's 8×2 ×${r8[0].toFixed(2)} from a higher start` : ''}). `
      + 'The 32×1 and 64×1 winners grow by about the same factor, so the growth is not specific to the larger network. It fits the change between sessions plus the optimism of picking the lowest of many validation scores (the 8×2 was not picked, so it has no such optimism).');
  }
  const f3 = Object.values(CONFIRM.families.F3).filter(isPair).find((c) => c.arm === 'FE_A0' && c.comparator.endsWith('A2'));
  if (f3) out.push(`On session 2, which no choice used, the 64×1 network (A0) still had lower error than the 8×2 for ${lowerFor(f3)} of 6 people. A network that had overfitted session 1 would be expected to lose that lead on new data.`);
  out.push('Not shown by this study: training-loss curves were not saved, and no run varied the network size with everything else fixed. A size sweep scored on session 2 would be the clean test.');
  return out;
}
