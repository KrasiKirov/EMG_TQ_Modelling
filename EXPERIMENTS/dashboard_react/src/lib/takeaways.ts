// The study's answer in words, from confirm.json's pre-registered decisions (shown under the page title).
// Rules (design 22 Nov 2026): only what the numbers show; each names its session; no percentages comparing settings;
// never "best" for a "no clear difference"; no claim the study did not test.
import { CONFIRM } from './v2';

const list = (xs: string[]) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

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
