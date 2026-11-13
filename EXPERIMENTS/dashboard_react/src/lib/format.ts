// Number formatting. ASCII '-' is used for negatives so numbers in the DOM stay machine-readable.

export function fmt(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  const s = v.toFixed(digits);
  return s === '-' + (0).toFixed(digits) ? (0).toFixed(digits) : s;
}

export function signed(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  const s = fmt(v, digits);
  return v > 0 && Number(s) !== 0 ? '+' + s : s;
}

/** ankle position label, e.g. "+0.24" or "-0.79" */
export function posLabel(p: number, digits = 2): string {
  const s = p.toFixed(digits);
  return p > 0 ? '+' + s : s;
}

export function range(lo: number, hi: number, digits = 2): string {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return '–';
  return `${fmt(lo, digits)} to ${fmt(hi, digits)}`;
}

export function pct(v: number, digits = 0): string {
  if (!Number.isFinite(v)) return '–';
  return `${(v * 100).toFixed(digits)} %`;
}

export function plural(n: number, one: string, many = one + 's'): string {
  return `${n} ${n === 1 ? one : many}`;
}

export function dirLabel(d: string, long = false): string {
  if (d === 'S1toS2') return long ? 'learn session 1 → test session 2' : 'learn 1 → test 2';
  if (d === 'S2toS1') return long ? 'learn session 2 → test session 1' : 'learn 2 → test 1';
  return d;
}

export function listJoin(items: string[]): string {
  return items.join(', ');
}
