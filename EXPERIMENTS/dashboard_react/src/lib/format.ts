// Number formatting. ASCII '-' is used for negatives so numbers in the DOM stay machine-readable.

export function fmt(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  const s = v.toFixed(digits);
  return s === '-' + (0).toFixed(digits) ? (0).toFixed(digits) : s;
}

/** ankle position label, e.g. "+0.24" or "-0.79" */
export function posLabel(p: number, digits = 2): string {
  const s = p.toFixed(digits);
  return p > 0 ? '+' + s : s;
}
