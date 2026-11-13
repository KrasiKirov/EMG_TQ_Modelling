import type { Summary } from './types';

/** numpy.percentile default ('linear'): sort, idx = (n-1)*p, interpolate between floor and ceil */
export function quantileSorted(a: ArrayLike<number>, p: number): number {
  const n = a.length;
  if (!n) return NaN;
  const i = (n - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

export function sortNumbers(values: Iterable<number>): number[] {
  const a = Array.from(values).filter((v) => Number.isFinite(v));
  a.sort((x, y) => x - y);
  return a;
}

export function quantile(values: Iterable<number>, p: number): number {
  return quantileSorted(sortNumbers(values), p);
}

export function median(values: Iterable<number>): number {
  return quantile(values, 0.5);
}

export function mean(values: Iterable<number>): number {
  let s = 0;
  let n = 0;
  for (const v of values) {
    if (Number.isFinite(v)) {
      s += v;
      n++;
    }
  }
  return n ? s / n : NaN;
}

export const EMPTY_SUMMARY: Summary = { n: 0, p1: NaN, p5: NaN, p25: NaN, p50: NaN, p75: NaN, p95: NaN, p99: NaN };

export function summarize(values: Iterable<number>): Summary {
  const a = sortNumbers(values);
  if (!a.length) return EMPTY_SUMMARY;
  return {
    n: a.length,
    p1: quantileSorted(a, 0.01),
    p5: quantileSorted(a, 0.05),
    p25: quantileSorted(a, 0.25),
    p50: quantileSorted(a, 0.5),
    p75: quantileSorted(a, 0.75),
    p95: quantileSorted(a, 0.95),
    p99: quantileSorted(a, 0.99),
  };
}
