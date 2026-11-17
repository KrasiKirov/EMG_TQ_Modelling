/**
 * Box-plot statistics over training seeds (Tukey box plot) and a distribution-free confidence interval of the median.
 */

export interface BoxStats {
  n: number;
  q1: number;
  median: number;
  q3: number;
  /** whisker ends: the smallest and largest run within 1.5 × IQR of the box */
  lo: number;
  hi: number;
  /** runs beyond the whiskers */
  outliers: number[];
  ci: MedianCI;
}

export interface MedianCI {
  lo: number;
  hi: number;
  /** exact coverage of the interval (it cannot be exactly 95 % with few runs) */
  level: number;
}

const finiteSorted = (values: Iterable<number>) => Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);

/** numpy.percentile default ('linear') on a sorted array */
function q(a: number[], p: number): number {
  if (!a.length) return NaN;
  const i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

/** P(X <= k) for X ~ Binomial(n, 1/2) */
function binomCdfHalf(n: number, k: number): number {
  let c = 1, s = 0;
  for (let i = 0; i <= k; i++) {
    s += c;
    c = (c * (n - i)) / (i + 1);
  }
  return s / 2 ** n;
}

/**
 * Order-statistic CI of the median: [x(k), x(n+1-k)] with the largest k whose coverage, 1 − 2·P(X ≤ k−1), is still at
 * least 95 %. With 10 runs that is the 2nd to the 9th smallest (97.9 %). With 5 runs nothing narrower than the full
 * range reaches 95 %, so the CI is the full range (93.8 %).
 */
export function medianCI(values: Iterable<number>): MedianCI {
  const a = finiteSorted(values), n = a.length;
  if (!n) return { lo: NaN, hi: NaN, level: NaN };
  let k = 1;
  while (k + 1 <= Math.floor((n + 1) / 2) && 1 - 2 * binomCdfHalf(n, k) >= 0.95) k++;
  return { lo: a[k - 1], hi: a[n - k], level: 1 - 2 * binomCdfHalf(n, k - 1) };
}

export function boxStats(values: Iterable<number>): BoxStats {
  const a = finiteSorted(values), n = a.length;
  const q1 = q(a, 0.25), q3 = q(a, 0.75), iqr = q3 - q1;
  const inside = a.filter((v) => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
  return {
    n, q1, median: q(a, 0.5), q3,
    lo: inside.length ? inside[0] : NaN,
    hi: inside.length ? inside[inside.length - 1] : NaN,
    outliers: a.filter((v) => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr),
    ci: medianCI(a),
  };
}
