

/** numpy.percentile default ('linear'): sort, idx = (n-1)*p, interpolate between floor and ceil */
function quantileSorted(a: ArrayLike<number>, p: number): number {
  const n = a.length;
  if (!n) return NaN;
  const i = (n - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

function sortNumbers(values: Iterable<number>): number[] {
  const a = Array.from(values).filter((v) => Number.isFinite(v));
  a.sort((x, y) => x - y);
  return a;
}

function quantile(values: Iterable<number>, p: number): number {
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
