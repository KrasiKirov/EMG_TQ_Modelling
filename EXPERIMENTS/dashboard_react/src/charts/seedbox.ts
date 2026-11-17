// Series for "one box per position over training seeds": Tukey box plot or median + 95 % CI, a large dot at the
// median, every seed as a small dot, outliers as rings. Boxes sit on category axis 0; dots, CI bars and medians on a
// hidden value axis 1 (x = position index + offset), so two models can sit side by side.
import { boxStats, type BoxStats } from '../lib/boxstats';
import { alpha, type ChartTokens } from '../theme';

export type SpreadMode = 'box' | 'ci';
export interface SeedPoint { seed: number; v: number }

export interface SeedSide {
  name: string;
  color: string;
  /** one entry per position (empty array = no runs) */
  cells: SeedPoint[][];
  offset: number;
}

export interface SeedSeriesOpts {
  mode: SpreadMode;
  compact?: boolean;
  /** maps a value onto the visible axis (clipped charts); identity by default */
  clamp?: (v: number) => number;
  /** values not drawn as dots (off the axis) */
  hidden?: (v: number) => boolean;
  boxWidth?: [number, number];
}

export const statsOf = (cells: SeedPoint[][]): BoxStats[] => cells.map((c) => boxStats(c.map((p) => p.v)));

export function seedSeries(side: SeedSide, t: ChartTokens, o: SeedSeriesOpts): Record<string, unknown>[] {
  const clamp = o.clamp ?? ((v: number) => v);
  const hidden = o.hidden ?? (() => false);
  const st = statsOf(side.cells);
  const small = o.compact ? 4 : 6;
  const box = {
    type: 'boxplot', name: side.name, xAxisIndex: 0, z: 3,
    boxWidth: o.boxWidth ?? (o.compact ? [5, 18] : [8, 30]),
    itemStyle: { color: alpha(side.color, 0.22), borderColor: side.color, borderWidth: 2 },
    emphasis: { itemStyle: { color: alpha(side.color, 0.3), borderColor: side.color, borderWidth: 2 } },
    data: st.map((s) => (o.mode === 'box' && s.n && !hidden(s.median) ? [s.lo, s.q1, s.median, s.q3, s.hi].map(clamp) : '-')),
  };
  const seedDots: unknown[] = [], outliers: unknown[] = [];
  side.cells.forEach((c, k) => {
    const s = st[k];
    const out = new Set(o.mode === 'box' ? s.outliers : []);
    c.forEach((p, i) => {
      if (hidden(p.v)) return;
      const x = k + side.offset + ((i % 5) - 2) * (o.compact ? 0.018 : 0.022);
      (out.has(p.v) ? outliers : seedDots).push({ value: [x, p.v, p.seed, k] });
    });
  });
  const ci = st.map((s, k) => (o.mode === 'ci' && s.n > 1 ? [k + side.offset, clamp(s.ci.lo), clamp(s.ci.hi), k] : null)).filter(Boolean);
  const series: Record<string, unknown>[] = [
    box,
    { type: 'scatter', name: `${side.name} · seeds`, xAxisIndex: 1, z: 4, symbolSize: small,
      itemStyle: { color: side.color, opacity: o.mode === 'ci' ? 0.35 : 0.7, borderColor: t.surface, borderWidth: 0.6 }, data: seedDots },
    { type: 'scatter', name: `${side.name} · outliers`, xAxisIndex: 1, z: 6, symbol: 'circle', symbolSize: o.compact ? 8 : 10,
      itemStyle: { color: 'transparent', borderColor: side.color, borderWidth: 2 }, data: outliers },
    { type: 'scatter', name: `${side.name} · median`, xAxisIndex: 1, z: 7, symbolSize: o.compact ? 8 : 11,
      itemStyle: { color: side.color, borderColor: t.surface, borderWidth: 1.5 },
      data: st.map((s, k) => (s.n && !hidden(s.median) ? { value: [k + side.offset, clamp(s.median), -1, k] } : null)).filter(Boolean) },
  ];
  if (ci.length)
    series.push({
      type: 'custom', name: `${side.name} · 95 % CI`, xAxisIndex: 1, z: 5, silent: true, data: ci, encode: { x: 0, y: [1, 2] },
      renderItem: (_: unknown, api: { value: (i: number) => number; coord: (p: number[]) => number[] }) => {
        const x = api.value(0), a = api.coord([x, api.value(1)]), b = api.coord([x, api.value(2)]), w = o.compact ? 4 : 6;
        const style = { stroke: side.color, lineWidth: 2.2 };
        return { type: 'group', children: [
          { type: 'line', shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style },
          { type: 'line', shape: { x1: a[0] - w, y1: a[1], x2: a[0] + w, y2: a[1] }, style },
          { type: 'line', shape: { x1: b[0] - w, y1: b[1], x2: b[0] + w, y2: b[1] }, style },
        ] };
      },
    });
  return series;
}

/** tooltip rows for one side at one position */
export function seedTooltip(side: SeedSide, k: number, digits: number, unit: string): string {
  const c = side.cells[k] ?? [];
  if (!c.length) return `<div><b style="color:${side.color}">${side.name}</b>: no runs</div>`;
  const s = boxStats(c.map((p) => p.v));
  const f = (v: number) => (Number.isFinite(v) ? v.toFixed(digits) : '–');
  const seedOf = (v: number) => c.filter((p) => p.v === v).map((p) => `seed ${p.seed}`).join(', ');
  return `<div style="margin-top:4px"><b style="color:${side.color}">${side.name}</b> · ${s.n} seeds (${c.map((p) => p.seed).sort((a, b) => a - b).join(', ')})<br/>` +
    `median <b>${f(s.median)}</b>${unit} · middle half ${f(s.q1)} to ${f(s.q3)}<br/>` +
    `whiskers ${f(s.lo)} to ${f(s.hi)} · ${Math.round(s.ci.level * 1000) / 10} % CI of the median ${f(s.ci.lo)} to ${f(s.ci.hi)}` +
    (s.outliers.length ? `<br/>outlier${s.outliers.length > 1 ? 's' : ''}: ${s.outliers.map((v) => `${f(v)} (${seedOf(v)})`).join('; ')}` : '') + '</div>';
}

/** the legend line under a seed chart */
export const SEED_KEY = {
  box: 'Box: middle half of seeds · whiskers: 1.5 × box · small dots: seeds.',
  ci: 'Bar: 95 % CI of the median over seeds · small dots: seeds.',
} as const;

