// ECharts options for the results page.
import type { EOption } from '../components/EChart';
import type { ChartTokens } from '../theme';
import { posLabel } from '../lib/format';
import { seedSeries, seedTooltip, type SeedSide, type SpreadMode } from './seedbox';
import { boxStats } from '../lib/boxstats';

export type SMetric = 'mae' | 'nmae' | 'r2';
export const SMETRICS: Record<SMetric, { short: string; unit: string; error: boolean }> = {
  mae: { short: 'MAE', unit: 'Nm', error: true },
  nmae: { short: 'nMAE', unit: '', error: true },
  r2: { short: 'R²', unit: '', error: false },
};

/** text under a box: the median in bold, then its interval in grey on the same line ("" without runs) */
function ciLabels(cells: { v: number }[][], d: number): string[] {
  const f = (v: number) => v.toFixed(Math.abs(v) >= 10 ? Math.min(d, 1) : d);
  return cells.map((c) => {
    const s = boxStats(c.map((p) => p.v));
    if (!s.n) return '';
    return `{m|${f(s.median)}}` + (s.n > 1 ? ` {ci|[${f(s.ci.lo)}, ${f(s.ci.hi)}]}` : '');
  });
}
const ciRich = (t: ChartTokens) => ({
  m: { fontSize: 13, fontWeight: 600, color: t.text, lineHeight: 18 },
  ci: { fontSize: 13, color: t.textMuted, lineHeight: 16 },
});

/** lowest R² drawn on a per-position axis; runs below it are marked with an arrow at the floor */
export const R2_FLOOR = -1;

const axisCommon = (t: ChartTokens) => ({
  axisLine: { lineStyle: { color: t.axis } }, axisTick: { lineStyle: { color: t.axis } },
  axisLabel: { color: t.text, fontSize: 13 }, splitLine: { lineStyle: { color: t.grid } },
});

/** one bar per position: the peak measured torque there (Nm) */
export function peakBarOption(positions: number[], peaks: number[], t: ChartTokens): EOption {
  return {
    animation: false,
    grid: { left: 8, right: 12, top: 30, bottom: 30, containLabel: true },
    tooltip: { trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 13 },
               valueFormatter: (v: number) => (Number.isFinite(v) ? `${v.toFixed(1)} Nm` : '–') },
    xAxis: { type: 'category', data: positions.map((p, i) => `p${i + 1}\n${posLabel(p)}`), ...axisCommon(t),
             axisLabel: { color: t.text, fontSize: 13, interval: 0, lineHeight: 16 }, name: 'ankle position (rad)', nameLocation: 'middle', nameGap: 40, nameTextStyle: { color: t.textMuted, fontSize: 13 } },
    yAxis: { type: 'value', min: 0, ...axisCommon(t), name: 'peak torque (Nm)', nameTextStyle: { color: t.textMuted, fontSize: 13, align: 'left' },
             axisLabel: { color: t.text, fontSize: 13 } },
    series: [{ type: 'bar', name: 'peak torque', barWidth: '55%', itemStyle: { color: '#9aa1aa' },
               data: peaks.map((v) => (Number.isFinite(v) ? Math.round(v * 10) / 10 : null)),
               label: { show: true, position: 'top', color: t.text, fontSize: 13 } }],
  };
}

// ── every seed at every position (setting vs its reference) ─────────────────
/** peaks: peak measured torque per position, drawn on a right-hand axis when the metric is an error (Nm) */
export function studySeedOption(sides: SeedSide[], positions: number[], m: SMetric, mode: SpreadMode, t: ChartTokens, compact: boolean, subj: string): EOption {
  const all = sides.flatMap((s) => s.cells.flat().map((p) => p.v)).filter(Number.isFinite);
  const err = SMETRICS[m].error;
  // errors: from 0 to a round number above the largest run; R²: from a round number below the lowest run to 1
  // R² is floored at R2_FLOOR so one very bad position (e.g. R² −112) does not squash every other box flat
  const lo = err ? 0 : Math.max(R2_FLOOR, Math.floor(Math.min(...all) * 20) / 20), hi = err ? undefined : 1;
  const off = (v: number) => !err && v < lo;
  const labels = compact ? [] : ciLabels(sides[0]?.cells ?? [], 2);
  const below = sides.flatMap((s) => s.cells.map((c, k) => {
    const xs = c.map((p) => p.v).filter(off);
    if (!xs.length) return null;
    const med = boxStats(c.map((p) => p.v)).median;
    const text = off(med) ? `median ${med.toFixed(med < -10 ? 0 : 1)}` : `${xs.length} below`;
    return { value: [k + s.offset, lo, -1, k], label: { formatter: `↓ ${text}` }, itemStyle: { color: s.color } };
  }).filter(Boolean));
  const unit = SMETRICS[m].unit ? ` ${SMETRICS[m].unit}` : '';
  const kOf = (p: { seriesType?: string; dataIndex: number; value?: unknown }) => (p.seriesType === 'boxplot' ? p.dataIndex : Array.isArray(p.value) ? Number(p.value[3]) : -1);
  return {
    animation: false,
    grid: { left: 8, right: 12, top: compact ? 10 : 30, bottom: compact ? 4 : 30, containLabel: true },
    tooltip: {
      trigger: 'item', confine: true, backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 13 },
      formatter: (p: { seriesType?: string; dataIndex: number; value?: unknown }) => {
        const k = kOf(p);
        if (!(k >= 0)) return '';
        return `<b>${subj} · position ${k + 1} (${posLabel(positions[k])} rad)${k < 2 ? ' · one of the 2 most plantarflexed' : ''}</b>`
          + sides.map((s) => seedTooltip(s, k, 3, unit)).join('');
      },
    },
    xAxis: [
      { type: 'category', data: positions.map((p, i) => `p${i + 1}\n${posLabel(p)}`), ...axisCommon(t),
        axisLabel: compact ? { color: t.text, fontSize: 13, interval: 0, lineHeight: 14 }
          : { color: t.text, fontSize: 13, interval: 0, lineHeight: 16, rich: ciRich(t), formatter: (v: string, k: number) => `${v}\n${labels[k] ?? ''}` },
        ...(compact ? {} : { name: 'ankle position (rad) · median [interval]', nameLocation: 'middle', nameGap: 60, nameTextStyle: { color: t.textMuted, fontSize: 13 } }) },
      { type: 'value', min: -0.5, max: positions.length - 0.5, show: false },
    ],
    yAxis: [{
      type: 'value', min: lo, max: hi, ...axisCommon(t), axisLine: { onZero: false, lineStyle: { color: t.axis } }, name: compact ? undefined : `${SMETRICS[m].short}${unit}`,
      nameTextStyle: { color: t.textMuted, fontSize: compact ? 12 : 13, align: 'left' },
      axisLabel: { color: t.text, fontSize: compact ? 12 : 13, formatter: (v: number) => String(+v.toFixed(err ? 2 : 3)) },
    }],
    series: [
      { type: 'scatter', xAxisIndex: 1, data: [], silent: true,
        markArea: { silent: true, itemStyle: { color: t.shade }, data: positions.length >= 2 ? [[{ xAxis: -0.5 }, { xAxis: 1.5 }]] : [] } },
      ...sides.flatMap((s) => seedSeries(s, t, { mode, compact, boxWidth: compact ? [4, 14] : [10, 34],
        clamp: (v) => (off(v) ? lo : v), hidden: off })),
      ...(below.length ? [{ type: 'scatter', name: 'below the axis', xAxisIndex: 1, z: 8, symbol: 'triangle', symbolRotate: 180, symbolSize: 12,
        data: below, label: { show: true, position: 'top', color: t.text, fontSize: 13, fontWeight: 600 } }] : []),
    ],
  };
}

/** one box per person: training time of every seed (minutes), with the median's CI beside it */
export function runtimeOption(people: { subj: string; seeds: number[]; seconds: number[] }[], t: ChartTokens): EOption {
  const side: SeedSide = { name: 'Seeds', color: t.a, offset: 0,
    cells: people.map((p) => p.seconds.map((s, i) => ({ seed: p.seeds[i], v: s / 60 })).filter((x) => Number.isFinite(x.v))) };
  const kOf = (p: { seriesType?: string; dataIndex: number; value?: unknown }) => (p.seriesType === 'boxplot' ? p.dataIndex : Array.isArray(p.value) ? Number(p.value[3]) : -1);
  const labels = ciLabels(side.cells, 1);
  return {
    animation: false,
    grid: { left: 8, right: 12, top: 30, bottom: 30, containLabel: true },
    tooltip: {
      trigger: 'item', confine: true, backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 13 },
      formatter: (p: { seriesType?: string; dataIndex: number; value?: unknown }) => {
        const k = kOf(p);
        return k >= 0 ? `<b>${people[k].subj}</b>` + seedTooltip(side, k, 1, ' min') : '';
      },
    },
    xAxis: [
      { type: 'category', data: people.map((p) => p.subj), ...axisCommon(t),
        axisLabel: { color: t.text, fontSize: 13, lineHeight: 18, rich: ciRich(t), formatter: (v: string, k: number) => `${v}\n${labels[k] ?? ''}` },
        name: 'median [interval]', nameLocation: 'middle', nameGap: 44, nameTextStyle: { color: t.textMuted, fontSize: 13 } },
      { type: 'value', min: -0.5, max: people.length - 0.5, show: false },
    ],
    yAxis: { type: 'value', min: 0, ...axisCommon(t), axisLine: { onZero: false, lineStyle: { color: t.axis } }, name: 'minutes per run', nameTextStyle: { color: t.textMuted, fontSize: 13, align: 'left' },
             axisLabel: { color: t.text, fontSize: 13 } },
    series: seedSeries(side, t, { mode: 'box', boxWidth: [10, 40] }),
  };
}
