// ECharts options for the window-study page.
import type { EOption } from '../components/EChart';
import type { ChartTokens } from '../theme';
import { alpha } from '../theme';
import { fmtEffect, PEOPLE, SMETRICS, type Comparison, type SMetric } from '../lib/study';
import { posLabel } from '../lib/format';
import { seedSeries, seedTooltip, type SeedSide, type SpreadMode } from './seedbox';
import { boxStats } from '../lib/boxstats';

/** text under a box: the median in bold, its interval below it in grey ("" without runs) */
function ciLabels(cells: { v: number }[][], d: number): string[] {
  const f = (v: number) => v.toFixed(Math.abs(v) >= 10 ? Math.min(d, 1) : d);
  return cells.map((c) => {
    const s = boxStats(c.map((p) => p.v));
    if (!s.n) return '';
    return `{m|${f(s.median)}}` + (s.n > 1 ? `\n{ci|[${f(s.ci.lo)}, ${f(s.ci.hi)}]}` : '');
  });
}
const ciRich = (t: ChartTokens) => ({
  m: { fontSize: 12, fontWeight: 600, color: t.text, lineHeight: 18 },
  ci: { fontSize: 10, color: t.textMuted, lineHeight: 14 },
});

/** lowest R² drawn on a per-position axis; runs below it are marked with an arrow at the floor */
export const R2_FLOOR = -1;

/** one colour per person */
export const PERSON_COLORS = ['#2a6fc2', '#d4612b', '#1f8a4c', '#7a4fc4', '#b8377f', '#9a7400'];

const axisCommon = (t: ChartTokens) => ({
  axisLine: { lineStyle: { color: t.axis } }, axisTick: { lineStyle: { color: t.axis } },
  axisLabel: { color: t.textMuted, fontSize: 11 }, splitLine: { lineStyle: { color: t.grid } },
});

/**
 * Dot plot: one row per setting, one dot per person (its effect against the comparison), a bar at the median,
 * the neutral line (ratio 1 / difference 0) and a shaded no-difference band.
 */
/** the change shown by a comparison bar: errors in % (negative = lower error), R² as a plain difference */
export const effectValue = (median: number, m: SMetric): number | null =>
  Number.isFinite(median) ? (SMETRICS[m].error ? (median - 1) * 100 : median) : null;

/**
 * One bar per setting: its median change against the comparison, nothing else.
 * Green: clearly better, red: clearly worse, grey: no clear change. Per-person values are in the tooltip.
 */
export function effectBarOption(rows: { label: string; c: Comparison }[], m: SMetric, t: ChartTokens): EOption {
  const err = SMETRICS[m].error;
  const short = SMETRICS[m].short;
  const vals = rows.map((r) => effectValue(r.c.median, m));
  const fmtv = (v: number) => (err ? `${v > 0 ? '+' : ''}${v.toFixed(1)} %` : `${v > 0 ? '+' : ''}${v.toFixed(3)}`);
  const color = (v: Comparison['verdict']) => (v === 'better' ? t.good : v === 'worse' ? t.bad : t.textMuted);
  return {
    animation: false,
    grid: { left: 8, right: 24, top: 8, bottom: 40, containLabel: true },
    tooltip: {
      trigger: 'item', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 },
      formatter: (p: { dataIndex: number }) => {
        const r = rows[p.dataIndex];
        if (!r) return '';
        return `<b>${r.label}</b><br/>median ${fmtEffect(r.c.median, m)} · better for ${r.c.better}/${r.c.n} people<br/>` +
          r.c.people.map((x) => `${x.subj}: ${fmtEffect(x.effect, m)}`).join('<br/>');
      },
    },
    xAxis: {
      type: 'value', boundaryGap: ['15%', '15%'], ...axisCommon(t),
      name: err ? `Change in ${short} (%) · negative = lower error` : `Change in ${short} · positive = better`,
      nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.textMuted, fontSize: 11 },
      axisLabel: { color: t.textMuted, fontSize: 11, formatter: (v: number) => (err ? `${v > 0 ? '+' : ''}${+v.toFixed(1)} %` : `${v > 0 ? '+' : ''}${+v.toFixed(3)}`) },
    },
    yAxis: { type: 'category', data: rows.map((r) => r.label), inverse: true, ...axisCommon(t), splitLine: { show: false },
             axisLabel: { color: t.text, fontSize: 12, width: 240, overflow: 'truncate' } },
    series: [{
      type: 'bar', barWidth: 14,
      data: rows.map((r, i) => ({
        value: vals[i], itemStyle: { color: color(r.c.verdict) },
        label: { show: vals[i] !== null, position: (vals[i] ?? 0) < 0 ? 'left' : 'right', color: t.text, fontSize: 11, formatter: () => fmtv(vals[i] ?? 0) },
      })),
      markLine: { silent: true, symbol: 'none', lineStyle: { color: t.axis, type: 'solid', width: 1 }, label: { show: false }, data: [{ xAxis: 0 }] },
    }],
  };
}

/** row index of a click on a comparison bar */
export function clickedRow(p: { dataIndex?: number }): number {
  return typeof p.dataIndex === 'number' ? p.dataIndex : -1;
}

/** one bar per position: the peak measured torque there (Nm) */
export function peakBarOption(positions: number[], peaks: number[], t: ChartTokens): EOption {
  return {
    animation: false,
    grid: { left: 8, right: 12, top: 30, bottom: 30, containLabel: true },
    tooltip: { trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 13 },
               valueFormatter: (v: number) => (Number.isFinite(v) ? `${v.toFixed(1)} Nm` : '–') },
    xAxis: { type: 'category', data: positions.map((p, i) => `p${i + 1}\n${posLabel(p)}`), ...axisCommon(t),
             axisLabel: { color: t.text, fontSize: 12, interval: 0, lineHeight: 16 }, name: 'ankle position (rad)', nameLocation: 'middle', nameGap: 40, nameTextStyle: { color: t.textMuted, fontSize: 12 } },
    yAxis: { type: 'value', min: 0, ...axisCommon(t), name: 'peak torque (Nm)', nameTextStyle: { color: t.textMuted, fontSize: 12, align: 'left' },
             axisLabel: { color: t.text, fontSize: 12 } },
    series: [{ type: 'bar', name: 'peak torque', barWidth: '55%', itemStyle: { color: '#9aa1aa' },
               data: peaks.map((v) => (Number.isFinite(v) ? Math.round(v * 10) / 10 : null)),
               label: { show: true, position: 'top', color: t.text, fontSize: 12 } }],
  };
}

/** strongly bad cells are dark enough to need white text */
const strong = (b: number, lim: number) => Number.isFinite(b) && b > 0.55 * lim;

/** colour position of an effect: <0 good, >0 bad, clipped (log ratio for errors, −difference for R²) */
export function badness(v: number, m: SMetric): number {
  if (!Number.isFinite(v)) return NaN;
  return SMETRICS[m].error ? Math.max(-0.5, Math.min(0.5, Math.log(v))) : Math.max(-0.25, Math.min(0.25, -v));
}

export interface GridCell { x: number; y: number; c: Comparison | null; windows: number }

/** length × overlap heat map of the median effect against the reference */
export function gridOption(cells: GridCell[], xs: string[], ys: string[], m: SMetric, t: ChartTokens): EOption {
  const lim = SMETRICS[m].error ? 0.5 : 0.25;
  return {
    animation: false,
    grid: { left: 8, right: 8, top: 8, bottom: 40, containLabel: true },
    tooltip: {
      backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 },
      formatter: (p: { dataIndex: number }) => {
        const g = cells[p.dataIndex];
        if (!g?.c) return 'not run';
        return `<b>${ys[g.y]} · ${xs[g.x]} overlap</b><br/>median ${fmtEffect(g.c.median, m)}, better for ${g.c.better}/${g.c.n} people<br/>` +
          `${g.c.seeds} seeds · ${Number.isFinite(g.windows) ? g.windows.toLocaleString() : '?'} training windows (median person)`;
      },
    },
    xAxis: { type: 'category', data: xs, ...axisCommon(t), splitArea: { show: false }, name: 'overlap between training windows', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.textMuted, fontSize: 11 } },
    yAxis: { type: 'category', data: ys, ...axisCommon(t), inverse: true },
    visualMap: { show: false, type: 'continuous', dimension: 2, min: -lim, max: lim,
                 inRange: { color: [t.good, t.surface, t.bad] } },
    series: [{
      type: 'heatmap',
      data: cells.map((g) => { const b = g.c ? badness(g.c.median, m) : NaN; return { value: [g.x, g.y, b], label: { color: strong(b, lim) ? '#ffffff' : t.text } }; }),
      label: { show: true, fontSize: 12, color: t.text,
               formatter: (p: { dataIndex: number }) => { const g = cells[p.dataIndex]; return g?.c ? fmtEffect(g.c.median, m) : '–'; } },
      itemStyle: { borderColor: t.surface, borderWidth: 3, borderRadius: 4 },
      emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 1 } },
    }],
  };
}

/** person × position heat map of one setting: the metric itself, or its effect against a comparison */
export function cellsOption(values: (number | null)[][], xs: string[], ys: string[], m: SMetric, mode: 'value' | 'effect', t: ChartTokens): EOption {
  const err = SMETRICS[m].error;
  const flat = values.flat().filter((v): v is number => v !== null && Number.isFinite(v));
  const data: { value: [number, number, number, number]; label: { color: string } }[] = [];
  values.forEach((row, y) => row.forEach((v, x) => {
    if (v === null || !Number.isFinite(v)) return;
    // colour value: effects use the shared good/bad scale; raw values: R² from 0 to 1, errors from 0 to the largest
    const c = mode === 'effect' ? badness(v, m) : err ? v / Math.max(...flat) : 1 - Math.max(0, Math.min(1, v));
    data.push({ value: [x, y, c, v], label: { color: (mode === 'effect' ? strong(c, err ? 0.5 : 0.25) : c > 0.75) ? '#ffffff' : t.text } });
  }));
  const [min, max, colors] = mode === 'effect' ? [err ? -0.5 : -0.25, err ? 0.5 : 0.25, [t.good, t.surface, t.bad]]
    : [0, 1, [alpha(t.good, 0.55), t.surface, t.bad]];
  return {
    animation: false,
    grid: { left: 8, right: 8, top: 8, bottom: 44, containLabel: true },
    tooltip: { backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 },
               formatter: (p: { value: [number, number, number, number] }) => `<b>${ys[p.value[1]]} · ${xs[p.value[0]].replace('\n', ' ')}</b><br/>` +
                 (mode === 'effect' ? fmtEffect(p.value[3], m) : `${SMETRICS[m].short} ${p.value[3].toFixed(3)}${SMETRICS[m].unit ? ' ' + SMETRICS[m].unit : ''}`) },
    xAxis: { type: 'category', data: xs, ...axisCommon(t), name: 'position: most plantarflexed (left) → most dorsiflexed (right)', nameLocation: 'middle', nameGap: 30, nameTextStyle: { color: t.textMuted, fontSize: 11 } },
    yAxis: { type: 'category', data: ys, ...axisCommon(t), inverse: true },
    visualMap: { show: false, type: 'continuous', dimension: 2, min, max, inRange: { color: colors } },
    series: [{
      type: 'heatmap', data,
      label: { show: true, fontSize: 11, color: t.text, formatter: (p: { value: [number, number, number, number] }) => (mode === 'effect' ? fmtEffect(p.value[3], m) : p.value[3].toFixed(2)) },
      itemStyle: { borderColor: t.surface, borderWidth: 2, borderRadius: 3 },
    }],
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
        return `<b>${subj} · position ${k + 1} (${posLabel(positions[k])} rad)${k < 2 ? ' · plantarflexed' : ''}</b>`
          + sides.map((s) => seedTooltip(s, k, 3, unit)).join('');
      },
    },
    xAxis: [
      { type: 'category', data: positions.map((p, i) => `p${i + 1}\n${posLabel(p)}`), ...axisCommon(t),
        axisLabel: compact ? { color: t.textMuted, fontSize: 9, interval: 0, lineHeight: 12 }
          : { color: t.text, fontSize: 12, interval: 0, lineHeight: 16, rich: ciRich(t), formatter: (v: string, k: number) => `${v}\n${labels[k] ?? ''}` },
        ...(compact ? {} : { name: 'ankle position (rad) · median [interval]', nameLocation: 'middle', nameGap: 76, nameTextStyle: { color: t.textMuted, fontSize: 12 } }) },
      { type: 'value', min: -0.5, max: positions.length - 0.5, show: false },
    ],
    yAxis: [{
      type: 'value', min: lo, max: hi, ...axisCommon(t), axisLine: { onZero: false, lineStyle: { color: t.axis } }, name: compact ? undefined : `${SMETRICS[m].short}${unit}`,
      nameTextStyle: { color: t.textMuted, fontSize: compact ? 10 : 12, align: 'left' },
      axisLabel: { color: compact ? t.textMuted : t.text, fontSize: compact ? 10 : 12, formatter: (v: number) => String(+v.toFixed(err ? 2 : 3)) },
    }],
    series: [
      { type: 'scatter', xAxisIndex: 1, data: [], silent: true,
        markArea: { silent: true, itemStyle: { color: t.shade }, data: positions.length >= 2 ? [[{ xAxis: -0.5 }, { xAxis: 1.5 }]] : [] } },
      ...sides.flatMap((s) => seedSeries(s, t, { mode, compact, boxWidth: compact ? [4, 14] : [10, 34],
        clamp: (v) => (off(v) ? lo : v), hidden: off })),
      ...(below.length ? [{ type: 'scatter', name: 'below the axis', xAxisIndex: 1, z: 8, symbol: 'triangle', symbolRotate: 180, symbolSize: 12,
        data: below, label: { show: true, position: 'top', color: t.text, fontSize: 12, fontWeight: 600 } }] : []),
    ],
  };
}

// ── training time or epochs per seed ─────────────────────────────────────────
/** x axis: every seed by name; one dot per person; a bar at the median over people */
export function perSeedOption(seeds: number[], values: Record<string, Record<string, number | null | undefined>>, unit: 'min' | 'epochs', t: ChartTokens): EOption {
  const v = (x: number | null | undefined) => (typeof x === 'number' ? (unit === 'min' ? x / 60 : x) : null);
  const med = seeds.map((s) => {
    const xs = PEOPLE.map((p) => v(values[p]?.[String(s)])).filter((x): x is number => x !== null).sort((a, b) => a - b);
    return xs.length ? Math.round(((xs[(xs.length - 1) >> 1] + xs[xs.length >> 1]) / 2) * 10) / 10 : null;
  });
  const label = unit === 'min' ? 'training time (min), median over people' : 'epochs trained, median over people';
  return {
    animation: false,
    grid: { left: 8, right: 12, top: 30, bottom: 30, containLabel: true },
    tooltip: {
      trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 },
      formatter: (ps: { dataIndex: number }[]) => {
        const k = ps[0]?.dataIndex ?? -1;
        if (k < 0) return '';
        const per = PEOPLE.map((p) => { const x = v(values[p]?.[String(seeds[k])]); return `${p}: ${x === null ? '–' : x.toFixed(unit === 'min' ? 1 : 0)}`; });
        return `<b>seed ${seeds[k]}</b> · median ${med[k] ?? '–'}<br/>${per.join('<br/>')}`;
      },
    },
    xAxis: { type: 'category', data: seeds.map((s) => `seed ${s}`), ...axisCommon(t), axisLabel: { color: t.text, fontSize: 11, interval: 0 } },
    yAxis: { type: 'value', ...axisCommon(t), min: 0, name: label, nameTextStyle: { color: t.textMuted, fontSize: 11, align: 'left' } },
    series: [{ type: 'bar', data: med, barWidth: '50%', itemStyle: { color: t.a }, label: { show: true, position: 'top', color: t.text, fontSize: 10 } }],
  };
}

// ── measured and predicted torque ────────────────────────────────────────────
export interface TorqueLine { name: string; color: string; width: number; t0: number; y: number[]; dash?: boolean }

export function torqueOption(lines: TorqueLine[], hz: number, t: ChartTokens, opts: { mini?: boolean; title?: string } = {}): EOption {
  const mini = !!opts.mini;
  const all = lines.flatMap((l) => l.y);
  return {
    animation: false,
    title: opts.title ? { text: opts.title, left: 0, top: 0, textStyle: { color: t.text, fontSize: 11, fontWeight: 600 } } : undefined,
    grid: { left: mini ? 2 : 8, right: mini ? 6 : 16, top: mini ? 22 : 40, bottom: mini ? 2 : 8, containLabel: true },
    legend: mini ? undefined : { top: 0, left: 0, icon: 'roundRect', itemWidth: 14, itemHeight: 3, textStyle: { color: t.text, fontSize: 12 }, data: lines.map((l) => l.name) },
    tooltip: mini ? undefined : {
      trigger: 'axis', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 },
      valueFormatter: (v: number) => (Number.isFinite(v) ? `${v.toFixed(2)} Nm` : '–'),
    },
    xAxis: {
      type: 'value', min: Math.min(...lines.map((l) => l.t0)), max: Math.max(...lines.map((l) => l.t0 + (l.y.length - 1) / hz)),
      ...axisCommon(t), splitLine: { show: false }, axisLabel: { show: !mini, color: t.textMuted, fontSize: 11, formatter: (v: number) => `${Math.round(v)} s` },
    },
    yAxis: {
      type: 'value', min: Math.floor(Math.min(...all)), max: Math.ceil(Math.max(...all)), ...axisCommon(t),
      axisLabel: { color: t.textMuted, fontSize: mini ? 9 : 11 }, splitNumber: mini ? 2 : 5,
    },
    series: lines.map((l) => ({
      type: 'line', name: l.name, showSymbol: false, silent: mini,
      lineStyle: { color: l.color, width: mini ? Math.max(0.8, l.width - 0.6) : l.width, type: l.dash ? [4, 3] : 'solid' }, itemStyle: { color: l.color },
      data: l.y.map((v, i) => [Math.round((l.t0 + i / hz) * 100) / 100, v]),
    })),
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
        name: 'median [interval], min', nameLocation: 'middle', nameGap: 62, nameTextStyle: { color: t.textMuted, fontSize: 12 } },
      { type: 'value', min: -0.5, max: people.length - 0.5, show: false },
    ],
    yAxis: { type: 'value', min: 0, ...axisCommon(t), axisLine: { onZero: false, lineStyle: { color: t.axis } }, name: 'minutes per run', nameTextStyle: { color: t.textMuted, fontSize: 12, align: 'left' },
             axisLabel: { color: t.text, fontSize: 12 } },
    series: seedSeries(side, t, { mode: 'box', boxWidth: [10, 40] }),
  };
}
