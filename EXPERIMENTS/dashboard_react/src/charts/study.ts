// ECharts options for the window-study page.
import type { EOption } from '../components/EChart';
import type { ChartTokens } from '../theme';
import { alpha } from '../theme';
import { band, fmtEffect, neutral, PEOPLE, SMETRICS, type Comparison, type SMetric } from '../lib/study';

/** one colour per person, readable in both themes */
export function personColors(t: ChartTokens): string[] {
  return t.dark ? ['#6aa8f0', '#f0915f', '#4cc38a', '#b493f0', '#e47bb8', '#d9b84a']
                : ['#2a6fc2', '#d4612b', '#1f8a4c', '#7a4fc4', '#b8377f', '#9a7400'];
}

const axisCommon = (t: ChartTokens) => ({
  axisLine: { lineStyle: { color: t.axis } }, axisTick: { lineStyle: { color: t.axis } },
  axisLabel: { color: t.textMuted, fontSize: 11 }, splitLine: { lineStyle: { color: t.grid } },
});

/**
 * Dot plot: one row per setting, one dot per person (its effect against the comparison), a bar at the median,
 * the neutral line (ratio 1 / difference 0) and a shaded no-difference band.
 */
export function dotOption(rows: { label: string; c: Comparison }[], m: SMetric, t: ChartTokens, narrow: boolean): EOption {
  const colors = personColors(t);
  const err = SMETRICS[m].error;
  const all = rows.flatMap((r) => r.c.people.map((p) => p.effect)).filter(Number.isFinite);
  const n0 = neutral(m), b = band(m);
  let lo = Math.min(n0 - 2 * b, ...all), hi = Math.max(n0 + 2 * b, ...all);
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  if (err) lo = Math.max(0.05, lo);
  // error ratios spanning more than 2.5× read better on a log axis (one person at 6× would squash the rest)
  const log = err && hi / lo > 2.5;
  const cats = rows.map((r) => `${r.label}   ${r.c.better}/${r.c.n}`);
  const verdictColor = (v: Comparison['verdict']) => (v === 'better' ? t.good : v === 'worse' ? t.bad : t.ink);
  return {
    animation: false,
    grid: { left: 8, right: 16, top: narrow ? 52 : 30, bottom: 34, containLabel: true },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { color: t.textMuted, fontSize: 11 },
              data: [...PEOPLE, 'median'] },
    tooltip: {
      trigger: 'item', backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 },
      formatter: (p: { seriesName: string; dataIndex: number; value: [number, number] }) => {
        const r = rows[p.value[1]];
        if (!r) return '';
        if (p.seriesName === 'median') {
          return `<b>${r.label}</b><br/>median ${fmtEffect(r.c.median, m)} · better for ${r.c.better}/${r.c.n} people<br/>` +
            r.c.people.map((x) => `${x.subj}: ${fmtEffect(x.effect, m)} (${x.b.toFixed(2)} vs ${x.a.toFixed(2)}, ${x.seeds} seeds)`).join('<br/>');
        }
        const x = r.c.people.find((q) => q.subj === p.seriesName);
        return x ? `<b>${r.label}</b> · ${x.subj}<br/>${SMETRICS[m].short}: ${x.b.toFixed(3)} vs ${x.a.toFixed(3)} → ${fmtEffect(x.effect, m)}<br/>${x.seeds} matched seeds` : '';
      },
    },
    xAxis: { type: log ? 'log' : 'value', ...(log ? { logBase: 2 } : {}), min: lo, max: hi, ...axisCommon(t),
             name: narrow ? (err ? `${SMETRICS[m].short} ratio, lower is better` : `${SMETRICS[m].short} difference, higher is better`)
               : err ? `${SMETRICS[m].short} ratio (setting ÷ comparison), lower is better` : `${SMETRICS[m].short} difference (setting − comparison), higher is better`,
             nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: t.textMuted, fontSize: 11 },
             axisLabel: { color: t.textMuted, fontSize: 11, formatter: (v: number) => (err ? `${+v.toFixed(2)}` : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`) } },
    yAxis: { type: 'category', data: cats, inverse: true, ...axisCommon(t), splitLine: { show: false },
             axisLabel: { color: t.text, fontSize: narrow ? 10 : 12, width: narrow ? 120 : 220, overflow: 'truncate' } },
    series: [
      ...PEOPLE.map((subj, i) => ({
        name: subj, type: 'scatter', symbolSize: 8, itemStyle: { color: colors[i % colors.length], opacity: 0.85 },
        data: rows.flatMap((r, k) => r.c.people.filter((x) => x.subj === subj).map((x) => [x.effect, k])),
        z: 3,
      })),
      {
        name: 'median', type: 'scatter', symbol: 'rect', symbolSize: [3, 20], z: 4,
        itemStyle: { color: t.ink },
        data: rows.map((r, k) => ({ value: [r.c.median, k], itemStyle: { color: verdictColor(r.c.verdict) } })),
        markLine: { silent: true, symbol: 'none', lineStyle: { color: t.textMuted, type: 'dashed' }, label: { show: false }, data: [{ xAxis: n0 }] },
        markArea: { silent: true, itemStyle: { color: t.shade }, data: [[{ xAxis: n0 - b }, { xAxis: n0 + b }]] },
      },
    ],
  };
}

/** row index of a click on a dot plot (a person's dot or the median bar) */
export function clickedRow(p: { value?: unknown }): number {
  return Array.isArray(p.value) ? Number(p.value[1]) : -1;
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
               formatter: (p: { dataIndex: number }) => { const g = cells[p.dataIndex]; return g?.c ? `${fmtEffect(g.c.median, m)}\n${g.c.better}/${g.c.n}` : '–'; } },
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
    xAxis: { type: 'category', data: xs, ...axisCommon(t), name: 'position: most toes-down (left) → most toes-up (right)', nameLocation: 'middle', nameGap: 30, nameTextStyle: { color: t.textMuted, fontSize: 11 } },
    yAxis: { type: 'category', data: ys, ...axisCommon(t), inverse: true },
    visualMap: { show: false, type: 'continuous', dimension: 2, min, max, inRange: { color: colors } },
    series: [{
      type: 'heatmap', data,
      label: { show: true, fontSize: 11, color: t.text, formatter: (p: { value: [number, number, number, number] }) => (mode === 'effect' ? fmtEffect(p.value[3], m) : p.value[3].toFixed(2)) },
      itemStyle: { borderColor: t.surface, borderWidth: 2, borderRadius: 3 },
    }],
  };
}
