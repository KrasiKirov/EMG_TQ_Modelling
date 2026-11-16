// ECharts options for the signal-pipeline page.
import type { EOption } from '../components/EChart';
import type { ChartTokens } from '../theme';
import { alpha } from '../theme';
import { CHANNELS, PIPE, pairs, times, type PipelinePerson } from '../lib/pipeline';

/** one colour per EMG channel (GM, GL, SOL, TA) and the angle, readable in both themes */
export function channelColors(t: ChartTokens): string[] {
  return t.dark ? ['#6aa8f0', '#f0915f', '#4cc38a', '#e47bb8', '#d9b84a'] : ['#2a6fc2', '#d4612b', '#1f8a4c', '#b8377f', '#9a7400'];
}

const axis = (t: ChartTokens) => ({
  axisLine: { lineStyle: { color: t.axis } }, axisTick: { lineStyle: { color: t.axis } },
  axisLabel: { color: t.textMuted, fontSize: 11 }, splitLine: { lineStyle: { color: t.grid } },
});
const tip = (t: ChartTokens) => ({ backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 } });
const fmt = (v: number, d = 3) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(d));

/** stacked panels sharing one time axis */
function stack(n: number, height: number, top = 30, gap = 34, bottom = 36) {
  const h = (height - top - bottom - gap * (n - 1)) / n;
  return Array.from({ length: n }, (_, i) => ({ left: 64, right: 18, top: top + i * (h + gap), height: h }));
}

export const STAGE_TITLES = [
  '1 · Raw EMG',
  '2 · Offset removed, band-pass 30–300 Hz',
  '3 · Rectified |x|',
  '4 · Envelope: 2 Hz low-pass',
  '5 · Model input: every 10th sample ÷ session-1 maximum',
];

/** the five stages of one channel over the 2 s excerpt */
export function stagesOption(p: PipelinePerson, ch: number, t: ChartTokens, height: number): EOption {
  const c = channelColors(t)[ch];
  const e = p.excerpt;
  const tx = times(e.t0, e.fs, e.raw[ch].length);
  const ti = times(p.inputs.t0, p.inputs.fs, p.inputs.emg[ch].length);
  const grids = stack(5, height);
  const x0 = tx[0], x1 = tx[tx.length - 1];
  const line = (i: number, name: string, y: number[], color: string, extra: object = {}) => ({
    name, type: 'line', xAxisIndex: i, yAxisIndex: i, data: pairs(tx, y), showSymbol: false, sampling: 'lttb',
    lineStyle: { width: 0.8, color }, itemStyle: { color }, animation: false, ...extra,
  });
  const max = p.trainmax[ch];
  return {
    animation: false,
    grid: grids,
    tooltip: { ...tip(t), trigger: 'axis', axisPointer: { type: 'line' },
               formatter: (ps: { seriesName: string; value: [number, number]; color: string }[]) =>
                 `<b>${ps[0]?.value[0].toFixed(3)} s</b><br/>` + ps.map((q) => `<span style="color:${q.color}">●</span> ${q.seriesName}: ${fmt(q.value[1])}`).join('<br/>') },
    axisPointer: { link: [{ xAxisIndex: 'all' }], lineStyle: { color: t.textMuted } },
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1, 2, 3, 4], filterMode: 'none', zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false }],
    xAxis: grids.map((_, i) => ({ type: 'value', gridIndex: i, min: x0, max: x1, ...axis(t),
      axisLabel: { show: i === 4, color: t.textMuted, fontSize: 11, formatter: (v: number) => `${v.toFixed(2)}` },
      ...(i === 4 ? { name: 'time in the trial (s)', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: t.textMuted, fontSize: 11 } } : {}) })),
    yAxis: grids.map((_, i) => ({ type: 'value', gridIndex: i, scale: i < 2, ...axis(t), splitNumber: 2,
      axisLabel: { color: t.textMuted, fontSize: 10, formatter: (v: number) => fmt(v, 2) },
      name: STAGE_TITLES[i], nameLocation: 'end', nameGap: 8,
      nameTextStyle: { color: t.text, fontSize: 12, fontWeight: 600, align: 'left', padding: [0, 0, 0, -56] } })),
    series: [
      line(0, `raw (${p.raw_unit})`, e.raw[ch], c),
      line(1, 'band-passed', e.band[ch], c),
      line(2, 'rectified', e.band[ch].map(Math.abs), c),
      line(3, 'envelope, causal (used)', e.env[ch], t.ink, { lineStyle: { width: 2, color: t.ink } }),
      line(3, 'envelope, zero-phase 4th order (earlier rounds, uses the future)', e.env_zerophase4[ch], t.textMuted,
           { lineStyle: { width: 1.4, type: 'dashed', color: t.textMuted } }),
      line(4, 'envelope ÷ max (1 kHz)', e.env[ch].map((v) => v / max), alpha(c, 0.45), { lineStyle: { width: 1, color: alpha(c, 0.45) } }),
      { name: 'model input (100 Hz)', type: 'scatter', xAxisIndex: 4, yAxisIndex: 4, symbolSize: 4, itemStyle: { color: c },
        data: pairs(ti, p.inputs.emg[ch]) },
    ],
  };
}

/** raw-EMG power spectrum with the kept band, and the filters' frequency responses */
export function spectrumOption(p: PipelinePerson, ch: number, t: ChartTokens, narrow: boolean): EOption {
  const c = channelColors(t)[ch];
  const db = p.psd.db[ch];
  const f = db.map((_, i) => i * p.psd.f_step);
  const F = PIPE.filters;
  const grids = narrow ? [{ left: 52, right: 14, top: 34, height: 150 }, { left: 52, right: 14, top: 268, height: 150 }]
    : [{ left: 52, right: '53%', top: 34, bottom: 70 }, { left: '55%', right: 14, top: 34, bottom: 70 }];
  const name = (s: string) => ({ name: s, nameLocation: 'middle' as const, nameGap: 26, nameTextStyle: { color: t.textMuted, fontSize: 11 } });
  const title = (s: string) => ({ name: s, nameLocation: 'end' as const, nameGap: 10, nameTextStyle: { color: t.text, fontSize: 12, fontWeight: 600, align: 'left' as const, padding: [0, 0, 0, -40] } });
  return {
    animation: false,
    grid: grids,
    legend: { bottom: narrow ? undefined : 0, top: narrow ? 446 : undefined, left: narrow ? 0 : '55%', itemWidth: 14, itemHeight: 8,
              textStyle: { color: t.textMuted, fontSize: 11 }, data: ['band-pass 30–300 Hz', '2 Hz low-pass, 2nd order (used)', '2 Hz low-pass, 4th order'] },
    tooltip: { ...tip(t), trigger: 'axis', formatter: (ps: { seriesName: string; value: [number, number]; color: string }[]) =>
      `<b>${ps[0]?.value[0].toFixed(ps[0]?.value[0] < 10 ? 2 : 0)} Hz</b><br/>` + ps.map((q) => `<span style="color:${q.color}">●</span> ${q.seriesName}: ${q.value[1].toFixed(1)} dB`).join('<br/>') },
    xAxis: [
      { type: 'value', gridIndex: 0, min: 0, max: 500, ...axis(t), ...name('frequency (Hz)') },
      { type: 'log', gridIndex: 1, min: 0.1, max: 500, logBase: 10, ...axis(t), ...name('frequency (Hz, log scale)') },
    ],
    yAxis: [
      { type: 'value', gridIndex: 0, min: -50, max: 0, ...axis(t), ...title(`Raw ${CHANNELS[ch]} power (dB below its peak)`) },
      { type: 'value', gridIndex: 1, min: -80, max: 5, ...axis(t), ...title('How much each filter lets through (dB)') },
    ],
    series: [
      { name: `${CHANNELS[ch]} power`, type: 'line', xAxisIndex: 0, yAxisIndex: 0, showSymbol: false, data: pairs(f, db),
        lineStyle: { width: 1, color: c }, itemStyle: { color: c },
        markArea: { silent: true, itemStyle: { color: t.shade }, label: { show: true, color: t.textMuted, fontSize: 11, position: 'insideTop' },
                    data: [[{ xAxis: 30, name: 'kept: 30–300 Hz' }, { xAxis: 300 }]] } },
      { name: 'band-pass 30–300 Hz', type: 'line', xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, data: pairs(F.f, F.band),
        lineStyle: { width: 1.6, color: c }, itemStyle: { color: c } },
      { name: '2 Hz low-pass, 2nd order (used)', type: 'line', xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, data: pairs(F.f, F.lp2),
        lineStyle: { width: 2, color: t.ink }, itemStyle: { color: t.ink },
        markLine: { silent: true, symbol: 'none', lineStyle: { color: t.textMuted, type: 'dotted' }, label: { color: t.textMuted, fontSize: 10, formatter: '−3 dB (cutoff)' }, data: [{ yAxis: -3 }] } },
      { name: '2 Hz low-pass, 4th order', type: 'line', xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, data: pairs(F.f, F.lp4),
        lineStyle: { width: 1.4, type: 'dashed', color: t.textMuted }, itemStyle: { color: t.textMuted } },
    ],
  };
}

/** the whole trial: scaled EMG, measured and active torque, angle; the excerpt shaded */
export function trialOption(p: PipelinePerson, t: ChartTokens, height: number): EOption {
  const v = p.trial_view;
  const tx = times(v.t0, v.fs, v.measured.length);
  const cc = channelColors(t);
  const grids = stack(3, height, 30, 34, 36);
  const ex: [number, number] = [p.excerpt.t0, p.excerpt.t0 + p.excerpt.raw[0].length / p.excerpt.fs];
  const shade = { silent: true, itemStyle: { color: t.shade }, data: [[{ xAxis: ex[0] }, { xAxis: ex[1] }]] };
  const titles = ['EMG model inputs (÷ session-1 maximum)', 'Torque (Nm): measured and active (= measured − passive)', 'Ankle angle (rad)'];
  const ln = (i: number, name: string, y: number[], color: string, extra: object = {}) => ({
    name, type: 'line', xAxisIndex: i, yAxisIndex: i, data: pairs(tx, y), showSymbol: false, lineStyle: { width: 1.1, color }, itemStyle: { color }, ...extra,
  });
  return {
    animation: false,
    grid: grids,
    legend: { top: 0, right: 0, itemWidth: 14, itemHeight: 8, textStyle: { color: t.textMuted, fontSize: 11 },
              data: [...CHANNELS, 'measured torque', 'active torque (label)'] },
    tooltip: { ...tip(t), trigger: 'axis', formatter: (ps: { seriesName: string; value: [number, number]; color: string }[]) =>
      `<b>${ps[0]?.value[0].toFixed(2)} s</b><br/>` + ps.map((q) => `<span style="color:${q.color}">●</span> ${q.seriesName}: ${fmt(q.value[1])}`).join('<br/>') },
    axisPointer: { link: [{ xAxisIndex: 'all' }], lineStyle: { color: t.textMuted } },
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1, 2], filterMode: 'none', zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false }],
    xAxis: grids.map((_, i) => ({ type: 'value', gridIndex: i, min: tx[0], max: tx[tx.length - 1], ...axis(t),
      axisLabel: { show: i === 2, color: t.textMuted, fontSize: 11 },
      ...(i === 2 ? { name: 'time in the trial (s)', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: t.textMuted, fontSize: 11 } } : {}) })),
    yAxis: grids.map((_, i) => ({ type: 'value', gridIndex: i, scale: i === 2, splitNumber: 2, ...axis(t), name: titles[i], nameLocation: 'end', nameGap: 8,
      nameTextStyle: { color: t.text, fontSize: 12, fontWeight: 600, align: 'left', padding: [0, 0, 0, -56] } })),
    series: [
      ...CHANNELS.map((c, k) => ln(0, c, v.emg[k], cc[k], k === 0 ? { markArea: shade } : {})),
      ln(1, 'measured torque', v.measured, t.b, { markArea: shade,
        markLine: { silent: true, symbol: 'none', lineStyle: { color: t.textMuted, type: 'dotted' }, label: { show: false }, data: [{ yAxis: 0 }] } }),
      ln(1, 'active torque (label)', v.active, t.ink, { lineStyle: { width: 1.4, color: t.ink } }),
      ln(2, 'angle (rad)', v.angle, cc[4], { markArea: shade }),
    ],
  };
}

/** one input window: 20 rows (oldest at top) × [GM, GL, SOL, TA, angle] */
export function windowOption(rows: number[][], endTime: number, t: ChartTokens): EOption {
  const feats = [...CHANNELS, 'angle (rad)'];
  const steps = rows.map((_, i) => (i === rows.length - 1 ? 'now' : `${((i - rows.length + 1) * 10).toFixed(0)} ms`));
  const emg: [number, number, number][] = [];
  const ang: { value: [number, number, number] }[] = [];
  rows.forEach((r, y) => r.forEach((v, x) => (x < 4 ? emg.push([x, y, v]) : ang.push({ value: [x, y, v] }))));
  return {
    animation: false,
    grid: { left: 64, right: 12, top: 24, bottom: 30 },
    tooltip: { ...tip(t), formatter: (q: { value: [number, number, number] }) => `<b>${feats[q.value[0]]}</b>, ${steps[q.value[1]]} (t = ${(endTime + (q.value[1] - rows.length + 1) / 100).toFixed(2)} s)<br/>value ${q.value[2].toFixed(3)}` },
    xAxis: { type: 'category', data: feats, position: 'top', ...axis(t), splitLine: { show: false }, axisLabel: { color: t.text, fontSize: 11 } },
    yAxis: { type: 'category', data: steps, ...axis(t), axisLabel: { color: t.textMuted, fontSize: 10, interval: 0 } },
    visualMap: { show: false, type: 'continuous', dimension: 2, seriesIndex: 0, min: 0, max: 1,
                 inRange: { color: [t.surface, t.dark ? '#6aa8f0' : '#2a6fc2'] } },
    series: [
      { type: 'heatmap', data: emg, itemStyle: { borderColor: t.surface, borderWidth: 1 },
        label: { show: true, fontSize: 9, color: t.text, formatter: (q: { value: number[] }) => q.value[2].toFixed(2) } },
      { type: 'heatmap', data: ang, itemStyle: { color: alpha(channelColors(t)[4], 0.18), borderColor: t.surface, borderWidth: 1 },
        label: { show: true, fontSize: 9, color: t.text, formatter: (q: { value: number[] }) => q.value[2].toFixed(3) } },
    ],
  };
}

/** passive torque subtracted at each position, both sessions */
export function passiveOption(p: PipelinePerson, t: ChartTokens): EOption {
  const s = p.passive_by_session;
  const col: Record<string, string> = { S1: t.a, S2: t.b };
  return {
    animation: false,
    grid: { left: 52, right: 16, top: 30, bottom: 44 },
    legend: { top: 0, right: 0, textStyle: { color: t.textMuted, fontSize: 11 } },
    tooltip: { ...tip(t), trigger: 'item', formatter: (q: { seriesName: string; value: [number, number] }) => `<b>${q.seriesName}</b><br/>angle ${q.value[0].toFixed(3)} rad<br/>passive ${q.value[1].toFixed(2)} Nm` },
    xAxis: { type: 'value', scale: true, ...axis(t), name: 'ankle angle (rad): plantarflexed ← → dorsiflexed', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: t.textMuted, fontSize: 11 } },
    yAxis: { type: 'value', ...axis(t), axisLine: { onZero: false, lineStyle: { color: t.axis } }, name: 'passive torque (Nm)', nameTextStyle: { color: t.textMuted, fontSize: 11, align: 'left' } },
    series: Object.keys(s).sort().map((k) => ({
      name: k === 'S1' ? 'session 1' : 'session 2', type: 'line', symbolSize: 7, data: s[k].map((e) => [e.angle, e.passive]),
      lineStyle: { color: col[k], width: 1.6 }, itemStyle: { color: col[k] },
      ...(k === 'S1' ? { markLine: { silent: true, symbol: 'none', lineStyle: { color: t.textMuted, type: 'dotted' }, label: { show: false }, data: [{ yAxis: 0 }] } } : {}),
    })),
  };
}
