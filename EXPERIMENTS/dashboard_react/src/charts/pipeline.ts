// ECharts options for the signal-pipeline page.
import type { EOption } from '../components/EChart';
import type { ChartTokens } from '../theme';
import { alpha } from '../theme';
import { CHANNELS, PIPE, pairs, times, type PipelinePerson } from '../lib/pipeline';

/** one colour per EMG channel (GM, GL, SOL, TA) and the angle */
export const CHANNEL_COLORS = ['#2a6fc2', '#d4612b', '#1f8a4c', '#b8377f', '#9a7400'];

const axis = (t: ChartTokens) => ({
  axisLine: { lineStyle: { color: t.axis } }, axisTick: { lineStyle: { color: t.axis } },
  axisLabel: { color: t.textMuted, fontSize: 11 }, splitLine: { lineStyle: { color: t.grid } },
});
const tip = (t: ChartTokens) => ({ backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 12 } });
const fmt = (v: number, d = 3) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(d));

export const STAGE_TITLES = [
  '1 · Raw EMG',
  '2 · Offset removed, band-pass 30–300 Hz',
  '3 · Rectified (absolute value)',
  '4 · Envelope: 2 Hz low-pass',
  '5 · Model input: every 10th sample ÷ session-1 maximum',
];

/** a single time-series chart: one line (or one set of dots) against time */
function series1(t: ChartTokens, yName: string, x: number[], y: number[], color: string, opts: { dots?: boolean; scale?: boolean; shade?: [number, number]; zero?: boolean; xmin?: number; xmax?: number } = {}): EOption {
  return {
    animation: false,
    grid: { left: 64, right: 18, top: 26, bottom: 34 },
    tooltip: { ...tip(t), trigger: 'axis', formatter: (ps: { value: [number, number] }[]) => `${ps[0]?.value[0].toFixed(2)} s · ${fmt(ps[0]?.value[1] ?? NaN)}` },
    xAxis: { type: 'value', min: opts.xmin ?? x[0], max: opts.xmax ?? x[x.length - 1], ...axis(t),
             name: 'time (s)', nameLocation: 'middle', nameGap: 22, nameTextStyle: { color: t.textMuted, fontSize: 11 },
             axisLabel: { color: t.textMuted, fontSize: 11, formatter: (v: number) => `${+v.toFixed(2)}` } },
    yAxis: { type: 'value', scale: !!opts.scale, splitNumber: 3, ...axis(t), name: yName, nameTextStyle: { color: t.textMuted, fontSize: 11, align: 'left' },
             axisLabel: { color: t.textMuted, fontSize: 10, formatter: (v: number) => fmt(v, 2) } },
    series: [opts.dots
      ? { type: 'scatter', symbolSize: 4, itemStyle: { color }, data: pairs(x, y) }
      : { type: 'line', showSymbol: false, sampling: 'lttb', data: pairs(x, y), lineStyle: { width: 1, color }, itemStyle: { color },
          ...(opts.shade ? { markArea: { silent: true, itemStyle: { color: t.shade }, data: [[{ xAxis: opts.shade[0] }, { xAxis: opts.shade[1] }]] } } : {}),
          ...(opts.zero ? { markLine: { silent: true, symbol: 'none', lineStyle: { color: t.textMuted, type: 'dotted' }, label: { show: false }, data: [{ yAxis: 0 }] } } : {}) }],
  };
}

/** stage i (0–4) of one channel over the 2 s excerpt, as its own chart */
export function stageOption(p: PipelinePerson, ch: number, t: ChartTokens, i: number): EOption {
  const c = CHANNEL_COLORS[ch];
  const e = p.excerpt;
  const tx = times(e.t0, e.fs, e.raw[ch].length);
  const lim = { xmin: tx[0], xmax: tx[tx.length - 1] };
  if (i === 0) return series1(t, p.raw_unit, tx, e.raw[ch], c, { scale: true, ...lim });
  if (i === 1) return series1(t, p.raw_unit, tx, e.band[ch], c, { scale: true, ...lim });
  if (i === 2) return series1(t, p.raw_unit, tx, e.band[ch].map(Math.abs), c, lim);
  if (i === 3) return series1(t, p.raw_unit, tx, e.env[ch], c, lim);
  const ti = times(p.inputs.t0, p.inputs.fs, p.inputs.emg[ch].length);
  return series1(t, '÷ session-1 max', ti, p.inputs.emg[ch], c, { dots: true, ...lim });
}

/** raw-EMG power spectrum, the kept band shaded */
export function spectrumOption(p: PipelinePerson, ch: number, t: ChartTokens): EOption {
  const c = CHANNEL_COLORS[ch];
  const db = p.psd.db[ch];
  const f = db.map((_, i) => i * p.psd.f_step);
  return {
    animation: false,
    grid: { left: 52, right: 16, top: 26, bottom: 40 },
    tooltip: { ...tip(t), trigger: 'axis', formatter: (ps: { value: [number, number] }[]) => `${ps[0]?.value[0].toFixed(0)} Hz · ${ps[0]?.value[1].toFixed(1)} dB` },
    xAxis: { type: 'value', min: 0, max: 500, ...axis(t), name: 'frequency (Hz)', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: t.textMuted, fontSize: 11 } },
    yAxis: { type: 'value', min: -50, max: 0, ...axis(t), name: 'dB below peak', nameTextStyle: { color: t.textMuted, fontSize: 11, align: 'left' } },
    series: [{ type: 'line', showSymbol: false, data: pairs(f, db), lineStyle: { width: 1, color: c }, itemStyle: { color: c },
      markArea: { silent: true, itemStyle: { color: t.shade }, label: { show: true, color: t.textMuted, fontSize: 11, position: 'insideTop' },
                  data: [[{ xAxis: 30, name: 'kept: 30–300 Hz' }, { xAxis: 300 }]] } }],
  };
}

/** frequency response of one filter in use: 'band' (30–300 Hz band-pass) or 'lp' (2 Hz low-pass, 2nd order) */
export function filterOption(which: 'band' | 'lp', t: ChartTokens): EOption {
  const F = PIPE.filters;
  const y = which === 'band' ? F.band : F.lp2;
  return {
    animation: false,
    grid: { left: 52, right: 16, top: 26, bottom: 40 },
    tooltip: { ...tip(t), trigger: 'axis', formatter: (ps: { value: [number, number] }[]) => `${ps[0]?.value[0].toFixed(ps[0]?.value[0] < 10 ? 2 : 0)} Hz · ${ps[0]?.value[1].toFixed(1)} dB` },
    xAxis: { type: 'log', min: 0.1, max: 500, logBase: 10, ...axis(t), name: 'frequency (Hz, log)', nameLocation: 'middle', nameGap: 24, nameTextStyle: { color: t.textMuted, fontSize: 11 } },
    yAxis: { type: 'value', min: -80, max: 5, ...axis(t), name: 'gain (dB)', nameTextStyle: { color: t.textMuted, fontSize: 11, align: 'left' } },
    series: [{ type: 'line', showSymbol: false, data: pairs(F.f, y), lineStyle: { width: 1.6, color: t.ink }, itemStyle: { color: t.ink },
      markLine: { silent: true, symbol: 'none', lineStyle: { color: t.textMuted, type: 'dotted' }, label: { color: t.textMuted, fontSize: 10, formatter: '−3 dB' }, data: [{ yAxis: -3 }] } }],
  };
}

/** the whole trial, one chart per signal: 'emg' (channel ch), 'torque' (active torque, the label) or 'angle'; the excerpt shaded */
export function trialPanelOption(p: PipelinePerson, t: ChartTokens, what: 'emg' | 'torque' | 'angle', ch = 0): EOption {
  const v = p.trial_view;
  const tx = times(v.t0, v.fs, v.measured.length);
  const shade: [number, number] = [p.excerpt.t0, p.excerpt.t0 + p.excerpt.raw[0].length / p.excerpt.fs];
  if (what === 'emg') return series1(t, '÷ session-1 max', tx, v.emg[ch], CHANNEL_COLORS[ch], { shade });
  if (what === 'torque') return series1(t, 'Nm', tx, v.active, t.ink, { shade, zero: true });
  return series1(t, 'rad', tx, v.angle, CHANNEL_COLORS[4], { shade, scale: true });
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
                 inRange: { color: [t.surface, '#2a6fc2'] } },
    series: [
      { type: 'heatmap', data: emg, itemStyle: { borderColor: t.surface, borderWidth: 1 },
        label: { show: true, fontSize: 9, color: t.text, formatter: (q: { value: number[] }) => q.value[2].toFixed(2) } },
      { type: 'heatmap', data: ang, itemStyle: { color: alpha(CHANNEL_COLORS[4], 0.18), borderColor: t.surface, borderWidth: 1 },
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
