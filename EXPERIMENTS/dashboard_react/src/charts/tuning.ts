// Charts of the tuning experiment (session-1 validation) and the overfitting check (validation vs session 2).
import type { EOption } from '../components/EChart';
import type { ChartTokens } from '../theme';
import type { OverfitPoint, TuningRun } from '../lib/v2';

const SIZES = ['8×1', '8×2', '16×1', '16×2', '32×1', '32×2', '64×1', '64×2'];
const sizeKey = (r: TuningRun) => `${r.units}×${r.layers}`;
const axis = (t: ChartTokens) => ({
  axisLine: { lineStyle: { color: t.axis } }, axisTick: { lineStyle: { color: t.axis } },
  axisLabel: { color: t.text, fontSize: 13 }, splitLine: { lineStyle: { color: t.grid } },
  nameTextStyle: { color: t.textMuted, fontSize: 13 },
});
const tooltip = (t: ChartTokens) => ({ trigger: 'item', confine: true, backgroundColor: t.tooltipBg, borderColor: t.tooltipBorder, textStyle: { color: t.text, fontSize: 13 } });
/** a fixed small offset per run so points of one size do not sit on top of each other */
const jitter = (i: number) => ((i * 0.618034) % 1 - 0.5) * 0.5;
const runTip = (r: TuningRun) => `<b>${r.cell} · ${r.round} · ${r.config}</b><br/>${sizeKey(r)}, dropout ${r.dropout.toFixed(3)}, learning rate ${r.lr.toPrecision(2)}<br/>validation nMAE ${r.score.toFixed(4)}`;

/** every tuning run's validation nMAE by network size; this setting's cell in colour, the cell winners ringed */
export function sizeOption(runs: TuningRun[], cell: string | null, isWinner: (id: string) => boolean, t: ChartTokens): EOption {
  const cats = SIZES.filter((s) => runs.some((r) => sizeKey(r) === s));
  // one position per run, shared by its dot and its winner ring
  const pos = new Map(runs.map((r, i) => [r, { value: [cats.indexOf(sizeKey(r)) + jitter(i), r.score], run: r }]));
  const pt = (r: TuningRun) => pos.get(r)!;
  const other = runs.filter((r) => r.cell !== cell), mine = runs.filter((r) => r.cell === cell);
  const fmt = (p: { data: { run: TuningRun } }) => runTip(p.data.run);
  return {
    animation: false,
    grid: { left: 8, right: 16, top: 40, bottom: 30, containLabel: true },
    legend: { top: 0, left: 0, textStyle: { color: t.text, fontSize: 13 } },
    tooltip: { ...tooltip(t), formatter: fmt },
    xAxis: { type: 'value', min: -0.5, max: cats.length - 0.5, ...axis(t), splitLine: { show: false },
      axisTick: { customValues: cats.map((_, i) => i), lineStyle: { color: t.axis } },
      axisLabel: { customValues: cats.map((_, i) => i), color: t.text, fontSize: 13, formatter: (v: number) => cats[Math.round(v)] ?? '' }, name: 'network: units × layers', nameLocation: 'middle', nameGap: 30 },
    yAxis: { type: 'value', scale: true, ...axis(t), axisLine: { onZero: false, lineStyle: { color: t.axis } }, name: 'validation nMAE (session 1)', nameLocation: 'end', nameGap: 12 },
    series: [
      { type: 'scatter', name: 'other cells', symbolSize: 7, itemStyle: { color: t.axis, opacity: 0.55 }, data: other.map(pt) },
      ...(mine.length ? [{ type: 'scatter', name: 'this setting\'s cell', symbolSize: 10, itemStyle: { color: t.a }, data: mine.map(pt) }] : []),
      { type: 'scatter', name: 'cell winners', symbol: 'circle', symbolSize: 16, itemStyle: { color: 'transparent', borderColor: t.b, borderWidth: 2 }, z: 5,
        data: runs.filter((r) => isWinner(r.id)).map(pt) },
    ],
  };
}

/** validation nMAE against one hyperparameter, every tuning run */
export function factorOption(runs: TuningRun[], key: 'dropout' | 'lr' | 'units', t: ChartTokens): EOption {
  const name = { dropout: 'dropout', lr: 'learning rate', units: 'units' }[key];
  const UNITS = [8, 16, 32, 64];
  const x = (r: TuningRun) => (key === 'units' ? UNITS.indexOf(r.units) + jitter(runs.indexOf(r)) * 0.6 : r[key]);
  return {
    animation: false,
    grid: { left: 8, right: 16, top: 30, bottom: 30, containLabel: true },
    tooltip: { ...tooltip(t), formatter: (p: { data: { run: TuningRun } }) => runTip(p.data.run) },
    xAxis: { type: key === 'lr' ? 'log' : 'value', ...axis(t), name, nameLocation: 'middle', nameGap: 30, splitLine: { show: false },
      ...(key === 'units' ? { min: -0.5, max: 3.5, axisTick: { customValues: [0, 1, 2, 3], lineStyle: { color: t.axis } }, axisLabel: { customValues: [0, 1, 2, 3], color: t.text, fontSize: 13, formatter: (v: number) => String(UNITS[Math.round(v)] ?? '') } } : {}),
      ...(key === 'lr' ? { axisLabel: { color: t.text, fontSize: 13, formatter: (v: number) => String(+v.toPrecision(1)) } } : {}) },
    yAxis: { type: 'value', scale: true, ...axis(t), axisLine: { onZero: false, lineStyle: { color: t.axis } }, name: 'validation nMAE', nameLocation: 'end', nameGap: 12 },
    series: [{ type: 'scatter', symbolSize: 7, itemStyle: { color: t.a, opacity: 0.6 }, data: runs.map((r) => ({ value: [x(r), r.score], run: r })) }],
  };
}

/** validation (session 1) against session-2 nMAE per setting, coloured by network size, with the y = x line */
export function overfitOption(pts: OverfitPoint[], current: string, t: ChartTokens): EOption {
  const sizes = [...new Set(pts.map((p) => p.size))].sort((a, b) => parseInt(a) - parseInt(b));
  const colours = [t.b, t.otherSys, t.a, t.good];
  const hi = Math.max(...pts.map((p) => Math.max(p.val, p.test))) * 1.08;
  return {
    animation: false,
    grid: { left: 8, right: 16, top: 40, bottom: 30, containLabel: true },
    legend: { top: 0, left: 0, textStyle: { color: t.text, fontSize: 13 } },
    tooltip: { ...tooltip(t), formatter: (p: { data: { pt?: OverfitPoint } }) => (p.data.pt
      ? `<b>${p.data.pt.title}</b><br/>${p.data.pt.size} · validation ${p.data.pt.val.toFixed(4)} → session 2 ${p.data.pt.test.toFixed(4)} (×${(p.data.pt.test / p.data.pt.val).toFixed(2)})` : '') },
    xAxis: { type: 'value', min: 0, max: +hi.toFixed(3), ...axis(t), name: 'validation nMAE (session 1)', nameLocation: 'middle', nameGap: 30 },
    yAxis: { type: 'value', min: 0, max: +hi.toFixed(3), ...axis(t), name: 'session-2 nMAE', nameLocation: 'end', nameGap: 12 },
    series: [
      { type: 'line', name: 'same error in both', showSymbol: false, silent: true, lineStyle: { color: t.axis, type: 'dashed' }, data: [[0, 0], [hi, hi]] },
      { type: 'line', name: 'twice the validation error', showSymbol: false, silent: true, lineStyle: { color: t.axis, type: 'dotted' }, data: [[0, 0], [hi / 2, hi]] },
      ...sizes.map((s, i) => ({ type: 'scatter', name: s, symbolSize: 12, itemStyle: { color: colours[i % colours.length] },
        data: pts.filter((p) => p.size === s).map((p) => ({ value: [p.val, p.test], pt: p,
          ...(p.id === current ? { symbolSize: 18, itemStyle: { color: colours[i % colours.length], borderColor: t.text, borderWidth: 2 } } : {}) })) })),
    ],
  };
}
