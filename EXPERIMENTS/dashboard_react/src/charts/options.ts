// ECharts option builders (presentation only; numbers come from src/lib).
import type { EOption, EParams } from '../components/EChart';
import type { BiasVarianceRow, CurveSeries, PositionStat } from '../lib/aggregate';
import { fmt, posLabel } from '../lib/format';
import { METRICS, metricLabel } from '../lib/metrics';
import type { MetricKey } from '../lib/types';
import { alpha, FONT, type ChartTokens } from '../theme';

const BOX_OFFSET = 0.23; // ECharts puts 2 box plot series at about ±0.23 of the category band

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] as string);
}

function sw(color: string, line = false): string {
  return line
    ? `<span style="display:inline-block;width:14px;height:0;border-top:2px solid ${color};vertical-align:middle"></span>`
    : `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${color}"></span>`;
}

/** Vertical zoom for box plots: a slim slider on the right edge, plus Shift + mouse wheel over the plot.
 *  filterMode 'none' keeps every box and point; only the visible range changes. */
export function yZoom(t: ChartTokens, top: number, bottom: number) {
  return [
    { type: 'slider', yAxisIndex: 0, filterMode: 'none', width: 12, right: 4, top, bottom, showDetail: false, showDataShadow: false,
      brushSelect: false, moveHandleSize: 0, borderColor: t.grid, backgroundColor: 'transparent', fillerColor: alpha(t.a, 0.18),
      handleStyle: { color: t.surface, borderColor: t.textMuted } },
    { type: 'inside', yAxisIndex: 0, filterMode: 'none', zoomOnMouseWheel: 'shift', moveOnMouseWheel: false, moveOnMouseMove: false },
  ];
}

function base(t: ChartTokens) {
  return {
    backgroundColor: 'transparent',
    animationDuration: 250,
    animationDurationUpdate: 200,
    textStyle: { fontFamily: FONT, color: t.text },
    tooltip: {
      backgroundColor: t.tooltipBg,
      borderColor: t.tooltipBorder,
      borderWidth: 1,
      padding: [8, 10],
      textStyle: { color: t.text, fontSize: 13, fontFamily: FONT },
      extraCssText: `border-radius:8px;box-shadow:0 6px 24px ${t.dark ? 'rgba(0,0,0,.45)' : 'rgba(20,25,35,.12)'};`,
      confine: true,
    },
  };
}

function valueAxis(t: ChartTokens, extra: Record<string, unknown> = {}) {
  return {
    type: 'value',
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: t.textMuted, fontSize: 12 },
    splitLine: { lineStyle: { color: t.grid, width: 1 } },
    nameTextStyle: { color: t.textMuted, fontSize: 12 },
    ...extra,
  };
}

// ── per-position box plot (A vs B) ───────────────────────────────────────────
export interface BoxInput {
  subj: string;
  positions: number[];
  peaks: number[];
  A: { exp: string | null; stats: PositionStat[] };
  B: { exp: string | null; stats: PositionStat[] };
  metric: MetricKey;
  floor: boolean;
  compact: boolean;
  /** phone-width chart: short peak labels */
  narrow?: boolean;
}

/** values beyond these limits are drawn at the limit (▼ / ▲ markers carry the true median) */
export interface BoxClip {
  low?: number;
  high?: number;
}

/** 1, 2, 2.5, 5 × 10^k at or above v */
function niceCeil(v: number): number {
  if (!(v > 0)) return v;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  const f = v / e;
  const n = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return n * e;
}

/**
 * A limit that keeps the bulk of the boxes readable when one or two boxes are far out
 * (e.g. JM's 12 Nm MAE at -0.08 rad next to boxes below 2 Nm). Undefined = no clipping needed.
 */
function robustLimit(magnitudes: number[]): number | undefined {
  const v = magnitudes.filter((x) => Number.isFinite(x) && x > 0).sort((a, b) => b - a);
  if (v.length < 6) return undefined;
  const cand = niceCeil(1.3 * v[2]);
  return v[0] > cand * 1.15 ? cand : undefined;
}

export function boxRange(inp: BoxInput): { lo?: number; hi?: number; clip: BoxClip } {
  const { metric, floor } = inp;
  const md = METRICS[metric];
  const all: number[] = [];
  const whiskLo: number[] = [];
  const whiskHi: number[] = [];
  for (const side of [inp.A, inp.B])
    side.stats.forEach((s) => {
      s.points.forEach((p) => all.push(p.v));
      if (s.summary.n) {
        whiskLo.push(s.summary.p5);
        whiskHi.push(s.summary.p95);
      }
    });
  const min = all.length ? Math.min(...all) : 0;
  const max = all.length ? Math.max(...all) : 0;
  const clip: BoxClip = {};
  let lo: number | undefined;
  let hi: number | undefined;
  if (md.floorable) {
    hi = 1;
    if (floor) lo = 0;
    else {
      lo = Math.max(-3, Math.floor(Math.min(0, min) * 2) / 2);
      if (min < lo) clip.low = lo;
    }
  } else if (metric === 'mae') {
    lo = 0;
    const lim = robustLimit(whiskHi);
    if (lim !== undefined && max > lim) {
      hi = lim;
      clip.high = lim;
    }
  } else {
    // bias: one symmetric limit from the sizes of both whisker ends
    const lim = robustLimit([...whiskLo.map((v) => -v), ...whiskHi]);
    if (lim !== undefined) {
      if (min < -lim) lo = clip.low = -lim;
      if (max > lim) hi = clip.high = lim;
    }
  }
  return { lo, hi, clip };
}

export function boxOption(inp: BoxInput, t: ChartTokens): { option: EOption; clip: BoxClip } {
  const { positions, metric, compact } = inp;
  const narrow = !!inp.narrow;
  const n = positions.length;
  const md = METRICS[metric];
  const { lo, hi, clip } = boxRange(inp);
  const clamp = (v: number) => (lo !== undefined && v < lo ? lo : hi !== undefined && v > hi ? hi : v);
  const isLow = (v: number) => clip.low !== undefined && v < clip.low;
  const isHigh = (v: number) => clip.high !== undefined && v > clip.high;

  const beyond = (v: number) => isLow(v) || isHigh(v);
  const boxSeries = (which: 'A' | 'B', color: string) => {
    const side = inp[which];
    return {
      type: 'boxplot',
      name: which,
      xAxisIndex: 0,
      z: 3,
      boxWidth: compact ? [5, 18] : [8, 34],
      itemStyle: { color: alpha(color, t.dark ? 0.22 : 0.16), borderColor: color, borderWidth: 1.5 },
      emphasis: { itemStyle: { color: alpha(color, 0.3), borderColor: color, borderWidth: 2 } },
      data: positions.map((_, k) => {
        const s = side.stats[k];
        if (!s || !s.summary.n) return '-';
        const q = s.summary;
        // a box whose median is off the axis is not drawn; its marker at the edge carries the median
        if (beyond(q.p50)) return '-';
        return [q.p5, q.p25, q.p50, q.p75, q.p95].map(clamp);
      }),
    };
  };
  const dotSeries = (which: 'A' | 'B', color: string) => {
    const side = inp[which];
    const off = which === 'A' ? -BOX_OFFSET : BOX_OFFSET;
    const data: [number, number, number][] = [];
    side.stats.forEach((s, k) =>
      s.points.forEach((p, i) => {
        if (beyond(p.v)) return; // summarised by the edge marker
        const jitter = ((i % 5) - 2) * (compact ? 0.018 : 0.022);
        data.push([k + off + jitter, p.v, p.seed]);
      }),
    );
    return {
      type: 'scatter',
      name: `${which} runs`,
      xAxisIndex: 1,
      z: 4,
      symbolSize: compact ? 4 : 6,
      itemStyle: { color, opacity: 0.85, borderColor: t.surface, borderWidth: compact ? 0.6 : 1 },
      emphasis: { scale: 1.6 },
      data,
    };
  };
  // one triangle at the edge per box with runs off the axis: label = true median, or how many runs are off the axis
  const clipMarkers = (which: 'A' | 'B', color: string) => {
    const side = inp[which];
    const off = which === 'A' ? -BOX_OFFSET : BOX_OFFSET;
    const data: unknown[] = [];
    side.stats.forEach((s, k) => {
      if (!s.summary.n) return;
      for (const low of [true, false]) {
        const out = s.points.filter((p) => (low ? isLow(p.v) : isHigh(p.v)));
        if (!out.length) continue;
        const m = s.summary.p50;
        const medOut = low ? isLow(m) : isHigh(m);
        const text = medOut ? fmt(m, Math.abs(m) >= 10 ? 0 : 1) : `${out.length} of ${s.points.length}`;
        data.push({
          value: [k + off, (low ? clip.low : clip.high) as number, k],
          symbolRotate: low ? 180 : 0,
          label: {
            show: true,
            position: low ? 'top' : 'bottom',
            distance: 3,
            formatter: text,
            color: t.text,
            fontSize: compact ? 10 : 11,
            fontWeight: 600,
            backgroundColor: alpha(t.surface, 0.85),
            padding: [1, 3],
            borderRadius: 3,
          },
        });
      }
    });
    return {
      type: 'scatter',
      name: `${which} off axis`,
      xAxisIndex: 1,
      z: 6,
      symbol: 'triangle',
      symbolSize: compact ? 8 : 10,
      itemStyle: { color, opacity: 1, borderColor: t.surface, borderWidth: 1 },
      data,
    };
  };
  const shade = {
    type: 'scatter',
    name: 'toes-down',
    xAxisIndex: 1,
    data: [] as number[][],
    silent: true,
    markArea: {
      silent: true,
      itemStyle: { color: t.shade },
      label: { show: false },
      data: n >= 2 ? [[{ name: 'toes-down', xAxis: -0.5 }, { xAxis: 1.5 }]] : [],
    },
    // bias: the zero line is the target
    markLine:
      metric === 'bias'
        ? { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: t.textMuted, type: 'solid', width: 1 }, data: [{ yAxis: 0 }] }
        : undefined,
  };

  const tooltipFormatter = (params: EParams) => {
    const ps: EParams[] = Array.isArray(params) ? params : [params];
    let k = -1;
    for (const p of ps) if (p.seriesType === 'boxplot') k = p.dataIndex;
    if (k < 0 && ps.length) k = Math.round(Number(ps[0].axisValue ?? (Array.isArray(ps[0].value) ? ps[0].value[0] : NaN)));
    if (!(k >= 0 && k < n)) return '';
    const unit = md.unit ? ' ' + md.unit : '';
    const rows = (['A', 'B'] as const)
      .map((w) => {
        const side = inp[w];
        const s = side.stats[k];
        const color = w === 'A' ? t.a : t.b;
        if (!side.exp || !s || !s.summary.n)
          return `<tr><td>${sw(color)} <b>${w}</b></td><td colspan="4" class="muted">not trained for ${esc(inp.subj)}</td></tr>`;
        const q = s.summary;
        return `<tr><td>${sw(color)} <b>${w}</b></td><td class="r"><b>${fmt(q.p50)}</b>${unit}</td><td class="r">${fmt(q.p5)} to ${fmt(q.p95)}</td><td class="r">${fmt(q.p1)} to ${fmt(q.p99)}</td><td class="r">${q.n}</td></tr>`;
      })
      .join('');
    const pk = inp.peaks[k];
    return `<div class="tt"><div><b>${esc(inp.subj)} · ${posLabel(positions[k], 3)} rad</b>${k < 2 ? ' · toes-down' : ''}</div>
      <div class="muted">peak voluntary torque ${fmt(pk, 1)} Nm</div>
      <table style="margin-top:6px"><tr class="muted"><td></td><td class="r">median ${esc(md.label)}</td><td class="r">5–95 %</td><td class="r">1–99 %</td><td class="r">runs</td></tr>${rows}</table></div>`;
  };

  const showPeak = metric === 'mae';
  const shortPeak = compact || narrow;
  const peakText = (pk: number) => (shortPeak ? fmt(pk, pk < 10 ? 1 : 0) : `${fmt(pk, 1)} Nm`);
  const option = {
    ...base(t),
    grid: { left: compact ? 6 : 10, right: compact ? 24 : 30, top: compact ? 10 : 30, bottom: compact ? 4 : 22, containLabel: true },
    dataZoom: yZoom(t, compact ? 10 : 30, compact ? 30 : showPeak ? 74 : 56),
    tooltip: { ...base(t).tooltip, trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: alpha(t.dark ? '#ffffff' : '#000000', 0.035) } }, formatter: tooltipFormatter },
    xAxis: [
      {
        type: 'category',
        data: positions.map((p) => posLabel(p)),
        axisLine: { lineStyle: { color: t.axis } },
        axisTick: { show: false },
        axisLabel: {
          color: t.text,
          fontSize: compact || narrow ? 11 : 12,
          interval: 0,
          hideOverlap: false,
          formatter: showPeak ? (v: string, i: number) => `${v}\n{pk|${peakText(inp.peaks[i])}}` : undefined,
          rich: { pk: { color: t.textMuted, fontSize: compact || narrow ? 10 : 11, padding: [3, 0, 0, 0] } },
        },
        name: compact ? undefined : 'ankle position (rad)',
        nameLocation: 'middle',
        nameGap: showPeak ? 46 : 30,
        nameTextStyle: { color: t.textMuted, fontSize: 12 },
      },
      { type: 'value', min: -0.5, max: n - 0.5, show: false, axisPointer: { show: false } },
    ],
    yAxis: valueAxis(t, {
      min: lo,
      max: hi,
      name: compact ? undefined : metricLabel(metric),
      nameLocation: 'end',
      nameGap: 12,
      nameTextStyle: { color: t.textMuted, fontSize: 12, align: 'left' },
      splitNumber: compact ? 4 : 5,
      // no fixed interval: ticks must keep up when the reader zooms close to R² = 1
      axisLabel: {
        color: t.textMuted,
        fontSize: compact ? 11 : 12,
        formatter: (v: number) => String(+v.toFixed(3)),
      },
    }),
    series: [shade, boxSeries('A', t.a), boxSeries('B', t.b), dotSeries('A', t.a), dotSeries('B', t.b), clipMarkers('A', t.a), clipMarkers('B', t.b)],
  };
  return { option, clip };
}

// ── waveform viewer ──────────────────────────────────────────────────────────
export interface WaveLine {
  name: string;
  color: string;
  width: number;
  data: Float32Array;
}

export function waveOption(time: number[], lines: WaveLine[], zoom: { start: number; end: number }, t: ChartTokens, mobile: boolean, demeaned = false): EOption {
  return {
    ...base(t),
    animation: false,
    grid: { left: 8, right: 16, top: mobile ? 92 : 62, bottom: mobile ? 64 : 72, containLabel: true },
    legend: {
      top: 0,
      left: 0,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 3,
      itemGap: 18,
      textStyle: { color: t.text, fontSize: 13 },
      data: lines.map((l) => l.name),
    },
    tooltip: {
      ...base(t).tooltip,
      trigger: 'axis',
      axisPointer: { type: 'line', lineStyle: { color: t.textMuted, width: 1, type: 'solid' } },
      formatter: (params: EParams) => {
        const ps: EParams[] = Array.isArray(params) ? params : [params];
        if (!ps.length) return '';
        const x = Number(ps[0].value?.[0] ?? ps[0].axisValue);
        const rows = ps
          .map((p) => `<tr><td>${sw(p.color, true)}</td><td>${esc(p.seriesName)}</td><td class="r"><b>${fmt(Number(p.value[1]))}</b> Nm</td></tr>`)
          .join('');
        return `<div class="tt"><b>${fmt(x, 1)} s</b><table style="margin-top:4px">${rows}</table></div>`;
      },
    },
    xAxis: {
      type: 'value',
      min: time[0],
      max: time[time.length - 1],
      axisLine: { lineStyle: { color: t.axis } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { color: t.textMuted, fontSize: 12, formatter: (v: number) => `${+v.toFixed(1)} s` },
    },
    yAxis: valueAxis(t, { name: demeaned ? 'demeaned torque (Nm)' : 'voluntary torque (Nm)', nameLocation: 'end', nameGap: 12, nameTextStyle: { color: t.textMuted, fontSize: 12, align: 'left' }, scale: true }),
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, start: zoom.start, end: zoom.end, zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false },
      {
        type: 'slider',
        xAxisIndex: 0,
        start: zoom.start,
        end: zoom.end,
        height: 26,
        bottom: 10,
        borderColor: t.grid,
        backgroundColor: 'transparent',
        fillerColor: alpha(t.dark ? '#ffffff' : '#1b1b1b', 0.06),
        dataBackground: { lineStyle: { color: t.axis, width: 1 }, areaStyle: { color: alpha(t.axis, 0.25) } },
        selectedDataBackground: { lineStyle: { color: t.textMuted, width: 1 }, areaStyle: { color: alpha(t.textMuted, 0.2) } },
        handleStyle: { color: t.surface, borderColor: t.textMuted },
        moveHandleStyle: { color: t.axis },
        textStyle: { color: t.textMuted, fontSize: 11 },
        labelFormatter: (v: number) => `${fmt(v, 1)} s`,
      },
    ],
    series: lines.map((l, i) => ({
      type: 'line',
      name: l.name,
      showSymbol: false,
      symbol: 'circle',
      symbolSize: 5,
      z: 10 - i,
      lineStyle: { color: l.color, width: l.width, join: 'round', cap: 'round' },
      itemStyle: { color: l.color },
      emphasis: { focus: 'none', lineStyle: { width: l.width } },
      data: time.map((x, j) => [x, Number.isFinite(l.data[j]) ? Math.round(l.data[j] * 100) / 100 : null]),
    })),
  };
}

// ── learning curves ──────────────────────────────────────────────────────────
export function learningOption(
  subj: string,
  ours: CurveSeries | null,
  last: CurveSeries | null,
  bestOurs: number | null,
  t: ChartTokens,
  compact: boolean,
  yMax?: number,
): EOption {
  const split = [
    ['fit', 'fit', t.fit],
    ['val', 'validation', t.val],
    ['test', 'test', t.test],
  ] as const;
  const series: Record<string, unknown>[] = [];
  for (const [model, curve, dashed] of [
    ['last semester', last, true],
    ['ours', ours, false],
  ] as const) {
    if (!curve) continue;
    for (const [k, label, color] of split)
      series.push({
        type: 'line',
        name: `${label} · ${model}`,
        showSymbol: false,
        symbolSize: 5,
        lineStyle: { color, width: dashed ? 1.5 : 2, type: dashed ? [5, 4] : 'solid', opacity: dashed ? 0.9 : 1 },
        itemStyle: { color },
        emphasis: { disabled: true },
        data: curve.epochs.map((e, i) => [e, curve[k][i]]),
        markLine:
          !dashed && k === 'fit' && bestOurs !== null
            ? {
                silent: true,
                symbol: 'none',
                lineStyle: { color: t.textMuted, type: 'solid', width: 1 },
                label: { formatter: `kept: epoch ${bestOurs}`, color: t.textMuted, fontSize: 11, position: 'end', distance: 4 },
                data: [{ xAxis: bestOurs }],
              }
            : undefined,
      });
  }
  return {
    ...base(t),
    animation: false,
    grid: { left: 4, right: compact ? 12 : 18, top: 34, bottom: 4, containLabel: true },
    tooltip: {
      ...base(t).tooltip,
      trigger: 'axis',
      axisPointer: { type: 'line', lineStyle: { color: t.textMuted, width: 1 } },
      formatter: (params: EParams) => {
        const ps: EParams[] = Array.isArray(params) ? params : [params];
        if (!ps.length) return '';
        const ep = ps[0].value?.[0];
        const rows = ps
          .map((p) => `<tr><td>${sw(p.color, true)}</td><td>${esc(p.seriesName)}</td><td class="r"><b>${fmt(Number(p.value[1]), 3)}</b></td></tr>`)
          .join('');
        return `<div class="tt"><b>${esc(subj)} · epoch ${ep}</b> <span class="muted">MAE (Nm)</span><table style="margin-top:4px">${rows}</table></div>`;
      },
    },
    xAxis: {
      type: 'value',
      min: 1,
      axisLine: { lineStyle: { color: t.axis } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { color: t.textMuted, fontSize: 11 },
      name: compact ? undefined : 'epoch',
      nameLocation: 'middle',
      nameGap: 26,
      nameTextStyle: { color: t.textMuted, fontSize: 12 },
    },
    yAxis: valueAxis(t, {
      min: 0,
      max: yMax,
      name: undefined, // unit is in the card title (a name here collided with the 'kept: epoch' label)
      nameLocation: 'end',
      nameGap: 8,
      nameTextStyle: { color: t.textMuted, fontSize: 11, align: 'left' },
      splitNumber: 4,
      axisLabel: { color: t.textMuted, fontSize: 11, formatter: (v: number) => String(+v.toFixed(2)) },
    }),
    series,
  };
}

// ── where the error comes from (stacked shares) ──────────────────────────────
/** components of one run's squared error, in stacking order (colours: chart tokens offset / otherSys / runToRun) */
export const BV_PARTS = [
  { key: 'offset', name: 'offset (whole curve shifted)', token: 'offset' },
  { key: 'other', name: 'other systematic (same in every run)', token: 'otherSys' },
  { key: 'run', name: 'run-to-run spread (random seed)', token: 'runToRun' },
] as const;

/** absolute = Nm² (share × MSE) instead of % of the MSE */
export function biasVarOption(rows: BiasVarianceRow[], t: ChartTokens, mobile: boolean, absolute = false): EOption {
  const label = (r: BiasVarianceRow) => `${r.subj} · ${r.model === 'original' ? 'last semester' : 'ours'}`;
  const cats = rows.map(label);
  const comp = [
    ['offset', BV_PARTS[0].name, t.offset, (r: BiasVarianceRow) => r.offsetShare],
    ['other', BV_PARTS[1].name, t.otherSys, (r: BiasVarianceRow) => r.otherSysShare],
    ['run', BV_PARTS[2].name, t.runToRun, (r: BiasVarianceRow) => r.runShare],
  ] as const;
  return {
    ...base(t),
    // the legend is HTML above the chart (it wrapped onto the bars on phones)
    grid: { left: 4, right: mobile ? 12 : 24, top: 8, bottom: 8, containLabel: true },
    tooltip: {
      ...base(t).tooltip,
      trigger: 'axis',
      axisPointer: { type: 'shadow', shadowStyle: { color: alpha(t.dark ? '#ffffff' : '#000000', 0.04) } },
      formatter: (params: EParams) => {
        const ps: EParams[] = Array.isArray(params) ? params : [params];
        if (!ps.length) return '';
        const r = rows[ps[0].dataIndex];
        if (!r) return '';
        const items = comp
          .map(([, name, color, f]) => `<tr><td>${sw(color)}</td><td>${esc(name)}</td><td class="r"><b>${fmt(f(r) * 100, 0)} %</b></td><td class="r muted">${fmt(f(r) * r.mse, 3)} Nm²</td></tr>`)
          .join('');
        return `<div class="tt"><b>${esc(label(r))}</b><div class="muted">mean squared error of one run: <b>${fmt(r.mse, 3)} Nm²</b> · ${r.runs} runs</div><table style="margin-top:4px">${items}</table></div>`;
      },
    },
    xAxis: absolute
      ? valueAxis(t, { min: 0, axisLabel: { color: t.textMuted, fontSize: 12, formatter: (v: number) => `${+v.toFixed(2)} Nm²` } })
      : valueAxis(t, { min: 0, max: 100, axisLabel: { color: t.textMuted, fontSize: 12, formatter: '{value} %' } }),
    yAxis: {
      type: 'category',
      inverse: true,
      data: cats,
      axisLine: { lineStyle: { color: t.axis } },
      axisTick: { show: false },
      axisLabel: {
        color: t.text,
        fontSize: 12,
        formatter: (v: string) => {
          const [s, m] = v.split(' · ');
          return `{s|${s}} {m|${m}}`;
        },
        rich: { s: { fontWeight: 650, color: t.text, width: 34 }, m: { color: t.textMuted } },
      },
    },
    series: comp.map(([key, name, color, f]) => ({
      type: 'bar',
      name,
      id: key,
      stack: 'share',
      barMaxWidth: 18,
      barCategoryGap: '38%',
      itemStyle: { color, borderColor: t.surface, borderWidth: 1 },
      emphasis: { disabled: true },
      data: rows.map((r) => (absolute ? Math.round(f(r) * r.mse * 10000) / 10000 : Math.round(f(r) * 1000) / 10)),
    })),
  };
}
