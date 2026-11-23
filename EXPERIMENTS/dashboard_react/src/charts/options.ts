// ECharts option builders (presentation only; numbers come from src/lib).
import type { EOption, EParams } from '../components/EChart';
import { fmt } from '../lib/format';
import { FONT, type ChartTokens } from '../theme'; // ECharts puts 2 box plot series at about ±0.23 of the category band

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] as string);
}

function sw(color: string, line = false): string {
  return line
    ? `<span style="display:inline-block;width:14px;height:0;border-top:2px solid ${color};vertical-align:middle"></span>`
    : `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${color}"></span>`;
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
      extraCssText: 'border-radius:4px;box-shadow:none;',
      confine: true,
    },
  };
}

function valueAxis(t: ChartTokens, extra: Record<string, unknown> = {}) {
  return {
    type: 'value',
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: t.textMuted, fontSize: 13 },
    splitLine: { lineStyle: { color: t.grid, width: 1 } },
    nameTextStyle: { color: t.textMuted, fontSize: 13 },
    ...extra,
  };
}

// ── waveform viewer ──────────────────────────────────────────────────────────
export interface WaveLine {
  name: string;
  color: string;
  width: number;
  data: Float32Array;
  /** this line's own sample times (s); without it, the chart's shared time axis */
  time?: ArrayLike<number>;
}

export function waveOption(time: number[], lines: WaveLine[], t: ChartTokens, demeaned = false, yName?: string): EOption {
  return {
    ...base(t),
    animation: false,
    grid: { left: 8, right: 16, top: 62, bottom: 16, containLabel: true },
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
      axisLabel: { color: t.text, fontSize: 13, formatter: (v: number) => `${+v.toFixed(1)} s` },
    },
    yAxis: valueAxis(t, { name: yName ?? (demeaned ? 'demeaned torque (Nm)' : 'voluntary torque (Nm)'), nameLocation: 'end', nameGap: 12, nameTextStyle: { color: t.textMuted, fontSize: 13, align: 'left' }, scale: true }),
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
      data: Array.from(l.data, (v, j) => [l.time ? l.time[j] : time[j], Number.isFinite(v) ? Math.round(v * 100) / 100 : null]),
    })),
  };
}
