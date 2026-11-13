import { useMemo, useRef } from 'react';
import ReactEChartsCore from 'echarts-for-react/esm/core';
import * as echarts from 'echarts/core';
import { BarChart, BoxplotChart, LineChart, ScatterChart } from 'echarts/charts';
import { DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

// tree-shaken ECharts: only what the explorer uses
echarts.use([BoxplotChart, ScatterChart, LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkAreaComponent, MarkLineComponent, CanvasRenderer]);

export type EOption = echarts.EChartsCoreOption;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type EParams = any;

const OPTS = { renderer: 'canvas' as const };

/**
 * Canvas chart that resizes with its container (echarts-for-react watches the element size).
 * Event handlers are kept in a ref so that a new handler does not re-create the chart.
 */
export function EChart({
  option,
  height,
  onClick,
  onDataZoom,
  label,
}: {
  option: EOption;
  height: number;
  onClick?: (p: EParams) => void;
  onDataZoom?: (p: EParams, chart: echarts.ECharts) => void;
  label: string;
}) {
  const clickRef = useRef(onClick);
  clickRef.current = onClick;
  const zoomRef = useRef(onDataZoom);
  zoomRef.current = onDataZoom;
  const events = useMemo(
    () => ({
      click: (p: EParams) => clickRef.current?.(p),
      datazoom: (p: EParams, chart: echarts.ECharts) => zoomRef.current?.(p, chart),
    }),
    [],
  );
  return (
    <div role="img" aria-label={label} style={{ width: '100%', height, position: 'relative' }}>
      <ReactEChartsCore echarts={echarts} option={option} notMerge lazyUpdate opts={OPTS} onEvents={events} style={{ width: '100%', height }} />
    </div>
  );
}
