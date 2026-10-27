import { Group, RangeSlider, Stack, Text } from '@mantine/core';
import { fmt } from '../lib/format';
import type { Summary } from '../lib/types';

/**
 * The "Custom range" control of the box plots: a labelled two-handle slider for the visible value window.
 * The window is React state in the caller, so re-rendering a chart cannot reset it (the old in-chart
 * slider was lost on every re-render).
 */
export function RangeWindow({ domain, value, onChange, step, digits, unit, hidden, total }: {
  domain: [number, number]; value: [number, number]; onChange: (v: [number, number]) => void;
  step: number; digits: number; unit: string; hidden: number; total: number;
}) {
  return (
    <Stack gap={6} className="range-window">
      <Group gap="lg" align="center" wrap="wrap">
        <Text size="sm" fw={600} style={{ whiteSpace: 'nowrap' }}>Showing {fmt(value[0], digits)} to {fmt(value[1], digits)}{unit}</Text>
        <RangeSlider aria-label="Visible range" style={{ flex: 1, minWidth: 220, maxWidth: 560 }} min={domain[0]} max={domain[1]} step={step}
          minRange={step * 2} value={value} onChange={onChange} label={(v) => fmt(v, digits)} />
      </Group>
      <Text size="xs" c="dimmed">{hidden ? `${hidden} of ${total} runs fall outside this range and are not drawn; a box whose median is outside is hidden.` : 'Every run is inside this range.'}</Text>
    </Stack>
  );
}

/** Domain the slider can span: whole R² range (from 0 when negatives are floored), else the data rounded to the step. */
export function rangeDomain(values: number[], isR2: boolean, floored: boolean, step: number): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [0, 1];
  const lo = Math.floor(Math.min(...finite) / step) * step, hi = Math.ceil(Math.max(...finite) / step) * step;
  return isR2 ? [floored ? 0 : Math.min(0, lo), 1] : [Math.min(0, lo), Math.max(hi, lo + 2 * step)];
}

/** A box drawn inside the window: hidden when its median is outside, otherwise its whiskers are cut at the edges. */
export function windowBox(q: Summary, lo: number, hi: number): number[] | '-' {
  if (q.p50 < lo || q.p50 > hi) return '-';
  return [q.p5, q.p25, q.p50, q.p75, q.p95].map((v) => Math.min(hi, Math.max(lo, v)));
}
