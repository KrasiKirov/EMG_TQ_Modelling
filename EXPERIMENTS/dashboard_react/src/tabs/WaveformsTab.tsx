import { useCallback, useEffect, useMemo, useRef } from 'react';
import { Badge, Chip, Group, SegmentedControl, Select, SimpleGrid, Stack, Text } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconArrowDown } from '../icons';
import { DS } from '../dataset';
import { near, positionsOf, runsOf, peakOf } from '../lib/data';
import { dirLabel, fmt, posLabel } from '../lib/format';
import { floorR2 } from '../lib/metrics';
import { friendly, resolve, shortId } from '../lib/params';
import { keyWave, pickRun, runWave, sortedByR2, timeAxis } from '../lib/waves';
import type { Run } from '../lib/types';
import { useView } from '../state';
import { waveOption, type WaveLine } from '../charts/options';
import { EChart } from '../components/EChart';
import { DataTable, EmptyState, Note, SectionCard, Swatch, type Column } from '../components/common';

const TIME = timeAxis(DS);

interface Side {
  which: 'A' | 'B';
  exp: string | null;
  runs: Run[];
  run: Run | null;
  choice: string;
}

export function WaveformsTab() {
  const { s, set, wpos, tokens, takeWaveScroll } = useView();
  const mobile = useMediaQuery('(max-width: 48em)');
  const positions = positionsOf(DS, s.wsubj, s.dir);
  // phones: start on the first 30 s of the 90 s trial (the full trace is too dense at 390 px)
  const zoom = useRef(
    typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(max-width: 48em)').matches ? { start: 0, end: 33 } : { start: 0, end: 100 },
  );
  const touch = typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(pointer: coarse)').matches : false;

  // opened from a heat-map cell or a box: bring the chart into view (below the sticky header and tab bar)
  useEffect(() => {
    if (!takeWaveScroll()) return;
    requestAnimationFrame(() => {
      const el = document.querySelector('[data-testid=wave-card]');
      if (!el) return;
      const top = el.getBoundingClientRect().top + window.scrollY - 64 - 56;
      window.scrollTo({ top: Math.max(0, top) });
    });
  }, [takeWaveScroll, s.wsubj, wpos]);

  const sides: Side[] = useMemo(
    () =>
      (['A', 'B'] as const).map((which) => {
        const res = resolve(DS, s[which], s.wsubj, s.dir);
        const runs = res && wpos !== null ? runsOf(DS, res.exp, s.wsubj, s.dir).filter((r) => near(r.pos, wpos)) : [];
        const choice = which === 'A' ? s.runA : s.runB;
        return { which, exp: res?.exp ?? null, runs, run: pickRun(runs, choice), choice };
      }),
    [s.A, s.B, s.wsubj, s.dir, wpos, s.runA, s.runB],
  );

  const lines = useMemo(() => {
    if (wpos === null) return [];
    const out: WaveLine[] = [];
    const key = keyWave(DS, s.wsubj, s.dir, wpos);
    if (key) out.push({ name: 'measured', color: tokens.ink, width: 1.8, data: key });
    for (const side of sides) {
      if (!side.exp || !side.run) continue;
      const w = runWave(DS, side.exp, s.wsubj, s.dir, side.run.seed, wpos);
      if (w) out.push({ name: `model ${side.which} · seed ${side.run.seed}`, color: side.which === 'A' ? tokens.a : tokens.b, width: 1.4, data: w });
    }
    return out;
  }, [sides, s.wsubj, s.dir, wpos, tokens]);

  const option = useMemo(() => waveOption(TIME, lines, zoom.current, tokens, !!mobile), [lines, tokens, mobile]);
  const onZoom = useCallback((_p: unknown, chart: { getOption: () => unknown }) => {
    const dz = (chart.getOption() as { dataZoom?: { start: number; end: number }[] }).dataZoom?.[0];
    if (dz) zoom.current = { start: dz.start, end: dz.end };
  }, []);

  const peak = wpos !== null ? peakOf(DS, s.wsubj, s.dir, wpos) : NaN;
  const k = wpos !== null ? positions.findIndex((p) => near(p, wpos)) : -1;

  return (
    <Stack gap="md">
      <SectionCard title="Choose a trial" description={`${dirLabel(s.dir, true)} · the measured voluntary torque is the same for both models`}>
        <Stack gap="md">
          <Stack gap={4}>
            <Text size="sm" fw={500}>
              Person
            </Text>
            <SegmentedControl
              value={s.wsubj}
              onChange={(v) => set({ wsubj: v, pos: positionsOf(DS, v, s.dir)[Math.max(0, k)] ?? null, subj: s.subj === 'All' ? 'All' : v })}
              data={DS.subjects}
              size="sm"
              style={{ alignSelf: 'flex-start', maxWidth: '100%' }}
              data-testid="wave-subj"
            />
          </Stack>
          <Stack gap={4}>
            <Text size="sm" fw={500}>
              Ankle position (rad)
            </Text>
            <Chip.Group multiple={false} value={wpos !== null ? String(wpos) : null} onChange={(v) => set({ pos: Number(v) })}>
              <Group gap={8}>
                {positions.map((p, i) => (
                  <Chip key={p} value={String(p)} size="sm" variant="outline" data-testid={`wave-pos-${i}`}>
                    {i < 2 ? <IconArrowDown size={13} style={{ marginRight: 4, verticalAlign: '-2px' }} /> : null}
                    {posLabel(p)}
                  </Chip>
                ))}
              </Group>
            </Chip.Group>
            <Text size="xs" c="dimmed">
              <IconArrowDown size={12} style={{ verticalAlign: '-2px' }} /> = toes-down (the 2 most plantarflexed positions)
            </Text>
          </Stack>
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
            {sides.map((side) => (
              <RunPicker key={side.which} side={side} onChange={(v) => set(side.which === 'A' ? { runA: v } : { runB: v })} floor={s.floor} />
            ))}
          </SimpleGrid>
        </Stack>
      </SectionCard>

      <SectionCard
        title={`${s.wsubj} · ${wpos !== null ? posLabel(wpos, 3) : '–'} rad${k >= 0 && k < 2 ? ' · toes-down' : ''}`}
        description={`voluntary torque over the 90 s trial · peak ${fmt(peak, 1)} Nm · ${touch ? 'drag the slider or pinch to zoom' : 'drag the slider or use the mouse wheel to zoom'}`}
        testId="wave-card"
      >
        {lines.length ? (
          <EChart option={option} height={mobile ? 340 : 420} label={`Measured and predicted torque for ${s.wsubj} at ${wpos} rad`} onDataZoom={onZoom} />
        ) : (
          <EmptyState title="No waveform for this choice" hint="This person/direction has no stored trial." />
        )}
        <Stack gap={6} mt="md">
          {sides.map((side) => (
            <Group key={side.which} gap={8} wrap="nowrap" align="flex-start" data-testid={`wave-caption-${side.which}`}>
              <span style={{ marginTop: 5 }}>
                <Swatch which={side.which} />
              </span>
              {side.run && side.exp ? (
                <Text size="sm" className="num">
                  <b>Model {side.which}</b> · {friendly(side.exp)} ({shortId(side.exp)}) · seed {side.run.seed} ({choiceLabel(side.choice)}) · R²{' '}
                  <b>{fmt(floorR2(side.run.r2, s.floor))}</b> · MAE <b>{fmt(side.run.mae)} Nm</b> of {fmt(peak, 1)} Nm peak · bias {fmt(side.run.bias)} Nm
                </Text>
              ) : (
                <Text size="sm" c="dimmed">
                  <b>Model {side.which}</b> is not trained for {s.wsubj} in this direction.
                </Text>
              )}
            </Group>
          ))}
        </Stack>
      </SectionCard>

      <RunsTable sides={sides} floor={s.floor} peak={peak} onPick={(w, seed) => set(w === 'A' ? { runA: `s${seed}` } : { runB: `s${seed}` })} />
    </Stack>
  );
}

function choiceLabel(c: string): string {
  if (c === 'best') return 'best run';
  if (c === 'worst') return 'worst run';
  if (c.startsWith('s')) return 'chosen';
  return 'typical run';
}

function RunPicker({ side, onChange, floor }: { side: Side; onChange: (v: string) => void; floor: boolean }) {
  const data = useMemo(() => {
    if (!side.runs.length) return [];
    const s = sortedByR2(side.runs);
    const typ = s[Math.floor((s.length - 1) / 2)];
    const best = s[s.length - 1];
    const worst = s[0];
    const r2 = (r: Run) => fmt(floorR2(r.r2, floor));
    return [
      {
        group: 'Summary runs',
        items: [
          { value: 'typical', label: `Typical (median R²) · seed ${typ.seed} · R² ${r2(typ)}` },
          { value: 'best', label: `Best · seed ${best.seed} · R² ${r2(best)}` },
          { value: 'worst', label: `Worst · seed ${worst.seed} · R² ${r2(worst)}` },
        ],
      },
      {
        group: 'Every run',
        items: [...side.runs].sort((a, b) => a.seed - b.seed).map((r) => ({ value: `s${r.seed}`, label: `Seed ${r.seed} · R² ${r2(r)} · MAE ${fmt(r.mae)} Nm` })),
      },
    ];
  }, [side.runs, floor]);
  const value = ['typical', 'best', 'worst'].includes(side.choice) || side.runs.some((r) => `s${r.seed}` === side.choice) ? side.choice : 'typical';
  return (
    <Select
      label={
        <Group gap={6} component="span" style={{ display: 'inline-flex' }}>
          <Swatch which={side.which} /> Model {side.which} run
        </Group>
      }
      description={side.exp ? `${friendly(side.exp)} · ${side.runs.length} runs` : 'not trained for this person/direction'}
      data={data}
      value={side.runs.length ? value : null}
      onChange={(v) => v && onChange(v)}
      disabled={!side.runs.length}
      placeholder="not trained"
      data-testid={`run-${side.which}`}
    />
  );
}

interface RunRow {
  which: 'A' | 'B';
  exp: string;
  run: Run;
  shown: boolean;
}

function RunsTable({ sides, floor, peak, onPick }: { sides: Side[]; floor: boolean; peak: number; onPick: (w: 'A' | 'B', seed: number) => void }) {
  const rows: RunRow[] = sides.flatMap((side) =>
    side.exp ? [...side.runs].sort((a, b) => a.seed - b.seed).map((r) => ({ which: side.which, exp: side.exp as string, run: r, shown: side.run?.seed === r.seed })) : [],
  );
  const columns: Column<RunRow>[] = [
    {
      key: 'model',
      label: 'Model',
      render: (r) => (
        <Group gap={8} wrap="nowrap">
          <Swatch which={r.which} />
          <Text size="sm" fw={600}>
            {r.which}
          </Text>
          <Text size="xs" c="dimmed">
            {shortId(r.exp)}
          </Text>
          {r.shown && (
            <Badge size="xs" variant="light" color="gray" style={{ flex: 'none' }}>
              shown
            </Badge>
          )}
        </Group>
      ),
      sort: (r) => r.which,
    },
    { key: 'seed', label: 'Seed', render: (r) => r.run.seed, sort: (r) => r.run.seed, align: 'right' },
    { key: 'r2', label: 'R²', render: (r) => fmt(floorR2(r.run.r2, floor)), sort: (r) => r.run.r2, align: 'right' },
    { key: 'mae', label: 'MAE (Nm)', render: (r) => fmt(r.run.mae), sort: (r) => r.run.mae, align: 'right' },
    { key: 'bias', label: 'Bias (Nm)', render: (r) => fmt(r.run.bias), sort: (r) => r.run.bias, align: 'right' },
    { key: 'r2dm', label: 'VAF (R² shape only)', render: (r) => fmt(floorR2(r.run.r2_demeaned, floor)), sort: (r) => r.run.r2_demeaned, align: 'right' },
  ];
  return (
    <SectionCard
      title="Every run at this position"
      description={`peak voluntary torque at this position ${fmt(peak, 1)} Nm · click a row to show that run · R² values below 0 follow the display rule`}
      testId="runs-table"
    >
      {rows.length ? (
        <DataTable
          columns={columns}
          rows={rows}
          rowKey={(r) => `${r.which}-${r.run.seed}`}
          rowTestId={(r) => `runs-row-${r.which}-${r.run.seed}`}
          onRowClick={(r) => onPick(r.which, r.run.seed)}
          rowClass={(r) => (r.shown ? 'selected-row' : undefined)}
          minWidth={560}
          dense
          caption="runs at the selected position"
        />
      ) : (
        <EmptyState compact title="Neither model is trained for this person/direction" />
      )}
      <Stack mt="sm">
        <Note>Typical run = the run with the median R² at this position (index ⌊(n−1)/2⌋ after sorting by raw R²).</Note>
      </Stack>
    </SectionCard>
  );
}
