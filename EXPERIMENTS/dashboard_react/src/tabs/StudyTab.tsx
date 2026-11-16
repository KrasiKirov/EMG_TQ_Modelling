import { useMemo, useState } from 'react';
import { Badge, Group, SegmentedControl, Select, SimpleGrid, Stack, Table, Text, Title } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { EChart } from '../components/EChart';
import { Note, SectionCard } from '../components/common';
import { cellsOption, clickedRow, dotOption, gridOption, type GridCell } from '../charts/study';
import {
  band, cellValue, compare, fmtEffect, neutral, GRID_LENGTHS, GRID_OVERLAPS, gridCell, has, PEOPLE, personValue, questions, REF,
  setting, SMETRIC_KEYS, SMETRICS, STUDY, totalVsActive, trainWindows, type Comparison, type SMetric, type Target,
} from '../lib/study';
import { median } from '../lib/stats';
import { useView } from '../state';

const VERDICT_TEXT: Record<Comparison['verdict'], string> = { better: 'clearly better', worse: 'clearly worse', same: 'no difference', mixed: 'mixed' };
const VERDICT_COLOR: Record<Comparison['verdict'], string> = { better: 'teal', worse: 'red', same: 'gray', mixed: 'yellow' };

/** the chosen model's settings (the window-study reference) */
const CHOSEN: [string, string][] = [
  ['Network', 'LSTM 16 units × 2 layers, dropout 0 (chosen on session-1 validation loss)'],
  ['Training', 'Nadam, learning rate 0.01 (halved after 6 flat epochs), batch 64, early stop 15, at most 150 epochs'],
  ['Window', '200 ms = 20 steps of 10 ms; a new training window every 10 ms (95 % overlap); time order; state reset per window'],
  ['Target', "active torque: each session's own relaxed-hold passive torque subtracted"],
  ['EMG', 'causal envelope (2nd-order 2 Hz low-pass), ÷ session-1 maximum; first and last 1 s of each trial cut'],
  ['Validation', 'last 20 % of each session-1 trial, one window span after training (purged split)'],
];

function label(id: string) { return setting(id)?.title ?? id; }

export function StudyTab() {
  const { tokens } = useView();
  const narrow = !!useMediaQuery('(max-width: 48em)');
  const [m, setM] = useState<SMetric>('mae');
  const [target, setTarget] = useState<Target>('active');
  const [pick, setPick] = useState<string>(REF.active);
  const [cellMode, setCellMode] = useState<'value' | 'effect'>('value');
  const qs = useMemo(() => questions(target), [target]);

  const controls = (
    <Group gap="sm" wrap="wrap" className="study-controls">
      <SegmentedControl size="xs" aria-label="Metric" value={m} onChange={(v) => setM(v as SMetric)}
        data={SMETRIC_KEYS.map((k) => ({ value: k, label: SMETRICS[k].short }))} />
      <SegmentedControl size="xs" aria-label="Predicted torque" value={target} onChange={(v) => setTarget(v as Target)}
        data={[{ value: 'active', label: 'Active torque' }, { value: 'total', label: 'Total torque' }]} />
    </Group>
  );

  return (
    <Stack gap="lg" data-testid="study-page">
      <div className="page-heading">
        <Text size="sm" c="dimmed">Window study · session 1 → session 2 · {PEOPLE.length} people · built {STUDY.built}</Text>
        <Title order={2}>What changes the error, and what does not</Title>
        <Text size="sm" c="dimmed">
          Every setting changes one thing from the reference and is compared with it person by person, on the seeds both have.
          All scores are computed within each position (one trial at a fixed angle).
        </Text>
        {controls}
      </div>

      <Answer m={m} />
      {qs.map((q) => {
        const rows = q.rows.map((id) => ({ label: label(id), c: compare(id, q.ref, m), id }));
        return (
          <SectionCard key={q.key} title={q.title} testId={`question-${q.key}`}
            description={<>{q.ask} Compared with <b>{label(q.ref)}</b>. Each dot is one person; the bar is the median (green: clearly better, red: clearly worse); the shaded band is ±{SMETRICS[m].error ? '5 %' : '0.02'}, too small to mean anything with 6 people. The number after each name is how many people improved.</>}>
            <EChart label={`${q.title}: ${SMETRICS[m].label} against ${label(q.ref)}`} height={70 + 44 * rows.length}
              option={dotOption(rows, m, tokens, narrow)} onClick={(p) => { const r = rows[clickedRow(p)]; if (r) setPick(r.id); }} />
            {q.note && <Text size="xs" c="dimmed" mt={4}>{q.note}</Text>}
          </SectionCard>
        );
      })}

      <Grid m={m} target={target} />
      {target === 'active' && <TotalVsActive m={m} onPick={setPick} />}

      <SectionCard testId="study-cells" title="Person × position" description={<>Where the error is. Pick a setting here, or click any dot in the charts above. {cellMode === 'value' ? 'Values are medians over seeds.' : <>Effect against <b>{label(setting(pick)?.target === 'total' ? REF.total : REF.active)}</b> on matched seeds.</>}</>}
        actions={<SegmentedControl size="xs" aria-label="Cell values" value={cellMode} onChange={(v) => setCellMode(v as 'value' | 'effect')} data={[{ value: 'value', label: 'Values' }, { value: 'effect', label: 'vs reference' }]} />}>
        <Select mb="sm" aria-label="Setting" searchable value={pick} onChange={(v) => v && setPick(v)}
          data={Object.values(STUDY.settings).map((s) => ({ value: s.id, label: `${s.title}` }))} maxDropdownHeight={320} />
        <Cells exp={pick} m={m} mode={cellMode} />
      </SectionCard>
      <Note>
        R² is computed within each position: it asks how well the torque is tracked at a fixed angle. R² over all positions
        pooled mostly rewards getting each angle's torque level right (the angle is an input), so it is not shown: it even
        ranks total torque above active torque (0.97 vs 0.96), while every within-position score ranks it lower. A run's
        R² is the median over its positions, because at positions where the torque hardly changes a small offset sends R²
        far below 0; MAE and RMSE are averaged over positions. Demeaned R² removes each position's mean offset and scores
        the shape only.
      </Note>
    </Stack>
  );
}

function Answer({ m }: { m: SMetric }) {
  const ref = REF.active;
  const score = (k: SMetric) => median(PEOPLE.map((p) => personValue(ref, p, k)));
  const pos = STUDY.positions[PEOPLE[0]]?.length ?? 8;
  const worst = useMemo(() => {
    const per = Array.from({ length: pos }, (_, i) => median(PEOPLE.map((p) => cellValue(ref, p, i, 'r2'))));
    const i = per.indexOf(Math.min(...per));
    return { i, v: per[i] };
  }, [ref, pos]);
  const verdicts = useMemo(() => {
    const out: { title: string; text: string; v: Comparison['verdict'] }[] = [];
    for (const q of questions('active')) {
      const cs = q.rows.map((id) => compare(id, q.ref, m));
      const strong = cs.filter((c) => c.verdict === 'better' || c.verdict === 'worse');
      const range = cs.length > 1 ? `${fmtEffect(Math.min(...cs.map((c) => c.median)), m)} to ${fmtEffect(Math.max(...cs.map((c) => c.median)), m)}` : fmtEffect(cs[0].median, m);
      const v: Comparison['verdict'] = strong.some((c) => c.verdict === 'worse') ? 'worse' : strong.length ? 'better' : cs.some((c) => c.verdict === 'mixed') ? 'mixed' : 'same';
      const text = strong.length
        ? strong.map((c) => `${label(c.exp)}: ${VERDICT_TEXT[c.verdict]} (${fmtEffect(c.median, m)}, ${c.better}/${c.n})`).join('; ')
        : v === 'mixed' ? `small and not consistent across people (${range})` : `no clear difference (${range})`;
      out.push({ title: q.title, text, v });
    }
    const tv = totalVsActive().map((id) => compare(id, setting(id)!.pair!, m));
    if (tv.length) {
      const worse = tv.filter((c) => (SMETRICS[m].error ? c.median > 1 : c.median < 0)).length;
      const ref2 = compare(REF.total, REF.active, m);
      // over all paired settings: clearly worse only if nearly every pair is worse AND the typical pair is outside the band
      const typical = median(tv.map((c) => c.median));
      const outside = Math.abs(typical - neutral(m)) > band(m);
      out.push({ title: 'Total instead of active torque', v: !outside ? 'same' : worse >= tv.length - 2 ? 'worse' : worse <= 2 ? 'better' : 'mixed',
                 text: `worse in ${worse} of ${tv.length} paired settings, typically ${fmtEffect(typical, m)}; at the reference ${fmtEffect(ref2.median, m)} (${ref2.better}/${ref2.n} people better)` });
    }
    return out;
  }, [m]);
  return (
    <SectionCard testId="study-answer" title="The chosen model" description="The window-study reference with the tuned network. Nothing below beats it clearly; it is also the shortest and cheapest window.">
      <SimpleGrid cols={{ base: 2, sm: 5 }} spacing="sm" mb="md">
        {SMETRIC_KEYS.map((k) => (
          <div key={k} className="study-kpi" data-active={k === m || undefined}>
            <Text size="xs" c="dimmed">{SMETRICS[k].label}</Text>
            <Text fz={22} fw={600}>{score(k).toFixed(k.startsWith('r2') ? 2 : 2)}{SMETRICS[k].unit ? <Text span size="sm" c="dimmed"> {SMETRICS[k].unit}</Text> : null}</Text>
          </div>
        ))}
        <div className="study-kpi">
          <Text size="xs" c="dimmed">Worst position, R²</Text>
          <Text fz={22} fw={600}>{worst.v.toFixed(2)}</Text>
          <Text size="xs" c="dimmed">position {worst.i + 1} of {pos} (toes down)</Text>
        </div>
      </SimpleGrid>
      <Text size="xs" c="dimmed" mb="md">Median over the {PEOPLE.length} people of each person's median over 10 seeds.</Text>
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
        <Table withRowBorders={false} verticalSpacing={4} fz="sm">
          <Table.Tbody>
            {CHOSEN.map(([k, v]) => <Table.Tr key={k}><Table.Td c="dimmed" w={96} valign="top">{k}</Table.Td><Table.Td>{v}</Table.Td></Table.Tr>)}
          </Table.Tbody>
        </Table>
        <Stack gap={8}>
          <Text size="sm" fw={600}>Answers, on {SMETRICS[m].label}</Text>
          {verdicts.map((v) => (
            <Group key={v.title} gap="xs" wrap="nowrap" align="flex-start">
              <Badge variant="light" color={VERDICT_COLOR[v.v]} w={112} style={{ flex: 'none' }}>{VERDICT_TEXT[v.v]}</Badge>
              <Text size="sm"><b>{v.title}:</b> {v.text}</Text>
            </Group>
          ))}
        </Stack>
      </SimpleGrid>
    </SectionCard>
  );
}

function Grid({ m, target }: { m: SMetric; target: Target }) {
  const { tokens } = useView();
  const ref = REF[target];
  const xs = GRID_OVERLAPS.map((o) => o[1]);
  const ys = GRID_LENGTHS.map((l) => l[1]);
  const cells: GridCell[] = [];
  GRID_LENGTHS.forEach((L, y) => GRID_OVERLAPS.forEach((o, x) => {
    const id = gridCell(L, o[0], target);
    cells.push({ x, y, c: id ? compare(id, ref, m) : null, windows: id ? trainWindows(id) : NaN });
  }));
  return (
    <SectionCard testId="study-grid" title="Window length × overlap"
      description={<>Median effect against <b>{label(ref)}</b> and how many people improved. Green is better, red worse. “Full” is a new training window every 10 ms. Long windows with little overlap have very few training windows (10 s at 0 %: about 56, against 56,000 for the reference), so the grid shows the amount of training data as much as the window length.</>}>
      <EChart label={`Window length by overlap, ${SMETRICS[m].label}`} height={330} option={gridOption(cells, xs, ys, m, tokens)} />
    </SectionCard>
  );
}

function TotalVsActive({ m, onPick }: { m: SMetric; onPick: (id: string) => void }) {
  const { tokens } = useView();
  const narrow = !!useMediaQuery('(max-width: 48em)');
  const ids = totalVsActive();
  const rows = ids.map((id) => ({ id, label: label(setting(id)!.pair!), c: compare(id, setting(id)!.pair!, m) }));
  return (
    <SectionCard testId="study-total" title="Total torque instead of active torque"
      description="Every setting run twice: predicting total measured torque, and active torque (each session's relaxed-hold passive torque subtracted). Each row is total ÷ active for the same setting; above 1 (or below 0 for R²) means total torque is worse.">
      <EChart label={`Total versus active torque, ${SMETRICS[m].label}`} height={70 + 30 * rows.length} option={dotOption(rows, m, tokens, narrow)}
        onClick={(p) => { const r = rows[clickedRow(p)]; if (r) onPick(r.id); }} />
    </SectionCard>
  );
}

function Cells({ exp, m, mode }: { exp: string; m: SMetric; mode: 'value' | 'effect' }) {
  const { tokens } = useView();
  if (!has(exp)) return <Text size="sm" c="dimmed">No results for this setting.</Text>;
  const s = setting(exp)!;
  const ref = s.target === 'total' ? REF.total : REF.active;
  const n = STUDY.positions[PEOPLE[0]]?.length ?? 8;
  const xs = Array.from({ length: n }, (_, i) => `p${i + 1}`);
  const values = PEOPLE.map((subj) => Array.from({ length: n }, (_, i) => {
    if (mode === 'value') return cellValue(exp, subj, i, m);
    const rs = new Set(Object.keys(STUDY.runs[ref]?.[subj] ?? {}));
    const seeds = Object.keys(STUDY.runs[exp]?.[subj] ?? {}).filter((k) => rs.has(k));
    if (!seeds.length) return null;
    const a = cellValue(ref, subj, i, m, seeds), b = cellValue(exp, subj, i, m, seeds);
    return SMETRICS[m].error ? b / a : b - a;
  }));
  const seeds = Math.max(...PEOPLE.map((p) => Object.keys(STUDY.runs[exp]?.[p] ?? {}).length));
  const epochs = median(Object.values(STUDY.epochs[exp] ?? {}).flatMap((x) => Object.values(x)).filter((v): v is number => typeof v === 'number'));
  return (
    <>
      <Text size="xs" c="dimmed" mb={4}>
        {s.title}: {s.length_s >= 1 ? `${s.length_s} s` : `${s.length_s * 1000} ms`} window, {s.step_ms} ms steps, {s.overlap_pct} % overlap, {s.target} torque
        {s.state === 'carry' ? `, state carried (${s.chunk_s} s chunks)` : ''}{s.shuffle ? ', shuffled' : ''}, at most {s.epochs} epochs · {seeds} seeds · median {epochs} epochs · {trainWindows(exp).toLocaleString()} training windows
      </Text>
      <EChart label={`${s.title}: ${SMETRICS[m].label} per person and position`} height={300} option={cellsOption(values, xs, PEOPLE, m, mode, tokens)} />
    </>
  );
}
