import { memo, useMemo } from 'react';
import { Alert, Badge, Button, Card, Group, SimpleGrid, Stack, Text, Title, Tooltip } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconAlertTriangle } from '../icons';
import { DS } from '../dataset';
import { computeKpis, heatmap, positionStats, scopeSubjects, type HeatCell, type KpiGroup, type KpiSide, type Kpis } from '../lib/aggregate';
import { dirLabel, fmt, listJoin, posLabel, range, signed } from '../lib/format';
import { compareHigherBetter, compareLabel, deltaThreshold, METRICS } from '../lib/metrics';
import { modelCodes, modelName, modelStatus, modelTitle, resolve, shortId, type ModelStatus } from '../lib/params';
import { near, peakOf, positionsOf } from '../lib/data';
import { quantile } from '../lib/stats';
import type { Dir, MetricKey, Params, Summary } from '../lib/types';
import { useView } from '../state';
import { boxOption } from '../charts/options';
import { EChart } from '../components/EChart';
import { EmptyState, Note, SectionCard, Swatch, VerdictBadge } from '../components/common';
import { luminance, mix, type ChartTokens } from '../theme';

export function OverviewTab() {
  const { s } = useView();
  const subjects = useMemo(() => scopeSubjects(DS, s.subj), [s.subj]);
  const kpis = useMemo(() => computeKpis(DS, s.A, s.B, subjects, s.dir, s.metric, s.floor), [s.A, s.B, subjects, s.dir, s.metric, s.floor]);
  const stA = useMemo(() => modelStatus(DS, s.A, subjects, s.dir), [s.A, subjects, s.dir]);
  const stB = useMemo(() => modelStatus(DS, s.B, subjects, s.dir), [s.B, subjects, s.dir]);
  return (
    <Stack gap="md">
      <ComparisonStrip stA={stA} stB={stB} subjects={subjects} />
      <UntrainedAlert stA={stA} stB={stB} />
      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="md">
        <KpiCard id="toes" title="Toes-down positions" sub="the 2 most plantarflexed positions of each person" g={kpis.toes} k={kpis} stA={stA} stB={stB} metric={s.metric} />
        <KpiCard id="other" title="Other 6 positions" sub="all remaining test positions" g={kpis.other} k={kpis} stA={stA} stB={stB} metric={s.metric} />
      </SimpleGrid>
      <PerPositionCard stA={stA} stB={stB} />
      <HeatmapCard />
    </Stack>
  );
}

// ── what is being compared (always visible, also on phones where the left panel is hidden) ──
function ComparisonStrip({ stA, stB, subjects }: { stA: ModelStatus; stB: ModelStatus; subjects: string[] }) {
  const { s } = useView();
  const better = compareHigherBetter(s.metric) ? `higher ${compareLabel(s.metric, false)} is better` : `lower ${compareLabel(s.metric, false)} is better`;
  return (
    <Card padding="sm" radius="lg" data-testid="compare-strip">
      <div className="summary-strip">
        <span style={{ color: 'var(--text-3)' }}>Comparing</span>
        <span>
          <Swatch which="A" /> <b>A</b>&nbsp;{modelName(stA)}
        </span>
        <span>
          <Swatch which="B" /> <b>B</b>&nbsp;{modelName(stB)}
        </span>
        <span style={{ color: 'var(--text-3)' }}>
          {dirLabel(s.dir)} · {subjects.length === 1 ? subjects[0] : `${subjects.length} people`} · {better}
        </span>
      </div>
    </Card>
  );
}

function UntrainedAlert({ stA, stB }: { stA: ModelStatus; stB: ModelStatus }) {
  const { s, setModel } = useView();
  const missing = (['A', 'B'] as const).filter((w) => !(w === 'A' ? stA : stB).found.length);
  if (!missing.length) return null;
  return (
    <Alert variant="light" color="gray" radius="lg" icon={<IconAlertTriangle size={20} />} title={`Model ${missing.join(' and ')} not trained for this view`} data-testid="untrained-alert">
      <Stack gap="xs">
        {missing.map((w) => {
          const st = w === 'A' ? stA : stB;
          const p = s[w];
          const canSwitch = String(p.code) === 'original';
          return (
            <Group key={w} gap="sm" wrap="wrap" align="center">
              <Text size="sm">
                <Swatch which={w} /> <b>{w}</b> ({modelTitle(st) === 'Not trained' ? 'this combination' : modelTitle(st)}){' '}
                {canSwitch && s.dir === 'S2toS1'
                  ? 'was only run with the original code for learn 1 → test 2.'
                  : `has no runs for ${s.subj === 'All' ? 'anyone' : s.subj} in ${dirLabel(s.dir)}.`}
              </Text>
              {canSwitch ? (
                <Button size="xs" variant="default" onClick={() => setModel(w, { ...p, code: 'framework' })} data-testid={`alert-framework-${w}`}>
                  Use the re-run in our framework{p.code === 'original' && DS.experiments['X00f_baseline_framework'] ? ' (X00f)' : ''}
                </Button>
              ) : (
                <Text size="sm" c="dimmed">
                  {st.hint ?? 'Pick another preset in the left panel.'}
                </Text>
              )}
            </Group>
          );
        })}
      </Stack>
    </Alert>
  );
}

// ── KPI cards ────────────────────────────────────────────────────────────────
function countText(x: KpiSide): string {
  if (!x.s.n) return 'not trained';
  const runs = x.runsMin === x.runsMax ? `${x.runsMin}` : `${x.runsMin}–${x.runsMax}`;
  const who = x.people.length === 1 ? x.people[0] : `${x.people.length} people`;
  return `${who} × ${x.positions} positions × ${runs} runs`;
}

function KpiCard({
  id,
  title,
  sub,
  g,
  k,
  stA,
  stB,
  metric,
}: {
  id: 'toes' | 'other';
  title: string;
  sub: string;
  g: KpiGroup;
  k: Kpis;
  stA: ModelStatus;
  stB: ModelStatus;
  metric: MetricKey;
}) {
  const md = METRICS[metric];
  const unit = md.unit ? ` ${md.unit}` : '';
  const comparable = Number.isFinite(g.delta);
  const verdict = comparable ? g.verdict : null;
  const deltaClass = verdict === 1 ? 'good' : verdict === -1 ? 'bad' : '';
  const thr = deltaThreshold(metric, g.peak.median);
  const isBias = metric === 'bias';
  const side = (w: 'A' | 'B') => {
    const q = (w === 'A' ? g.a : g.b).s;
    const st = w === 'A' ? stA : stB;
    const codes = modelCodes(st) || 'not trained';
    return (
      <Stack gap={4} style={{ minWidth: 0 }}>
        <Group gap={6} wrap="nowrap" style={{ minWidth: 0 }}>
          <Swatch which={w} />
          <Text size="sm" fw={500} style={{ flex: 'none' }}>
            Model {w}
          </Text>
          <Text size="xs" c="dimmed" truncate="end" title={`${modelTitle(st)} (${codes})`} style={{ minWidth: 0 }}>
            {codes}
          </Text>
        </Group>
        <Text className="kpi-value" data-testid={`kpi-${id}-${w}`}>
          {fmt(q.p50)}
          {q.n > 0 && unit && (
            <Text span size="sm" c="dimmed" fw={500}>
              {unit}
            </Text>
          )}
        </Text>
        <Text size="xs" c="dimmed" className="num" data-testid={`kpi-${id}-${w}-p5p95`}>
          5–95 %: {range(q.p5, q.p95)}
        </Text>
        <Text size="xs" c="dimmed" className="num" data-testid={`kpi-${id}-${w}-p1p99`}>
          1–99 %: {range(q.p1, q.p99)}
        </Text>
      </Stack>
    );
  };
  const cA = countText(g.a);
  const cB = countText(g.b);
  const counts = cA === cB ? `each model: ${cA}` : `A: ${cA} · B: ${cB}`;
  // like-for-like note: which people the comparison uses
  const notes: string[] = [];
  if (k.common.length && (k.onlyA.length || k.onlyB.length)) {
    const parts = [k.onlyA.length ? `B not trained for ${listJoin(k.onlyA)}` : '', k.onlyB.length ? `A not trained for ${listJoin(k.onlyB)}` : ''].filter(Boolean);
    notes.push(`Compared on ${listJoin(k.common)} only (${parts.join('; ')}).`);
  } else if (!k.common.length && (k.onlyA.length || k.onlyB.length)) notes.push('No person in view has both models, so there is no B − A; each model is shown on its own people.');
  return (
    <Card data-testid={`kpi-card-${id}`}>
      <Group justify="space-between" align="flex-start" mb="md" wrap="nowrap" gap="sm">
        <Stack gap={2} style={{ minWidth: 0 }}>
          <Title order={3}>{title}</Title>
          <Text size="sm" c="dimmed">
            {compareLabel(metric)}: median over runs at each position, then over positions · {sub}
          </Text>
        </Stack>
        <Tooltip
          label={comparable ? `Better / worse only when |B − A| is at least ${fmt(thr, 2)}${unit}; smaller differences count as about the same.` : 'Needs both models on at least one person.'}
        >
          <span style={{ flex: 'none', display: 'inline-flex' }}>
            <VerdictBadge
              verdict={verdict}
              neutralLabel={!comparable ? 'no comparison' : 'about the same'}
              betterLabel={isBias ? 'B closer to 0' : 'B better'}
              worseLabel={isBias ? 'B further from 0' : 'B worse'}
            />
          </span>
        </Tooltip>
      </Group>
      <SimpleGrid cols={{ base: 2, xs: 3 }} spacing="md" verticalSpacing="md">
        {side('A')}
        {side('B')}
        <Stack gap={4}>
          <Text size="sm" fw={500}>
            B − A
          </Text>
          <Text className={`kpi-delta num ${deltaClass}`} data-testid={`kpi-${id}-delta`}>
            {signed(g.delta)}
          </Text>
          <Text size="xs" c="dimmed">
            {!comparable ? 'needs both models' : isBias ? 'smaller |bias| is better' : md.higherBetter ? 'higher is better' : 'lower is better'}
          </Text>
        </Stack>
      </SimpleGrid>
      <Text size="xs" c="dimmed" mt="sm" className="num" data-testid={`kpi-${id}-n`}>
        {counts}
      </Text>
      {notes.map((n) => (
        <Group key={n} gap={6} mt="md" wrap="nowrap" align="flex-start" data-testid={`kpi-${id}-scope`}>
          <IconAlertTriangle size={14} style={{ flex: 'none', marginTop: 3, color: 'var(--text-3)' }} />
          <Text size="sm" c="dimmed">
            {n}
          </Text>
        </Group>
      ))}
      <Text size="sm" c="dimmed" mt="md" className="num" data-testid={`kpi-${id}-peak`}>
        Peak voluntary torque at these positions: median {fmt(g.peak.median, 1)} Nm (range {fmt(g.peak.min, 1)} to {fmt(g.peak.max, 1)} Nm)
        {metric === 'mae' || isBias ? ', the scale to read the error against.' : '.'}
      </Text>
      {isBias && (
        <Text size="xs" c="dimmed" mt={6}>
          Size of the bias (|bias|) per run, so positive and negative offsets do not cancel; the box plots below keep the sign.
        </Text>
      )}
    </Card>
  );
}

// ── per-position box plots ───────────────────────────────────────────────────
function PerPositionCard({ stA, stB }: { stA: ModelStatus; stB: ModelStatus }) {
  const { s, openWaveform, tokens } = useView();
  const narrow = !!useMediaQuery('(max-width: 48em)');
  const md = METRICS[s.metric];
  const all = s.subj === 'All';
  const subjects = scopeSubjects(DS, s.subj);
  return (
    <SectionCard
      title={`${md.label} per position`}
      description={`${dirLabel(s.dir, true)} · ${all ? 'each person' : s.subj} · spread over training runs (random seeds); the test trials are the same for every run`}
      testId="per-position"
    >
      <Stack gap="md">
        <div className="legend-row">
          <span>
            <Swatch which="A" /> A: {modelName(stA)}
          </span>
          <span>
            <Swatch which="B" /> B: {modelName(stB)}
          </span>
        </div>
        <Text size="xs" c="dimmed" mt={-8}>
          box = 25–75 % · whiskers = 5–95 % · line in box = median · dots = runs · shaded = toes-down · positions in rad
          {s.metric === 'mae' ? ' · grey number under each position = its peak voluntary torque (Nm)' : ''}
          {s.metric === 'bias' ? ' · line at 0 = no bias' : ''}
        </Text>
        {all ? (
          <SimpleGrid cols={{ base: 1, lg: 2, xl: 3 }} spacing="lg" verticalSpacing="lg">
            {subjects.map((subj) => (
              <SubjectBox key={subj} subj={subj} compact narrow={narrow} A={s.A} B={s.B} dir={s.dir} metric={s.metric} floor={s.floor} tokens={tokens} onPick={openWaveform} />
            ))}
          </SimpleGrid>
        ) : (
          subjects.map((subj) => (
            <SubjectBox key={subj} subj={subj} compact={false} narrow={narrow} A={s.A} B={s.B} dir={s.dir} metric={s.metric} floor={s.floor} tokens={tokens} onPick={openWaveform} />
          ))
        )}
        <Note>Click a box or a dot to open that person and position in the waveform viewer. Hover for median, 5–95 %, 1–99 %, number of runs and peak torque.</Note>
      </Stack>
    </SectionCard>
  );
}

function clipText(low: number | undefined, high: number | undefined, unit: string): string {
  const parts: string[] = [];
  if (low !== undefined) parts.push(`below ${fmt(low, Math.abs(low) < 1 ? 2 : 1)}${unit}`);
  if (high !== undefined) parts.push(`above ${fmt(high, high < 1 ? 2 : 1)}${unit}`);
  return `▼ / ▲ = runs ${parts.join(' or ')}, off the axis; label = that box's true median (or how many runs are off it).`;
}

const SubjectBox = memo(function SubjectBox({
  subj,
  compact,
  narrow,
  A,
  B,
  dir,
  metric,
  floor,
  tokens,
  onPick,
}: {
  subj: string;
  compact: boolean;
  narrow: boolean;
  A: Params;
  B: Params;
  dir: Dir;
  metric: MetricKey;
  floor: boolean;
  tokens: ChartTokens;
  onPick: (subj: string, pos: number) => void;
}) {
  const positions = positionsOf(DS, subj, dir);
  const built = useMemo(() => {
    const a = positionStats(DS, A, subj, dir, metric, floor);
    const b = positionStats(DS, B, subj, dir, metric, floor);
    const peaks = positions.map((p) => peakOf(DS, subj, dir, p));
    const { option, clip } = boxOption({ subj, positions, peaks, A: a, B: b, metric, floor, compact, narrow }, tokens);
    return { option, clip, a, b };
  }, [A, B, subj, dir, metric, floor, compact, narrow, tokens, positions]);
  const missing = [!built.a.exp && 'A', !built.b.exp && 'B'].filter(Boolean) as string[];
  const height = compact ? (metric === 'mae' ? 244 : 228) : metric === 'mae' ? 430 : 410;
  const unit = METRICS[metric].unit ? ` ${METRICS[metric].unit}` : '';
  const clipped = built.clip.low !== undefined || built.clip.high !== undefined;
  const iesNote = subj === 'IES' ? 'no passive trials: nothing subtracted' : '';
  return (
    <Stack gap={6} data-testid={`box-${subj}`}>
      {compact && (
        <Group justify="space-between" gap="xs" wrap="nowrap" align="flex-start">
          <Stack gap={0} style={{ minWidth: 0 }}>
            <Text fw={650}>{subj}</Text>
            <Text size="xs" c="dimmed" style={{ minHeight: 18 }}>
              {iesNote || ' '}
            </Text>
          </Stack>
          {missing.length > 0 && (
            <Badge size="sm" color="gray" variant="light" tt="none" style={{ flex: 'none' }} leftSection={<IconAlertTriangle size={11} />}>
              {missing.join(' & ')} not trained
            </Badge>
          )}
        </Group>
      )}
      {!compact && (iesNote || missing.length > 0) && (
        <Group gap="sm" wrap="wrap">
          {missing.length > 0 && (
            <Badge color="gray" variant="light" tt="none" size="md" leftSection={<IconAlertTriangle size={12} />}>
              Model {missing.join(' & ')} not trained for {subj} in this direction
            </Badge>
          )}
          {iesNote && (
            <Text size="sm" c="dimmed">
              {subj}: {iesNote}
            </Text>
          )}
        </Group>
      )}
      {!positions.length || (missing.length === 2 && !compact) ? (
        <EmptyState title="Not trained for this person/direction" hint="Pick another preset, or switch Code to “our framework” (the original code only ran learn 1 → test 2)." />
      ) : (
        <EChart
          option={built.option}
          height={height}
          label={`${subj}: ${METRICS[metric].label} per position, model A vs model B`}
          onClick={(p) => {
            let k = -1;
            if (p.seriesType === 'boxplot') k = p.dataIndex;
            else if (p.seriesType === 'scatter' && Array.isArray(p.value)) k = Math.round(p.value[0]);
            if (k >= 0 && k < positions.length) onPick(subj, positions[k]);
          }}
        />
      )}
      {clipped && (
        <Text className="hint-line" data-testid={`clip-${subj}`}>
          {clipText(built.clip.low, built.clip.high, unit)}
        </Text>
      )}
    </Stack>
  );
});

// ── heat map ─────────────────────────────────────────────────────────────────
function HeatmapCard() {
  const { s, openWaveform, tokens, wpos } = useView();
  const md = METRICS[s.metric];
  const isBias = s.metric === 'bias';
  const cells = useMemo(() => heatmap(DS, s.A, s.B, s.dir, s.metric, s.floor), [s.A, s.B, s.dir, s.metric, s.floor]);
  const anyFinite = cells.some((c) => Number.isFinite(c.d));
  const span = useMemo(() => {
    const v = cells.filter((c) => Number.isFinite(c.d)).map((c) => Math.abs(c.d));
    const minSpan = md.floorable ? 0.1 : 0.2;
    return Math.max(v.length ? quantile(v, 0.9) : 0, minSpan);
  }, [cells, md.floorable]);
  const colour = (c: HeatCell) => cellColour(c, span, tokens);
  const unit = md.unit ? ` ${md.unit}` : '';
  return (
    <SectionCard
      title="Difference B − A, every person × position"
      description={`median over runs of B minus median of A · ${compareLabel(s.metric, false)}${unit} · ${dirLabel(s.dir)}`}
      testId="heatmap"
      actions={
        anyFinite ? (
          <div className="legend-row">
            <span>
              <Swatch color="var(--c-good)" /> {isBias ? 'B closer to 0' : 'B better'}
            </span>
            <span>
              <span className="swatch" style={{ background: 'var(--surface-2)', boxShadow: 'inset 0 0 0 1px var(--line)' }} /> about the same
            </span>
            <span>
              <Swatch color="var(--c-bad)" /> {isBias ? 'B further from 0' : 'B worse'}
            </span>
          </div>
        ) : undefined
      }
    >
      {anyFinite ? (
        <>
          <div className="heat-wrap">
            <div className="heat-grid" role="grid" aria-label="B minus A per person and position">
              <div />
              {Array.from({ length: 8 }, (_, k) => (
                <div key={k} className={`heat-head ${k < 2 ? 'toes' : ''}`}>
                  {k < 2 ? `toes-down ${k + 1}` : `pos ${k + 1}`}
                </div>
              ))}
              {DS.subjects.map((subj) => (
                <HeatRow
                  key={subj}
                  subj={subj}
                  cells={cells.filter((c) => c.subj === subj)}
                  colour={colour}
                  selected={(c) => s.pos !== null && s.wsubj === c.subj && wpos !== null && near(wpos, c.pos)}
                  onPick={(c) => openWaveform(c.subj, c.pos)}
                  metric={s.metric}
                  aExp={resolve(DS, s.A, subj, s.dir)?.exp}
                  bExp={resolve(DS, s.B, subj, s.dir)?.exp}
                />
              ))}
            </div>
          </div>
          <Group mt="md" gap="lg">
            <Note>
              Click a cell to open the waveforms at that person and position. Positions differ slightly per person, so columns are ordered from most
              toes-down (most negative angle) upwards. Grey = about the same: {md.floorable ? 'R² within ±0.02' : 'within ±0.05 Nm or ±2 % of the peak torque, whichever is larger'}.
              {md.floorable && s.floor ? ' R² below 0 counts as 0.' : ''}
              {isBias ? ' Compared on the size of the bias (|bias|); the tooltip also gives the signed medians.' : ''}
            </Note>
          </Group>
        </>
      ) : (
        <EmptyState
          title="No person has both models in this direction"
          hint="The difference needs both A and B. Use the button above (or the left panel) to pick a model that was trained in this direction."
        />
      )}
    </SectionCard>
  );
}

function cellColour(c: HeatCell, span: number, t: ChartTokens): { bg: string; fg: string } {
  const surface = t.dark ? '#262a33' : '#f3f4f6';
  if (!Number.isFinite(c.d)) return { bg: surface, fg: t.textMuted };
  const hue = c.verdict === 1 ? t.good : c.verdict === -1 ? t.bad : null;
  if (!hue) return { bg: surface, fg: t.text };
  const k = Math.min(1, Math.abs(c.d) / span);
  const bg = mix(t.dark ? '#1d2129' : '#ffffff', hue, 0.14 + 0.62 * Math.sqrt(k));
  return { bg, fg: luminance(bg) > 0.3 ? '#16181c' : '#f4f5f7' };
}

function TipSide({ which, exp, q, metric, unit }: { which: 'A' | 'B'; exp?: string; q: Summary; metric: MetricKey; unit: string }) {
  return (
    <Stack gap={0}>
      <Text size="sm" className="num">
        <Swatch which={which} /> <b>{which}</b> {exp ? shortId(exp) : ''} ·{' '}
        {q.n ? (
          <>
            median {compareLabel(metric, false)} <b>{fmt(q.p50)}</b>
            {unit}
          </>
        ) : (
          'not trained'
        )}
      </Text>
      {q.n > 0 && (
        <Text size="xs" c="dimmed" className="num" pl={16}>
          5–95 %: {range(q.p5, q.p95)} · 1–99 %: {range(q.p1, q.p99)} · {q.n} runs
        </Text>
      )}
    </Stack>
  );
}

function HeatRow({
  subj,
  cells,
  colour,
  selected,
  onPick,
  metric,
  aExp,
  bExp,
}: {
  subj: string;
  cells: HeatCell[];
  colour: (c: HeatCell) => { bg: string; fg: string };
  selected: (c: HeatCell) => boolean;
  onPick: (c: HeatCell) => void;
  metric: MetricKey;
  aExp?: string;
  bExp?: string;
}) {
  const md = METRICS[metric];
  const unit = md.unit ? ` ${md.unit}` : '';
  return (
    <>
      <div className="heat-subj">{subj}</div>
      {Array.from({ length: 8 }, (_, k) => {
        const c = cells.find((x) => x.k === k);
        if (!c) return <div key={k} className="heat-cell empty" />;
        const { bg, fg } = colour(c);
        const ok = Number.isFinite(c.d);
        const label = (
          <Stack gap={4}>
            <Text size="sm" fw={600}>
              {subj} · {posLabel(c.pos, 3)} rad{k < 2 ? ' · toes-down' : ''} · peak {fmt(c.peak, 1)} Nm
            </Text>
            <TipSide which="A" exp={aExp} q={c.sa} metric={metric} unit={unit} />
            <TipSide which="B" exp={bExp} q={c.sb} metric={metric} unit={unit} />
            {metric === 'bias' && (
              <Text size="xs" c="dimmed" className="num">
                signed median bias: A {fmt(c.biasA)} Nm · B {fmt(c.biasB)} Nm
              </Text>
            )}
            <Text size="sm" className="num">
              B − A: <b>{ok ? signed(c.d) + unit : 'not available'}</b>
            </Text>
          </Stack>
        );
        return (
          <Tooltip key={k} label={label} openDelay={120} maw={420}>
            <button
              type="button"
              className={`heat-cell ${selected(c) ? 'selected' : ''} ${ok ? '' : 'empty'}`}
              style={{ background: bg, color: fg }}
              onClick={() => onPick(c)}
              aria-label={`${subj} position ${posLabel(c.pos)}: B minus A ${ok ? signed(c.d) : 'not available'}`}
            >
              <span className="v" data-testid={`heat-${subj}-${k}`}>
                {ok ? signed(c.d) : '–'}
              </span>
              <span className="p">{posLabel(c.pos)} rad</span>
              {(metric === 'mae' || metric === 'bias') && <span className="p">peak {fmt(c.peak, 1)} Nm</span>}
            </button>
          </Tooltip>
        );
      })}
    </>
  );
}


