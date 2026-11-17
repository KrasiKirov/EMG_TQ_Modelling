// How each trial is cut into training, validation and test windows, and how often the network updates and predicts.
// All times come from the export (EXPERIMENTS/analysis/pipeline_stages.py: split_structure, window_settings).
import { SimpleGrid, Text } from '@mantine/core';
import { DataTable, Note, SectionCard, type Column } from '../components/common';
import { PIPE, type PipelinePerson, type WindowSetting } from '../lib/pipeline';
import type { ChartTokens } from '../theme';
import { alpha } from '../theme';

const n0 = (v: number) => v.toLocaleString('en-US');
const s2 = (v: number) => v.toFixed(2);

/** whole-recording timeline of one session-1 trial and its session-2 twin */
function Timeline({ p, t }: { p: PipelinePerson; t: ChartTokens }) {
  const sp = p.splits;
  const s1 = sp.trials.find((r) => r.session === 'S1' && Math.abs(r.angle - p.angle) < 1e-3) ?? sp.trials[0];
  const s2t = sp.trials.find((r) => r.session !== 'S1' && Math.abs(r.angle - p.angle) < 0.02) ?? sp.trials[sp.trials.length - 1];
  const tr = s1.train!, va = s1.val!, te = s2t.test!;
  const recEnd = s1.end + 1 / 100 + PIPE.trim_end_s;
  const W = 900, L = 96, R = 16, x = (v: number) => L + (v / recEnd) * (W - L - R);
  const bar = (y: number, a: number, b: number, c: string, h = 22) => <rect x={x(a)} y={y} width={Math.max(1.5, x(b) - x(a))} height={h} fill={c} rx={2} />;
  const label = (y: number, s: string) => <text x={L - 8} y={y + 15} textAnchor="end" fontSize={12} fill={t.text}>{s}</text>;
  const ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90].filter((v) => v <= recEnd);
  const trim = alpha(t.textMuted, 0.35);
  return (
    <svg viewBox={`0 0 ${W} 150`} width="100%" role="img" aria-label="Timeline of one trial: trimmed ends, training, gap, validation, and the session-2 test">
      {ticks.map((v) => <g key={v}><line x1={x(v)} x2={x(v)} y1={18} y2={112} stroke={t.grid} /><text x={x(v)} y={128} textAnchor="middle" fontSize={11} fill={t.textMuted}>{v} s</text></g>)}
      {label(24, 'Session 1')}
      {bar(24, 0, s1.start, trim)}{bar(24, s1.end + 0.01, recEnd, trim)}
      {bar(24, tr.first_label, tr.last_label, t.fit)}
      {bar(24, tr.last_label + 0.01, va.first_label, t.bad)}
      {bar(24, va.first_label, va.last_label, t.val)}
      <text x={x((tr.first_label + tr.last_label) / 2)} y={39} textAnchor="middle" fontSize={12} fill="#fff" fontWeight={600}>training · {n0(tr.n)} windows</text>
      <text x={x((va.first_label + va.last_label) / 2)} y={39} textAnchor="middle" fontSize={12} fill="#fff" fontWeight={600}>validation · {n0(va.n)}</text>
      <text x={x(tr.last_label)} y={62} textAnchor="middle" fontSize={11} fill={t.bad}>▲ {Math.round((va.first_label - tr.last_label - 0.01) * 1000)} ms gap</text>
      {label(76, 'Session 2')}
      {bar(76, 0, s2t.start, trim)}{bar(76, s2t.end + 0.01, recEnd, trim)}
      {bar(76, te.first_label, te.last_label, t.test)}
      <text x={x((te.first_label + te.last_label) / 2)} y={91} textAnchor="middle" fontSize={12} fill="#fff" fontWeight={600}>test · one prediction every 10 ms · {n0(te.n)}</text>
      <rect x={L} y={138} width={10} height={10} fill={trim} /><text x={L + 14} y={147} fontSize={11} fill={t.textMuted}>trimmed (1 s each end)</text>
      <text x={W - R} y={147} textAnchor="end" fontSize={11} fill={t.textMuted}>bars span label times; each window's inputs start {Math.round(sp.span_samples * 10)} ms before its label</text>
    </svg>
  );
}

/** zoom on the training / validation boundary, one tick per 10 ms sample */
function Boundary({ p, t }: { p: PipelinePerson; t: ChartTokens }) {
  const sp = p.splits;
  const s1 = sp.trials.find((r) => r.session === 'S1' && Math.abs(r.angle - p.angle) < 1e-3) ?? sp.trials[0];
  const tr = s1.train!, va = s1.val!;
  const span = sp.span_samples / 100, dt = 0.01;
  const a = tr.last_label - span - 3 * dt, b = va.first_label + 8 * dt;
  const W = 900, L = 150, R = 16, x = (v: number) => L + ((v - a) / (b - a)) * (W - L - R);
  const win = (y: number, end: number, c: string, name: string, opacity = 1) => (
    <g opacity={opacity}>
      <text x={L - 8} y={y + 12} textAnchor="end" fontSize={12} fill={t.text}>{name}</text>
      <rect x={x(end - span) - 2} y={y} width={x(end) - x(end - span) + 4} height={16} fill={alpha(c, 0.25)} stroke={c} rx={3} />
      {Array.from({ length: sp.window }, (_, i) => <circle key={i} cx={x(end - span + i * sp.step * dt)} cy={y + 8} r={2.4} fill={c} />)}
      <path d={`M${x(end)},${y - 3} l0,22`} stroke={c} strokeWidth={2} />
      <text x={x(end) + 5} y={y + 12} fontSize={11} fill={c}>label {s2(end)} s</text>
    </g>
  );
  const ticks: number[] = [];
  for (let v = Math.ceil(a * 10) / 10; v <= b + 1e-9; v += 0.05) ticks.push(+v.toFixed(2));
  return (
    <svg viewBox={`0 0 ${W} 170`} width="100%" role="img" aria-label="The last training windows and the first validation window share no input sample">
      {ticks.map((v) => <g key={v}><line x1={x(v)} x2={x(v)} y1={8} y2={140} stroke={t.grid} /><text x={x(v)} y={156} textAnchor="middle" fontSize={11} fill={t.textMuted}>{s2(v)} s</text></g>)}
      {win(12, tr.last_label - 2 * dt, t.fit, 'training (earlier)', 0.45)}
      {win(38, tr.last_label - dt, t.fit, 'training (earlier)', 0.7)}
      {win(64, tr.last_label, t.fit, 'last training')}
      <rect x={x(tr.last_label + dt) - 3} y={92} width={x(va.first_label - dt) - x(tr.last_label + dt) + 6} height={8} fill={alpha(t.bad, 0.3)} rx={2} />
      <text x={L - 8} y={100} textAnchor="end" fontSize={11} fill={t.bad}>labels never used</text>
      {win(112, va.first_label, t.val, 'first validation')}
    </svg>
  );
}

const COLS: Column<WindowSetting>[] = [
  { key: 'id', label: 'Setting', render: (r) => <span title={r.title}>{r.id.replace(/_/g, ' ')}</span> },
  { key: 'win', label: 'Window', render: (r) => r.state === 'carry' ? <>whole trial, state carried</> : <>{r.steps} × {r.step_ms} ms</> },
  { key: 'span', label: 'Covers', align: 'right', render: (r) => r.state === 'carry' ? '—' : `${s2(r.span_s + r.step_ms / 1000)} s` },
  { key: 'stride', label: 'New window every', align: 'right', render: (r) => r.state === 'carry' ? '—' : `${r.stride_ms} ms` },
  { key: 'ov', label: 'Overlap', align: 'right', render: (r) => r.state === 'carry' ? '—' : `${r.overlap_pct} %` },
  { key: 'n', label: 'Training windows', align: 'right', render: (r) => r.state === 'carry' ? '—' : r.passes_per_epoch > 1 ? <span title={`each window is used ${r.passes_per_epoch} times per epoch, so an epoch has as many updates as the reference`}>{n0(r.train_windows)} × {r.passes_per_epoch}</span> : n0(r.train_windows), sort: (r) => r.train_windows },
  { key: 'gap', label: 'Gap before validation', align: 'right', render: (r) => `${s2(r.val_gap_s)} s` },
  { key: 'upd', label: 'Updates per epoch', align: 'right', render: (r) => r.state === 'carry' ? <span title={`8 trials side by side, ${r.chunk_s} s chunks`}>{n0(r.updates_per_epoch)} ({r.chunk_s} s chunks)</span> : <>{n0(r.updates_per_epoch)}{r.shuffle ? ' · shuffled' : ''}</>, sort: (r) => r.updates_per_epoch },
  { key: 'ep', label: 'Epochs (median)', align: 'right', render: (r) => r.epochs_median == null ? 'running' : `${r.epochs_median} / ${r.epoch_limit}`, sort: (r) => r.epochs_median ?? -1 },
  { key: 'tot', label: 'Updates in all', align: 'right', render: (r) => r.epochs_median == null ? '—' : `≈ ${n0(Math.round(r.updates_per_epoch * r.epochs_median / 100) * 100)}`, sort: (r) => r.updates_per_epoch * (r.epochs_median ?? 0) },
];

export function WindowStructure({ p, subj, t }: { p: PipelinePerson; subj: string; t: ChartTokens }) {
  const sp = p.splits;
  const s1 = sp.trials.find((r) => r.session === 'S1') ?? sp.trials[0];
  const tr = s1.train!, va = s1.val!;
  const ref = PIPE.settings.find((s) => s.id === PIPE.setting) ?? PIPE.settings[0];
  const nS1 = sp.trials.filter((r) => r.session === 'S1').length;
  const nS2 = sp.trials.length - nS1;
  const kpis: [string, string, string][] = [
    ['Window', `${sp.window} × 10 ms`, `${Math.round(sp.span_samples * 10)} ms from first to last sample`],
    ['New training window', 'every 10 ms', `${n0(tr.n)} per trial, ${n0(sp.n_train)} in all (${nS1} trials)`],
    ['Weight update', `every ${sp.batch} windows`, `${n0(sp.updates_per_epoch)} per epoch; validation once per epoch`],
    ['Test prediction', 'every 10 ms', `${n0(sp.n_test)} over ${nS2} session-2 trials`],
  ];
  return (
    <>
      <SectionCard testId="pipeline-splits" title="How each trial is cut into windows"
        description={`${subj}, session 1: first 80 % of windows train, the rest validate. Session 2 is test only.`}>
        <Timeline p={p} t={t} />
        <SimpleGrid cols={4} spacing="sm" mt="md">
          {kpis.map(([k, v, sub]) => (
            <div key={k} className="study-kpi">
              <Text size="xs" c="dimmed">{k}</Text>
              <Text fz={20} fw={600}>{v}</Text>
              <Text size="xs" c="dimmed">{sub}</Text>
            </div>
          ))}
        </SimpleGrid>
        <Text size="sm" fw={600} mt="lg" mb={4}>The training / validation boundary, one dot per 10 ms sample</Text>
        <Boundary p={p} t={t} />
        <SimpleGrid cols={2} spacing="lg" mt="sm">
          <Text size="sm" c="dimmed">Training labels {s2(tr.first_label)}–{s2(tr.last_label)} s · validation windows start at {s2(va.first_input)} s, so no sample is shared.</Text>
          <Text size="sm" c="dimmed">Validation labels {s2(va.first_label)}–{s2(va.last_label)} s · session 2 predicted every 10 ms from {s2(s1.start + sp.span_samples / 100)} s.</Text>
        </SimpleGrid>
      </SectionCard>

      <SectionCard testId="pipeline-updates" title="When the weights change"
        description={`${n0(sp.n_train)} windows in time order; one update per ${sp.batch} (${s2(sp.batch / 100)} s of a trial).`}>
        <SimpleGrid cols={3} spacing="sm">
          {[
            ['One update', `${sp.batch} windows → loss → gradient → Nadam step. ${n0(sp.updates_per_epoch)} updates make one epoch (one pass over all training windows).`],
            ['After each epoch', `The ${n0(sp.n_val)} validation windows are predicted once; the weights do not change. If the validation loss has not improved for ${6} epochs the learning rate halves.`],
            ['Stopping', `After 15 epochs without improvement, or at ${ref.epoch_limit} epochs. The best epoch's weights are kept. The reference ran a median of ${ref.epochs_median} epochs, about ${n0(Math.round(ref.updates_per_epoch * (ref.epochs_median ?? 0) / 1000) * 1000)} updates.`],
          ].map(([k, v]) => <div key={k} className="pipe-step"><Text size="sm" fw={600}>{k}</Text><Text size="xs" c="dimmed">{v}</Text></div>)}
        </SimpleGrid>
      </SectionCard>

      <SectionCard testId="pipeline-settings" title="Window layout of every setting in the window study"
        description="Same 8 session-1 trials per person; session 2 predicted every 10 ms.">
        <div className="nowrap-cells"><DataTable columns={COLS} rows={PIPE.settings} rowKey={(r) => r.id} dense minWidth={980} stickyFirst caption="Window layout per setting" /></div>
        <Note>WC rows carry the LSTM state across chunks instead of using windows. Epochs: median over finished runs.</Note>
      </SectionCard>
    </>
  );
}
