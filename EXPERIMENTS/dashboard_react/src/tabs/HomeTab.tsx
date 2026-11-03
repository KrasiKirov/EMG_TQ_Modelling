import { useState } from 'react';
import { SegmentedControl, Text, Title, UnstyledButton } from '@mantine/core';
import { EChart } from '../components/EChart';
import { CALIBRATION, catalogRuns } from '../lib/catalog';
import { DS } from '../dataset';
import { fmt, pct } from '../lib/format';
import { median } from '../lib/stats';
import { FINAL_ID } from '../lib/presentation';
import { useView } from '../state';

/** median over (person, angle) cells of the median over seeds — the dashboard's summary rule */
function cellScore(id: string, toes: 0 | 1, key: 'mae' | 'r2', floor: boolean): number {
  const cells: number[] = [];
  for (const subj of DS.subjects) {
    const rows = catalogRuns(id, subj, 'S1toS2').filter((r) => r.toes === toes);
    for (const pos of [...new Set(rows.map((r) => r.pos))]) {
      const v = median(rows.filter((r) => r.pos === pos).map((r) => r[key]));
      cells.push(key === 'r2' && floor ? Math.max(v, 0) : v);
    }
  }
  return median(cells);
}

const STEPS = [
  { id: 'X00_baseline', n: '01', title: 'Last semester', what: 'The original LSTM. Every angle weighted by its torque swing, so toes-down barely counts.' },
  { id: 'norm/A0', n: '02', title: 'Balanced positions', what: 'Each angle’s target divided by its swing: every angle counts the same. No auto-gain.' },
  { id: 'norm/N0', n: '03', title: 'Normalization study', what: 'Four ways to scale the EMG. None beat the simplest: ÷ each muscle’s day-1 maximum.' },
  { id: FINAL_ID, n: '04', title: '10 s calibration', what: 'On day 2, 10 s of pushing per angle fits a stretch and a shift. The LSTM is not retrained.' },
];

const RECIPE: [string, string][] = [
  ['Data', '6 people · train day 1 → test day 2 · 8 angles'],
  ['Inputs', 'GM, GL, SOL, TA envelopes + ankle angle, 100 Hz'],
  ['Window', '0.2 s (20 samples)'],
  ['EMG scaling', '÷ day-1 maximum per muscle · no auto-gain'],
  ['Target', 'Total measured torque'],
  ['Positions', 'Balanced: target ÷ each angle’s swing'],
  ['Model', 'LSTM 8 × 2 · dropout 0.3 · Nadam, lr 0.003'],
  ['Calibration', '10 s per angle on day 2 · torque ≈ a × ŷ + b'],
];

export function HomeTab() {
  const { s, set, tokens } = useView();
  const [part, setPart] = useState<'toes' | 'other'>('toes');
  const open = (id: string) => { set({ experiment: id, tab: 'overview', runA: 'typical' }); window.scrollTo({ top: 0 }); };
  const V = CALIBRATION.variants;
  const fin = V.N0cal?.group;
  const people = CALIBRATION.people;
  const label = { N0: 'N0 · no calibration', N0cal: 'N0 + 10 s calibration', A0: 'A0 · active torque', A0cal: 'A0 + 10 s calibration' } as const;
  const axis = { axisLabel: { color: tokens.textMuted }, axisLine: { lineStyle: { color: tokens.axis } } };
  const tooltip = { trigger: 'axis' as const, backgroundColor: tokens.tooltipBg, borderColor: tokens.tooltipBorder, textStyle: { color: tokens.text }, valueFormatter: (v: number) => `${fmt(v)} Nm` };
  const personOption = {
    color: [tokens.textMuted, tokens.a],
    tooltip, legend: { top: 0, textStyle: { color: tokens.text } },
    grid: { left: 52, right: 16, top: 40, bottom: 32 },
    xAxis: { type: 'category', data: people, ...axis },
    yAxis: { type: 'value', name: 'MAE (Nm)', nameTextStyle: { color: tokens.textMuted }, ...axis, splitLine: { lineStyle: { color: tokens.grid } } },
    series: (['N0', 'N0cal'] as const).map((k) => ({ name: label[k], type: 'bar', barGap: '8%', barMaxWidth: 34, data: people.map((p) => V[k]?.person[part][p] ?? null), itemStyle: { borderRadius: [4, 4, 0, 0] } })),
  };
  const ranks = [...new Set(Object.values(V).flatMap((v) => Object.keys(v.angle)))].map(Number).sort((a, b) => a - b);
  const angleOption = {
    color: [tokens.textMuted, tokens.a, tokens.b, tokens.good],
    tooltip, legend: { top: 0, textStyle: { color: tokens.text } },
    grid: { left: 52, right: 16, top: 64, bottom: 44 },
    xAxis: { type: 'category', data: ranks.map((r) => (r === 1 ? '1 · most toes-down' : String(r))), name: 'angle, from toes-down to toes-up', nameLocation: 'middle', nameGap: 28, nameTextStyle: { color: tokens.textMuted }, ...axis },
    yAxis: { type: 'value', name: 'MAE (Nm)', nameTextStyle: { color: tokens.textMuted }, ...axis, splitLine: { lineStyle: { color: tokens.grid } } },
    series: (['N0', 'N0cal', 'A0', 'A0cal'] as const).filter((k) => V[k]).map((k) => ({ name: label[k], type: 'line', smooth: false, symbolSize: 7, lineStyle: { width: k.endsWith('cal') ? 3 : 2, type: k.endsWith('cal') ? 'solid' : 'dashed' }, data: ranks.map((r) => V[k].angle[String(r)] ?? null) })),
  };
  const improved = people.filter((p) => (V.N0cal?.person.toes[p] ?? Infinity) < (V.N0?.person.toes[p] ?? -Infinity)).length;

  return (
    <div className="home">
      <section className="home-hero">
        <div className="home-hero-text">
          <Text className="eyebrow">Capstone · EMG → ankle torque</Text>
          <Title order={1} className="home-title">Predicting ankle torque from four muscles</Title>
          <Text className="home-lead">An LSTM reads 0.2 s of calf and shin EMG plus the ankle angle and predicts the torque. It learns on day 1 and is tested on day 2, with a 10-second calibration per angle.</Text>
          <div className="home-actions">
            <UnstyledButton className="cta" onClick={() => open(FINAL_ID)}>Open the final model →</UnstyledButton>
            <UnstyledButton className="cta ghost" onClick={() => set({ tab: 'comparison' })}>Compare models</UnstyledButton>
          </div>
        </div>
        {fin && <div className="final-card" aria-label="Final model results">
          <div className="final-card-head"><span className="pill">Final model</span><b>N0 + 10 s calibration</b></div>
          {(['toes', 'other'] as const).map((k) => <div key={k} className="final-row" data-toes={k === 'toes' || undefined}>
            <span className="final-label">{k === 'toes' ? 'Toes-down angles' : 'Other angles'}</span>
            <span className="final-big">{fmt(fin[k].mae)}<small> Nm</small></span>
            <span className="final-sub">MAE · peak ≈ {k === 'toes' ? 6 : 18} Nm</span>
            <span className="final-pair"><b>{fmt(fin[k].r2)}</b> R²</span>
            <span className="final-pair"><b>{pct(fin[k].vaf, 1)}</b> VAF</span>
          </div>)}
          <Text size="xs" c="dimmed">6 people × 3 seeds · scored from 30 s to the end of each trial · medians</Text>
        </div>}
      </section>

      <section className="home-section">
        <div className="home-section-head"><Text className="eyebrow">How we got here</Text><Title order={2}>Four steps</Title></div>
        <div className="steps">
          {STEPS.map((st) => <UnstyledButton key={st.id} className="step" data-final={st.id === FINAL_ID || undefined} onClick={() => open(st.id)}>
            <span className="step-n">{st.n}</span>
            <b className="step-title">{st.title}</b>
            <span className="step-what">{st.what}</span>
            <span className="step-num"><b>{fmt(cellScore(st.id, 1, 'mae', s.floor))}</b> Nm toes-down · R² <b>{fmt(cellScore(st.id, 1, 'r2', s.floor))}</b></span>
            <span className="step-open">Open →</span>
          </UnstyledButton>)}
        </div>
        <Text size="xs" c="dimmed" mt="xs">Each card uses its model page’s numbers: its own target, on the whole trial; the calibrated model from 30 s on. Last semester predicted active torque.</Text>
      </section>

      <section className="home-grid">
        <div className="home-panel">
          <div className="panel-head"><div><Text className="eyebrow">Per person</Text><Title order={3}>Calibration helps {improved} of {people.length} people</Title></div>
            <SegmentedControl size="xs" value={part} onChange={(v) => setPart(v as 'toes' | 'other')} data={[{ value: 'toes', label: 'Toes-down' }, { value: 'other', label: 'Other angles' }]} /></div>
          <EChart option={personOption} height={300} label={`MAE per person at ${part === 'toes' ? 'toes-down' : 'other'} angles, N0 with and without calibration`} />
          <Text size="xs" c="dimmed">Both bars scored from 30 s on, so they are judged on the same stretch.</Text>
        </div>
        <div className="home-panel">
          <div className="panel-head"><div><Text className="eyebrow">Per angle</Text><Title order={3}>The error left is no longer at toes-down</Title></div></div>
          <EChart option={angleOption} height={300} label="MAE by ankle angle for N0 and A0, with and without calibration" />
          <Text size="xs" c="dimmed">Median over the 6 people at each angle (JM and HM have 7 angles). Solid = calibrated.</Text>
        </div>
      </section>

      <section className="home-grid">
        <div className="home-panel">
          <Text className="eyebrow">The final model</Text><Title order={3} mb="sm">Every design choice</Title>
          <div className="recipe">{RECIPE.map(([k, v]) => <div key={k}><span>{k}</span><b>{v}</b></div>)}</div>
        </div>
        <div className="home-panel limits">
          <Text className="eyebrow">Read with care</Text><Title order={3} mb="sm">Limits</Title>
          <ul>
            <li>The calibration needs the torque sensor for 10 s at each angle on day 2.</li>
            <li>Its 10 s are the start of the same trial; it was tried after the main study.</li>
            <li>Six people, one trial per angle per day. Nothing was tested in real time.</li>
            <li>R² is unreliable where the torque barely moves (YES’s most toes-down angle). Read MAE next to the peak.</li>
            <li>VAF ignores offsets, so it hardly moves with the calibration.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
