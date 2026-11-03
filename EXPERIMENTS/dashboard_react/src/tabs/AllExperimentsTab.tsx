import { useMemo } from 'react';
import { Text, Title, UnstyledButton } from '@mantine/core';
import { DS } from '../dataset';
import { familiesTable, FAMILY_NAMES } from '../lib/aggregate';
import { catalogTitle, EXTRA } from '../lib/catalog';
import { STORIES, STUDY_ORDER } from '../lib/experimentStory';
import { fmt } from '../lib/format';
import { shortId } from '../lib/params';
import { FINAL_ID } from '../lib/presentation';
import { coverage, pageScore } from '../lib/summary';
import { useView } from '../state';

interface Entry { key: string; page?: string; name: string; code: string; what: string; toesMae: number; toesR2: number; otherMae: number; otherR2: number; people: number; seeds: number }
interface Section { title: string; note: string; entries: Entry[] }

const WHAT: Record<string, string> = {
  'norm/N0cal': 'The session-1 LSTM, plus a 10 s stretch-and-shift per angle in session 2. Not retrained.',
  'norm/N0': 'Total torque · EMG ÷ session-1 maximum · balanced positions · 0.2 s · no auto-gain.',
  'norm/A0': 'As above, but predicts active torque: each session’s relaxed-hold passive torque subtracted.',
  'norm/A0cal': 'Active-torque LSTM plus the 10 s calibration.',
  'norm/N1': 'Each muscle ÷ its session-1 99th percentile, so one spike cannot set the scale.',
  'norm/N2': 'Each muscle ÷ its peak in the maximum-effort trials (% of MVC).',
  'norm/N3': 'Each session ÷ its own 99th percentile. Needs all of session 2’s EMG first.',
  'norm/NR0': 'One ridge regression per angle, no LSTM. Reference only.',
  'norm/NR1': 'One ridge regression per angle, no LSTM. Reference only.',
  'norm/NR2': 'One ridge regression per angle, no LSTM. Reference only.',
  'norm/NR3': 'One ridge regression per angle, no LSTM. Reference only.',
  'X00_baseline': 'Last semester’s code unchanged: plain loss, EMG ÷ training maximum.',
  'main/session-passive': '0.5 s window, corrected trial labels, passive map re-measured in session 2; 10 seeds.',
  H_pgain_posw: 'The family study’s reference LSTM, 8 units × 2 layers.',
  F_gru8: 'A GRU cell instead of the LSTM cell, same size.',
  F_lstm32: 'A four times wider LSTM: 32 units × 2 layers.',
  F_mlp: 'A feed-forward network on the flattened window: no memory.',
  F_tcn: 'A temporal convolution network over the window.',
};

export function AllExperimentsTab() {
  const { s, set } = useView();
  const open = (id: string) => { set({ experiment: id, tab: 'overview', runA: 'typical' }); window.scrollTo({ top: 0 }); };
  const sections = useMemo<Section[]>(() => {
    const page = (id: string): Entry => ({
      key: id, page: id, name: catalogTitle(id), code: id === 'main/session-passive' ? 'retest map' : shortId(id.replace(/^(norm|main|family)\//, '')), what: WHAT[id] ?? STORIES[id]?.purpose ?? '',
      toesMae: pageScore(id, 1, 'mae', s.floor, s.dir), toesR2: pageScore(id, 1, 'r2', s.floor, s.dir),
      otherMae: pageScore(id, 0, 'mae', s.floor, s.dir), otherR2: pageScore(id, 0, 'r2', s.floor, s.dir), ...coverage(id, s.dir),
    });
    const families = familiesTable(DS, s.floor).map((r): Entry => ({
      key: r.id, page: EXTRA[`family/${r.id}`] ? `family/${r.id}` : undefined, name: FAMILY_NAMES[r.id] ?? r.name, code: r.id, what: WHAT[r.id] ?? '',
      toesMae: r.toesMae, toesR2: r.toesR2, otherMae: r.otherMae, otherR2: r.otherR2, people: r.persons.length, seeds: Math.round(r.runs / Math.max(r.persons.length, 1)), // runs = distinct (person, seed) pairs
    }));
    return [
      { title: 'Final model', note: 'Calibrated entries are scored from 30 s to the end of each trial; the others on the whole trial.', entries: ['norm/N0cal', 'norm/N0', 'norm/A0', 'norm/A0cal'].map(page) },
      { title: 'Normalization study', note: 'Total torque, balanced positions, 0.2 s, no auto-gain. N1 and N2 were run on YES, JM and EG only. Ridge rows are a non-LSTM reference.', entries: ['norm/N0', 'norm/N1', 'norm/N2', 'norm/N3', 'norm/NR0', 'norm/NR1', 'norm/NR2', 'norm/NR3'].filter((id) => EXTRA[id]).map(page) },
      { title: 'Network types', note: 'Report 1. Each run is scored on its own target, so compare these rows with each other only. Only the GRU has a page with waveforms.', entries: families },
      { title: 'Earlier models', note: 'Each scored on its own target, on the whole trial.', entries: ['X00_baseline', 'main/session-passive'].map(page) },
      { title: 'Controlled single changes', note: 'Last semester’s LSTM with one setting changed at a time.', entries: STUDY_ORDER.filter((id) => DS.experiments[id] && id !== 'X00_baseline').map(page) },
    ];
  }, [s.floor, s.dir]);

  return (
    <div className="catalog">
      <header className="catalog-head">
        <Text className="eyebrow">Every experiment</Text>
        <Title order={2} className="model-title">All experiments</Title>
        <Text className="model-desc">{sections.reduce((n, x) => n + x.entries.length, 0)} entries in {sections.length} groups. Numbers: median over seeds, then over angles; MAE in Nm, R² below 0 shown as 0. Train session {s.dir === 'S1toS2' ? '1 → test session 2' : '2 → test session 1'}. Click a row to open its page.</Text>
      </header>
      {sections.map((sec) => <section key={sec.title} className="catalog-section">
        <div className="catalog-section-head"><Title order={3}>{sec.title}</Title><Text size="sm" c="dimmed">{sec.note}</Text></div>
        <div className="catalog-table" role="table" aria-label={sec.title}>
          <div className="catalog-row catalog-header" role="row">
            <span role="columnheader">Experiment</span><span role="columnheader" className="num">Toes-down MAE</span><span role="columnheader" className="num">R²</span><span role="columnheader" className="num">Other MAE</span><span role="columnheader" className="num">R²</span><span role="columnheader" className="num">People × seeds</span>
          </div>
          {sec.entries.map((x) => <UnstyledButton key={x.key} component={x.page ? 'button' : 'div'} className="catalog-row" role="row" data-final={x.page === FINAL_ID || undefined} data-link={x.page ? true : undefined} onClick={() => x.page && open(x.page)} disabled={!x.page}>
            <span role="cell" className="catalog-name"><b>{x.name}</b><span className="catalog-code">{x.code}</span><span className="catalog-what">{x.what}</span></span>
            <span role="cell" className="num big">{fmt(x.toesMae)}</span><span role="cell" className="num">{fmt(x.toesR2)}</span>
            <span role="cell" className="num big">{fmt(x.otherMae)}</span><span role="cell" className="num">{fmt(x.otherR2)}</span>
            <span role="cell" className="num muted">{x.people ? `${x.people} × ${x.seeds}` : '–'}{x.page && <span className="catalog-open">Open →</span>}</span>
          </UnstyledButton>)}
        </div>
      </section>)}
    </div>
  );
}
