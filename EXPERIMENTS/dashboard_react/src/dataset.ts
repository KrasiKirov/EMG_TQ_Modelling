import raw from './data/results.json';
import { buildDataset } from './lib/data';
import type { RawData } from './lib/types';
import { TITLE_OVERRIDES } from './lib/experimentStory';

/** Experiments that use EMG auto-gain are left out of the dashboard entirely (rows, curves and lists). */
function withoutAutoGain(r: RawData): RawData {
  const drop = new Set(Object.keys(r.experiments).filter((id) => String(r.experiments[id].params.emg) === 'autogain'));
  const col = r.cols.indexOf('experiment');
  return {
    ...r,
    experiments: Object.fromEntries(Object.entries(r.experiments).filter(([id]) => !drop.has(id))),
    rows: r.rows.filter((row) => !drop.has(String(row[col]))),
    learning_curves: (r.learning_curves ?? []).filter((lc) => !drop.has(lc.experiment)),
    bias_variance: undefined, // its "ours" rows are the auto-gain model
  };
}

/** the whole data set, parsed once (Vite inlines the JSON as one JSON.parse call) */
export const DS = buildDataset(withoutAutoGain(raw as RawData));
Object.entries(TITLE_OVERRIDES).forEach(([id, title]) => { if (DS.experiments[id]) DS.experiments[id].title = title; });
