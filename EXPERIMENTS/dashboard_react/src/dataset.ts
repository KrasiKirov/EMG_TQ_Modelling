import raw from './data/results.json';
import { buildDataset } from './lib/data';
import type { RawData } from './lib/types';
import { TITLE_OVERRIDES } from './lib/experimentStory';

/** the whole data set, parsed once (Vite inlines the JSON as one JSON.parse call) */
export const DS = buildDataset(raw as RawData);
Object.entries(TITLE_OVERRIDES).forEach(([id, title]) => { if (DS.experiments[id]) DS.experiments[id].title = title; });
