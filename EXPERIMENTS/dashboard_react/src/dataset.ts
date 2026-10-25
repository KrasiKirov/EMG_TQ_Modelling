import raw from './data/results.json';
import { buildDataset } from './lib/data';
import type { RawData } from './lib/types';

/** the whole data set, parsed once (Vite inlines the JSON as one JSON.parse call) */
export const DS = buildDataset(raw as RawData);
