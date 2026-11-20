// Session-1 validation loss of the hyperparameter tuning (written by EXPERIMENTS/analysis/tuning_val.py, which
// applies select_hparams.py's rule). Session 2 was never used to choose.
import raw from '../data/tuning-val.json';

export interface ValNetwork {
  n_params: number; mean_log_ratio: number; se: number; better_people: number; n_people: number;
  /** person -> median validation loss over seeds (Nm²) */
  median: Record<string, number>;
  /** person -> seed -> best validation loss (Nm²) */
  seeds: Record<string, Record<string, number>>;
}
interface ValData {
  units: string; baseline: string; best: string; chosen: string; networks: Record<string, ValNetwork>; seeds_per_person: number[];
  screen: { configs: number; seeds_per_person: number[]; best: string; chosen: string; finalists: string[] };
}
export const TUNING_VAL = raw as unknown as ValData;
/** the change of validation loss against last semester's network, mean over people of the log-ratio, as a fraction */
export const changeVsBaseline = (n: ValNetwork) => Math.exp(n.mean_log_ratio) - 1;
