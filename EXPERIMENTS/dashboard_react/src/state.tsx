import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';
import { DS } from './dataset';
import { type TabKey, type ViewState } from './lib/hash';
import { near, positionsOf } from './lib/data';
import { DEFAULT_A_ID, DEFAULT_B_ID } from './lib/params';
import { NEW_MODEL_ID } from './lib/presentation';
import type { Dir, MetricKey, Params } from './lib/types';
import { TOKENS, type ChartTokens } from './theme';

const DEFAULT_STATE: ViewState = {
  tab: 'study',
  experiment: NEW_MODEL_ID,
  compareA: 'X00_baseline',
  compareB: NEW_MODEL_ID,
  subj: 'All',
  dir: 'S1toS2',
  metric: 'r2',
  floor: true,
  A: { ...DS.experiments[DEFAULT_A_ID].params },
  B: { ...DS.experiments[DEFAULT_B_ID].params },
  wsubj: DS.subjects[0] ?? 'YES',
  pos: null,
  runA: 'typical',
  runB: 'typical',
};

interface Ctx {
  s: ViewState;
  set: (patch: Partial<ViewState>) => void;
  setTab: (t: TabKey) => void;
  setSubj: (v: string) => void;
  setDir: (d: Dir) => void;
  setMetric: (m: MetricKey) => void;
  setModel: (which: 'A' | 'B', p: Params) => void;
  swap: () => void;
  openWaveform: (subj: string, pos: number) => void;
  /** true once after openWaveform: the waveform tab then scrolls its chart into view */
  takeWaveScroll: () => boolean;
  /** the position shown in the waveform viewer (resolved against the person's positions) */
  wpos: number | null;
  tokens: ChartTokens;
}

const ViewContext = createContext<Ctx | null>(null);

export function useView(): Ctx {
  const c = useContext(ViewContext);
  if (!c) throw new Error('ViewContext missing');
  return c;
}

export function ViewProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<ViewState>(DEFAULT_STATE);
  const waveScroll = useRef(false);
  const takeWaveScroll = useCallback(() => {
    const v = waveScroll.current;
    waveScroll.current = false;
    return v;
  }, []);

  const set = useCallback((patch: Partial<ViewState>) => setS((prev) => ({ ...prev, ...patch })), []);

  // The URL hash belongs to the Results page (#setting=<id>, StudyTab); this older view state is no longer written to it.

  const wpos = useMemo(() => {
    const ps = positionsOf(DS, s.wsubj, s.dir);
    if (!ps.length) return null;
    if (s.pos !== null) {
      const want = s.pos;
      const m = ps.find((p) => near(p, want));
      if (m !== undefined) return m;
      // positions differ slightly between sessions (e.g. CH -0.326 vs -0.312): keep the nearest one
      return ps.reduce((best, p) => (Math.abs(p - want) < Math.abs(best - want) ? p : best), ps[0]);
    }
    return ps[0];
  }, [s.wsubj, s.dir, s.pos]);

  const value = useMemo<Ctx>(
    () => ({
      s,
      set,
      setTab: (tab) => {
        set({ tab });
        // a new tab starts at its top (the tab bar stays visible, it is sticky)
        requestAnimationFrame(() => window.scrollTo({ top: 0 }));
      },
      setSubj: (v) => setS((prev) => ({ ...prev, subj: v, wsubj: v !== 'All' ? v : prev.wsubj, pos: v !== 'All' && v !== prev.wsubj ? null : prev.pos })),
      setDir: (dir) => set({ dir }),
      setMetric: (metric) => set({ metric }),
      setModel: (which, p) => set(which === 'A' ? { A: { ...p } } : { B: { ...p } }),
      swap: () => setS((prev) => ({ ...prev, A: prev.B, B: prev.A, runA: prev.runB, runB: prev.runA })),
      openWaveform: (subj, pos) => {
        waveScroll.current = true;
        setS((prev) => ({ ...prev, tab: 'waveforms', wsubj: subj, pos, subj: prev.subj === 'All' ? 'All' : subj }));
      },
      takeWaveScroll,
      wpos,
      tokens: TOKENS,
    }),
    [s, set, wpos, takeWaveScroll],
  );

  return <ViewContext.Provider value={value}>{children}</ViewContext.Provider>;
}
