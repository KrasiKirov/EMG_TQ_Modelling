import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useComputedColorScheme, useMantineColorScheme } from '@mantine/core';
import { DS } from './dataset';
import { hashFromState, stateFromHash, themeFromHash, type TabKey, type ViewState } from './lib/hash';
import { near, positionsOf } from './lib/data';
import { DEFAULT_A_ID, DEFAULT_B_ID } from './lib/params';
import { NEW_MODEL_ID } from './lib/presentation';
import type { Dir, MetricKey, Params } from './lib/types';
import { chartTokens, type ChartTokens } from './theme';

const DEFAULT_STATE: ViewState = {
  tab: 'home',
  experiment: NEW_MODEL_ID,
  compareA: 'main/session-passive',
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
  scheme: 'light' | 'dark';
  toggleScheme: () => void;
  tokens: ChartTokens;
}

const ViewContext = createContext<Ctx | null>(null);

export function useView(): Ctx {
  const c = useContext(ViewContext);
  if (!c) throw new Error('ViewContext missing');
  return c;
}

function writeHash(h: string) {
  if (window.location.hash === h) return;
  try {
    window.history.replaceState(null, '', h);
  } catch {
    window.location.replace(h);
  }
}

export function ViewProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<ViewState>(() => stateFromHash(DS, window.location.hash, DEFAULT_STATE));
  const { setColorScheme } = useMantineColorScheme();
  const scheme = useComputedColorScheme('light', { getInitialValueInEffect: false });
  const [themeInHash, setThemeInHash] = useState(() => themeFromHash(window.location.hash) !== null);
  const lastWritten = useRef('');
  const waveScroll = useRef(false);
  const takeWaveScroll = useCallback(() => {
    const v = waveScroll.current;
    waveScroll.current = false;
    return v;
  }, []);

  const set = useCallback((patch: Partial<ViewState>) => setS((prev) => ({ ...prev, ...patch })), []);

  // state -> hash
  useEffect(() => {
    const h = hashFromState(DS, s, themeInHash ? scheme : null);
    lastWritten.current = h;
    writeHash(h);
  }, [s, scheme, themeInHash]);

  // hash -> state (deep links, manual edits, back/forward)
  useEffect(() => {
    const onHash = () => {
      if (window.location.hash === lastWritten.current) return;
      setS((prev) => stateFromHash(DS, window.location.hash, prev));
      const t = themeFromHash(window.location.hash);
      if (t) {
        setThemeInHash(true);
        setColorScheme(t);
      }
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [setColorScheme]);

  // the toggle saves the choice (localStorage); a theme given in the URL stays in the URL and follows the toggle
  const toggleScheme = useCallback(() => setColorScheme(scheme === 'dark' ? 'light' : 'dark'), [scheme, setColorScheme]);

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
      scheme,
      toggleScheme,
      tokens: chartTokens(scheme),
    }),
    [s, set, wpos, scheme, toggleScheme, takeWaveScroll],
  );

  return <ViewContext.Provider value={value}>{children}</ViewContext.Provider>;
}
