import { createTheme, rem, type MantineColorSchemeManager, type MantineColorScheme, type MantineColorsTuple } from '@mantine/core';
import { themeFromHash } from './lib/hash';

export const FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";

const slate: MantineColorsTuple = ['#f2f4f6', '#e4e7eb', '#c8ced6', '#aab3be', '#8f9aa7', '#7d8998', '#727e8e', '#5f6a7a', '#525d6d', '#3b4452'];
// cool neutral dark palette: [0] text … [7] body/cards … [9] deepest
const dark: MantineColorsTuple = ['#e4e6ea', '#c3c7ce', '#9aa0ab', '#737a87', '#3a404c', '#2e333d', '#262a33', '#1d2129', '#16191f', '#101216'];

export const theme = createTheme({
  fontFamily: FONT,
  fontFamilyMonospace: "ui-monospace, 'Cascadia Mono', 'SF Mono', Menlo, Consolas, monospace",
  primaryColor: 'slate',
  // dark: 8 is Mantine's default; it drives the light-variant badge colours (green / red stay green / red in dark mode)
  primaryShade: { light: 9, dark: 8 },
  autoContrast: true,
  colors: { slate, dark },
  defaultRadius: 'sm',
  fontSizes: { xs: rem(12), sm: rem(13), md: rem(14), lg: rem(16), xl: rem(20) },
  lineHeights: { xs: '1.4', sm: '1.6', md: '1.5', lg: '1.5', xl: '1.4' },
  headings: {
    fontFamily: FONT,
    fontWeight: '650',
    sizes: {
      h1: { fontSize: rem(22), lineHeight: '1.25' },
      h2: { fontSize: rem(18), lineHeight: '1.3' },
      h3: { fontSize: rem(16), lineHeight: '1.35' },
      h4: { fontSize: rem(14), lineHeight: '1.4' },
    },
  },
  spacing: { xs: rem(8), sm: rem(12), md: rem(16), lg: rem(24), xl: rem(32) },
  components: {
    Card: { defaultProps: { withBorder: true, radius: 'md', padding: 'lg', shadow: 'none' } },
    Badge: { defaultProps: { radius: 'sm', tt: 'none' } },
    // same look as the chart tooltips: a light card with a border (styles.css .app-tooltip)
    Tooltip: { defaultProps: { withArrow: false, multiline: true, maw: 360, transitionProps: { duration: 80 } }, classNames: { tooltip: 'app-tooltip' } },
    Select: { defaultProps: { allowDeselect: false, checkIconPosition: 'right', comboboxProps: { shadow: 'md' } } },
  },
});

export const STORAGE_KEY = 'emg-dash-theme';

/** colour scheme: URL hash (theme=dark|light) overrides, else localStorage (try/catch), else system */
export function schemeManager(): MantineColorSchemeManager {
  let handler: ((e: StorageEvent) => void) | null = null;
  const ok = (v: unknown): v is MantineColorScheme => v === 'light' || v === 'dark' || v === 'auto';
  return {
    get: (defaultValue) => {
      const h = typeof window !== 'undefined' ? themeFromHash(window.location.hash) : null;
      if (h) return h;
      try {
        const v = window.localStorage.getItem(STORAGE_KEY);
        if (ok(v)) return v;
      } catch {
        /* storage blocked */
      }
      return defaultValue;
    },
    set: (value) => {
      try {
        window.localStorage.setItem(STORAGE_KEY, value);
      } catch {
        /* storage blocked */
      }
    },
    subscribe: (onUpdate) => {
      handler = (e) => {
        if (e.key === STORAGE_KEY && ok(e.newValue)) onUpdate(e.newValue);
      };
      window.addEventListener('storage', handler);
    },
    unsubscribe: () => {
      if (handler) window.removeEventListener('storage', handler);
      handler = null;
    },
    clear: () => {
      try {
        window.localStorage.removeItem(STORAGE_KEY);
      } catch {
        /* storage blocked */
      }
    },
  };
}

/** colours used inside canvas charts (CSS variables do not reach the canvas) */
export interface ChartTokens {
  dark: boolean;
  text: string;
  textMuted: string;
  axis: string;
  grid: string;
  surface: string;
  tooltipBg: string;
  tooltipBorder: string;
  a: string;
  b: string;
  ink: string;
  good: string;
  bad: string;
  shade: string;
  fit: string;
  val: string;
  test: string;
  offset: string;
  otherSys: string;
  runToRun: string;
}

export function chartTokens(scheme: 'light' | 'dark'): ChartTokens {
  return scheme === 'dark'
    ? {
        dark: true,
        text: '#e4e6ea',
        textMuted: '#9aa0ab',
        axis: '#4a515e',
        grid: '#2c313b',
        surface: '#1d2129',
        tooltipBg: '#262a33',
        tooltipBorder: '#3a404c',
        a: '#6abfb1',
        b: '#d9a35d',
        ink: '#eceae4',
        good: '#4cc38a',
        bad: '#ef6b5b',
        shade: 'rgba(200, 206, 220, 0.07)',
        fit: '#c98500',
        val: '#199e70',
        test: '#9085e9',
        offset: '#c3c7ce',
        otherSys: '#6b7280',
        runToRun: '#e07ab8',
      }
    : {
        dark: false,
        text: '#25282d',
        textMuted: '#6b7079',
        axis: '#c9cdd3',
        grid: '#eceef1',
        surface: '#ffffff',
        tooltipBg: '#ffffff',
        tooltipBorder: '#dfe2e6',
        a: '#248378',
        b: '#c1833c',
        ink: '#1b1b1b',
        good: '#1f9d55',
        bad: '#c0392b',
        shade: 'rgba(95, 106, 122, 0.075)',
        fit: '#eda100',
        val: '#1baf7a',
        test: '#6250d6',
        offset: '#3b4452',
        otherSys: '#a3abb8',
        runToRun: '#c2408f',
      };
}

/** hex '#rrggbb' + alpha -> rgba() */
export function alpha(hex: string, a: number): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

/** mix two '#rrggbb' colours: t = 0 -> c1, 1 -> c2 */
export function mix(c1: string, c2: string, t: number): string {
  const p = (c: string) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const a = p(c1);
  const b = p(c2);
  const m = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return '#' + m.map((v) => v.toString(16).padStart(2, '0')).join('');
}

export function luminance(hex: string): number {
  const ch = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
}
