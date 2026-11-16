import { createTheme, rem, type MantineColorsTuple } from '@mantine/core';

export const FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";

// one neutral blue for links, selected controls and the active page
const blue: MantineColorsTuple = ['#eef4fb', '#dbe7f5', '#b5cdea', '#8cb1de', '#6996d3', '#4f83cb', '#3f77c6', '#2f64ae', '#25589c', '#1a4b8a'];

export const theme = createTheme({
  fontFamily: FONT,
  fontFamilyMonospace: "ui-monospace, 'Cascadia Mono', 'SF Mono', Menlo, Consolas, monospace",
  primaryColor: 'blue',
  primaryShade: 7,
  autoContrast: true,
  colors: { blue },
  white: '#ffffff',
  black: '#1f2328',
  defaultRadius: 'sm',
  fontSizes: { xs: rem(12), sm: rem(13), md: rem(14), lg: rem(16), xl: rem(20) },
  lineHeights: { xs: '1.4', sm: '1.6', md: '1.5', lg: '1.5', xl: '1.4' },
  headings: {
    fontFamily: FONT,
    fontWeight: '600',
    sizes: {
      h1: { fontSize: rem(22), lineHeight: '1.25' },
      h2: { fontSize: rem(18), lineHeight: '1.3' },
      h3: { fontSize: rem(16), lineHeight: '1.35' },
      h4: { fontSize: rem(14), lineHeight: '1.4' },
    },
  },
  spacing: { xs: rem(8), sm: rem(12), md: rem(16), lg: rem(24), xl: rem(32) },
  components: {
    Card: { defaultProps: { withBorder: true, radius: 'sm', padding: 'lg', shadow: 'none' } },
    Badge: { defaultProps: { radius: 'sm', tt: 'none' } },
    // same look as the chart tooltips: a light card with a border (styles.css .app-tooltip)
    Tooltip: { defaultProps: { withArrow: false, multiline: true, maw: 360, transitionProps: { duration: 80 } }, classNames: { tooltip: 'app-tooltip' } },
    Select: { defaultProps: { allowDeselect: false, checkIconPosition: 'right', comboboxProps: { shadow: 'md' } } },
  },
});

/** colours used inside canvas charts (CSS variables do not reach the canvas) */
export interface ChartTokens {
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

/** the chart colours: white page, grey axes, blue = model A, orange = model B */
export const TOKENS: ChartTokens = {
  text: '#1f2328',
  textMuted: '#5f6670',
  axis: '#c4c8ce',
  grid: '#eceef1',
  surface: '#ffffff',
  tooltipBg: '#ffffff',
  tooltipBorder: '#d5d9de',
  a: '#2f64ae',
  b: '#d0682a',
  ink: '#1f2328',
  good: '#1e7f45',
  bad: '#b8322a',
  shade: 'rgba(95, 102, 112, 0.07)',
  fit: '#d08a00',
  val: '#1e8f66',
  test: '#5a4bc8',
  offset: '#3b4452',
  otherSys: '#a3abb8',
  runToRun: '#b43c86',
};

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
