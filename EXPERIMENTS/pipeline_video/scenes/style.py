"""Shared look and helpers for every chapter scene.

Scenes only READ data: data/*.npz (arrays), narration/values.py (display strings built from
data/facts_*.json), data/code_excerpts.json (real code lines) and audio/beats.json (durations).
"""
import json
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UL, UP, Axes, Create, DashedVMobject, FadeIn, FadeOut, Line,
                   MarkupText, Rectangle, RoundedRectangle, Scene, Text, VGroup, VMobject, config)
from pygments import lex
from pygments.lexers import PythonLexer, TypeScriptLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, String

VIDEO = Path(__file__).resolve().parent.parent
DATA, AUDIO = VIDEO / 'data', VIDEO / 'audio'
sys.path.insert(0, str(VIDEO / 'narration'))
from values import values                                    # noqa: E402

V = values()
EX = json.loads((DATA / 'code_excerpts.json').read_text(encoding='utf-8'))
BEATS = json.loads((AUDIO / 'beats.json').read_text(encoding='utf-8')) if (AUDIO / 'beats.json').exists() else {}
GAP = json.loads((VIDEO / 'narration' / 'narration.json').read_text(encoding='utf-8'))['gap_s']

# ── palette (STORYBOARD.md §2) ───────────────────────────────────────────────
BG, INK, DIM, FAINT, PANEL = '#0E1116', '#E6EDF3', '#8B949E', '#30363D', '#161B22'
C_GM, C_GL, C_SOL, C_TA = '#2EC4B6', '#A78BFA', '#FFC857', '#F15BB5'
MUSCLE = {'GM': C_GM, 'GL': C_GL, 'SOL': C_SOL, 'TA': C_TA}
MUSCLE_C = [C_GM, C_GL, C_SOL, C_TA]
C_ANGLE, C_S1, C_S2 = '#C9D1D9', '#4C9AFF', '#FF7A45'
C_TQ, C_PASSIVE, C_PRED, C_GOOD, C_BAD, C_HI = '#FFFFFF', '#8B949E', '#9BE564', '#3FB950', '#F85149', '#F0E442'
ANGLE_C = ['#8E6CEF', '#6C8CF5', '#4FA8E8', '#3EC1C9', '#4FD39B', '#8BDD6A', '#C9DE4B', '#F5D547']  # toes down → up
FONT, MONO = 'Segoe UI', 'Consolas'
config.background_color = BG

MAIN_TOP, MAIN_BOTTOM = 3.25, -0.95          # animation area (y), code panel below it


def T(s, size=26, color=INK, weight='NORMAL', font=FONT, **kw):
    return Text(str(s), font=font, font_size=size, color=color, weight=weight, **kw)


def M(s, size=26, color=INK, font=FONT, **kw):
    return MarkupText(str(s), font=font, font_size=size, color=color, **kw)


def chapter_tag(n, title):
    tag = VGroup(T(f'{n}', 22, C_HI, weight='BOLD'), T(title, 22, DIM)).arrange(RIGHT, buff=0.18)
    return tag.to_corner(UL, buff=0.32)


# ── code panel: real lines, syntax colours, file:line header ─────────────────
TOKEN_COLOURS = [(Comment, '#8B949E'), (String, '#A5D6FF'), (Number, '#79C0FF'), (Keyword, '#FF7B72'),
                 (Name.Builtin, '#FFA657'), (Name.Function, '#D2A8FF'), (Name.Class, '#D2A8FF'),
                 (Operator, '#FF7B72')]


def _esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _markup(line, lexer):
    out = []
    for tok, val in lex(line, lexer):
        val = val.rstrip('\n')
        if not val:
            continue
        col = next((c for t, c in TOKEN_COLOURS if tok in t), None)
        out.append(f'<span foreground="{col}">{_esc(val)}</span>' if col else _esc(val))
    return ''.join(out) or ' '


class CodePanel(VGroup):
    """Excerpt ``key`` of data/code_excerpts.json, optionally only lines ``first``..``last`` (file line numbers)."""

    def __init__(self, key, first=None, last=None, width=13.6, max_height=2.55, max_font=17, min_font=10.5):
        super().__init__()
        e = EX[key]
        a, b = first or e['first'], last or e['last']
        lines = e['lines'][a - e['first']:b - e['first'] + 1]
        body = textwrap.dedent('\n'.join(lines)).split('\n')
        lexer = TypeScriptLexer() if e['path'].endswith(('.ts', '.tsx')) else PythonLexer()
        longest = max(len(l) for l in body) + 5
        fs = min(max_font, (width - 0.5) / (0.00763 * longest), max_height / (0.0175 * len(body) + 0.01))
        fs = max(fs, min_font)
        nums = VGroup(*[T(str(a + i), fs * 0.85, '#6E7681', font=MONO) for i in range(len(body))])
        lh = fs * 0.0175
        rows = VGroup(*[M(_markup(l, lexer), fs, INK, font=MONO) if l.strip() else
                        Rectangle(width=0.01, height=lh * 0.5, stroke_opacity=0, fill_opacity=0) for l in body])
        for i, (n, r, line) in enumerate(zip(nums, rows, body)):
            y = -i * lh
            n.move_to([0, y, 0], aligned_edge=RIGHT)
            # spaces have no glyphs: shift each row by its own indentation (Consolas: 0.00763 units per char per pt)
            indent = (len(line) - len(line.lstrip(' '))) * 0.00763 * fs
            r.move_to([0.25 + indent, y, 0], aligned_edge=LEFT)
            # align baselines roughly: MarkupText height varies with glyphs; centre on the row
            r.set_y(y)
            n.set_y(y)
        code = VGroup(nums, rows)
        head = T(f"{e['path']}:{a}" + (f'–{b}' if b > a else ''), 15, DIM, font=MONO)
        h = code.height + 0.62
        w = max(code.width, head.width) + 0.5
        box = RoundedRectangle(corner_radius=0.1, width=min(width, max(w, 4.0)), height=h, fill_color=PANEL,
                               fill_opacity=1, stroke_color=FAINT, stroke_width=1.5)
        head.move_to(box.get_corner(UL) + [0.25 + head.width / 2, -0.22, 0])
        code.move_to(box.get_center() + [0, -0.14, 0]).align_to(box, LEFT).shift(RIGHT * 0.25)
        self.box, self.head, self.rows, self.nums, self.first, self.lh = box, head, rows, nums, a, lh
        self.add(box, head, code)

    def mark(self, first, last=None, color=C_HI):
        """Translucent bar behind file lines first..last (an Animation-ready mobject)."""
        last = last or first
        r0, r1 = self.rows[first - self.first], self.rows[last - self.first]
        top, bot = r0.get_top()[1] + 0.05, r1.get_bottom()[1] - 0.05
        bar = Rectangle(width=self.box.width - 0.2, height=max(top - bot, self.lh), fill_color=color, fill_opacity=0.13,
                        stroke_width=0).move_to([self.box.get_center()[0], (top + bot) / 2, 0])
        return bar


def code_at_bottom(panel):
    return panel.move_to([0, -2.42, 0]).to_edge(DOWN, buff=0.22)


# ── plots without LaTeX: Axes with our own Text ticks ────────────────────────
class Plot(VGroup):
    def __init__(self, x_range, y_range, width, height, xticks=(), yticks=(), xlabel='', ylabel='', tick_size=16,
                 axis_color=FAINT, xfmt=None, yfmt=None, grid=False):
        super().__init__()
        # Axes only maps coordinates (it would draw the x axis at y = 0); the visible frame is ours
        self.ax = Axes(x_range=[x_range[0], x_range[1], (x_range[1] - x_range[0])], y_range=[y_range[0], y_range[1],
                       (y_range[1] - y_range[0])], x_length=width, y_length=height, tips=False,
                       axis_config=dict(include_numbers=False, include_ticks=False, stroke_opacity=0, stroke_width=0))
        x0, x1, y0, y1 = x_range[0], x_range[1], y_range[0], y_range[1]
        self.frame = VGroup(Line(self.ax.c2p(x0, y0), self.ax.c2p(x1, y0), stroke_color=axis_color, stroke_width=1.6),
                            Line(self.ax.c2p(x0, y0), self.ax.c2p(x0, y1), stroke_color=axis_color, stroke_width=1.6))
        self.add(self.ax, self.frame)
        fx = xfmt or (lambda v: f'{v:g}')
        fy = yfmt or (lambda v: f'{v:g}')
        self.ticks = VGroup()
        for v in xticks:
            p = self.ax.c2p(v, y_range[0])
            self.ticks.add(Line(p, p + DOWN * 0.07, stroke_color=axis_color, stroke_width=1.5),
                           T(fx(v).replace('-', '−'), tick_size, DIM).next_to(p, DOWN, buff=0.1))
        for v in yticks:
            p = self.ax.c2p(x_range[0], v)
            self.ticks.add(Line(p, p + LEFT * 0.07, stroke_color=axis_color, stroke_width=1.5),
                           T(fy(v).replace('-', '−'), tick_size, DIM).next_to(p, LEFT, buff=0.1))
            if grid:
                self.ticks.add(DashedVMobject(Line(p, self.ax.c2p(x_range[1], v), stroke_color=FAINT, stroke_width=1),
                                              num_dashes=40))
        self.add(self.ticks)
        if xlabel:
            self.xlabel = T(xlabel, tick_size, DIM).next_to(self.frame[0], DOWN, buff=0.42 if len(xticks) else 0.15)
            self.add(self.xlabel)
        if ylabel:
            self.ylabel = T(ylabel, tick_size, DIM).rotate(np.pi / 2).next_to(self.frame[1], LEFT,
                                                                              buff=0.62 if len(yticks) else 0.15)
            self.add(self.ylabel)

    def p(self, x, y):
        return self.ax.c2p(x, y)

    def curve(self, x, y, color=INK, width=2.0, max_points=1500, clip=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) > max_points:
            idx = np.linspace(0, len(x) - 1, max_points).astype(int)
            x, y = x[idx], y[idx]
        if clip is not None:
            y = np.clip(y, *clip)
        pts = [self.ax.c2p(a, b) for a, b in zip(x, y)]
        return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)


def envelope_curve(plot, x, y, color, width=2.0, max_points=1500):
    """Min/max-preserving decimation for dense signals (keeps spikes visible)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) <= max_points:
        return plot.curve(x, y, color, width, max_points)
    n = max_points // 2
    edges = np.linspace(0, len(x), n + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        i, j = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        for k in sorted((i, j)):
            xs.append(x[k])
            ys.append(y[k])
    return plot.curve(xs, ys, color, width, max_points=10 ** 9)


def load(name):
    return np.load(DATA / f'{name}.npz', allow_pickle=False)


def facts():
    from values import facts as _f
    return _f()


# ── narration-timed beats ────────────────────────────────────────────────────
class BeatScene(Scene):
    def setup(self):
        self.beat_log = []

    @contextmanager
    def beat(self, bid, gap=GAP):
        dur = BEATS[bid]['seconds']
        t0 = self.renderer.time
        self.add_sound(str(AUDIO / f'{bid}.wav'))
        self._beat = (t0, dur)
        yield dur
        rest = t0 + dur + gap - self.renderer.time
        if rest > 0.02:
            self.wait(rest)
        self.beat_log.append((bid, t0, self.renderer.time))

    def left(self):
        t0, dur = self._beat
        return max(0.1, t0 + dur - self.renderer.time)

    def show_code(self, panel, old=None, rt=0.6):
        anims = [FadeIn(panel, shift=UP * 0.1)]
        if old is not None:
            anims.append(FadeOut(old))
        self.play(*anims, run_time=rt)
        return panel

    def mark(self, panel, first, last=None, rt=0.4, color=C_HI):
        """Highlight file lines first..last; the bar belongs to the panel, so it leaves with it."""
        bar = panel.mark(first, last, color)
        self.play(FadeIn(bar), run_time=rt)
        self.remove(bar)
        panel.add(bar)
        return bar

    def unmark(self, panel, bar, rt=0.3):
        self.play(FadeOut(bar), run_time=rt)
        panel.remove(bar)
