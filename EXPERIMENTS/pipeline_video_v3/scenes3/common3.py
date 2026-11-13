"""Helpers shared by the v3 chapters: the stacked recording rows (4 envelopes, angle, torque) and plot placement."""
import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Line, Rectangle, VGroup

from style3 import C_ANGLE, C_TQ, DIM, FAINT, MUSCLE_C, Plot, T, envelope_curve

NAMES = ['GM', 'GL', 'soleus', 'TA']


def place(plot, left, bottom):
    """Move ``plot`` so the corner of its frame (x0, y0) sits at (left, bottom)."""
    x0, y0 = plot.x_range[0], plot.y_range[0]
    plot.shift(np.array([left, bottom, 0]) - plot.p(x0, y0))
    return plot


class Rows(VGroup):
    """Four envelope rows, the angle row and a taller torque row, all on the same time axis [t0, t1].
    ``self.x(t)`` gives the screen x of time t; ``self.top`` / ``self.bottom`` the y span of all rows;
    ``self.tq`` is the torque Plot (use ``self.tq.p(t, Nm)``)."""

    def __init__(self, z, t0, t1, left=-4.9, width=11.8, top=2.9, row_h=0.55, gap=0.13, tq_h=1.9, tq_range=(-35, 2),
                 xticks=None, tick_size=16):
        super().__init__()
        t = z['eg_t']
        m = (t >= t0) & (t <= t1)
        tt, env, ang, tq = t[m], z['eg_env'][m], z['eg_angle'][m], z['eg_tq'][m]
        self.t0, self.t1 = t0, t1
        self.curves, self.static, self.plots = VGroup(), VGroup(), []
        y = top
        for i, name in enumerate(NAMES):
            p = place(Plot((t0, t1), (0, 1), width, row_h, axis_color=FAINT), left, y - row_h)
            c = envelope_curve(p, tt, env[:, i], MUSCLE_C[i], 2.0)
            lab = T(name, 18, MUSCLE_C[i]).next_to(p.p(t0, 0.5), LEFT, buff=0.2)
            self.plots.append(p)
            self.curves.add(c)
            self.static.add(p, lab)
            y -= row_h + gap
        p = place(Plot((t0, t1), (-0.1, 0.05), width, row_h * 0.7, axis_color=FAINT), left, y - row_h * 0.7)
        self.plots.append(p)
        self.curves.add(p.curve(tt, ang, C_ANGLE, 2.0))
        self.static.add(p, T('angle', 18, C_ANGLE).next_to(p.p(t0, -0.025), LEFT, buff=0.2))
        y -= row_h * 0.7 + gap + 0.08
        if xticks is None:
            span = t1 - t0
            step = 10 if span > 40 else (2 if span > 8 else 1)
            xticks = list(np.arange(np.ceil(t0 / step) * step, t1 + 1e-9, step))
        self.tq = place(Plot((t0, t1), tq_range, width, tq_h, xticks=xticks, yticks=[-30, 0], tick_size=tick_size,
                             xfmt=lambda v: f'{v:g} s', yfmt=lambda v: f'{v:g}'), left, y - tq_h)
        self.curves.add(self.tq.curve(tt, tq, C_TQ, 2.4))
        self.static.add(self.tq, T('torque\n(Nm)', 18, C_TQ).next_to(self.tq.p(t0, -15), LEFT, buff=0.55))
        self.top, self.bottom = top, y - tq_h
        self.add(self.static, self.curves)

    def x(self, t):
        return self.tq.p(t, 0)[0]

    def window_box(self, t_end, width_s=0.19, color='#F0E442', opacity=0.18):
        """Highlight of the 20 samples ending at ``t_end`` across every row."""
        x0, x1 = self.x(t_end - width_s), self.x(t_end)
        r = Rectangle(width=max(x1 - x0, 0.04), height=self.top - self.bottom + 0.1, stroke_color=color,
                      stroke_width=2.5, fill_color=color, fill_opacity=opacity)
        return r.move_to([(x0 + x1) / 2, (self.top + self.bottom) / 2, 0])


def tq_at(z, t_query):
    t, y = z['eg_t'], z['eg_tq']
    return float(np.interp(t_query, t, y))


def push_bounds(z, t_from, thr=-10.0):
    """(start, end) in seconds of the first push (torque below ``thr``) that starts after ``t_from``."""
    t, y = z['eg_t'], z['eg_tq']
    below = y < thr
    i = int(np.argmax((t >= t_from) & below))
    j = i + int(np.argmax(~below[i:]))
    return float(t[i]), float(t[j])
