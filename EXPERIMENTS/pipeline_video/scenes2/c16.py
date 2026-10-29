"""Chapter 16: potential techniques, auto-gain (beats c16_b01 .. c16_b09)."""
from manim import *

from style2 import *

F3 = facts()['x03']
X3 = load('x03')
Y6 = load('y06')
GREY_B = '#6E7681'          # "before" bars
LO_C, TOP_C = '#B0B8C1', C_TQ   # floor line (light grey) and top line (white)


# ── local helpers ────────────────────────────────────────────────────────────
def checkmark(color=C_GOOD, s=0.3):
    pts = [np.array([-0.5, 0.0, 0]) * s, np.array([-0.15, -0.42, 0]) * s, np.array([0.55, 0.5, 0]) * s]
    return VMobject(stroke_color=color, stroke_width=7).set_points_as_corners(pts)


def check_row(text, size=26, width_chars=50, mark=None):
    mark = mark or checkmark()
    txt = W(text, size, INK, width=width_chars)
    row = VGroup(mark, txt).arrange(RIGHT, buff=0.3, aligned_edge=UP)
    mark.align_to(txt, UP).shift(DOWN * 0.08)
    return row


def rr(w, h, color=FAINT, fill=PANEL, r=0.14):
    return RoundedRectangle(corner_radius=r, width=w, height=h, fill_color=fill, fill_opacity=1,
                            stroke_color=color, stroke_width=2)


def chip(text, w, h, color=FAINT, size=24, tcolor=INK, fill=PANEL):
    box = rr(w, h, color, fill)
    return VGroup(box, T(text, size, tcolor).move_to(box))


def place_plot(P, cx, cy):
    """Centre the plotting area (not the labels) of a Plot at (cx, cy)."""
    P.shift(np.array([cx, cy, 0]) - P.ax.get_center())
    return P


def gbars(pairs, labels, cb, ca, width, height, vmax, nd=2, label_size=22, vsize=20, bw=None):
    """Before/after bars per group. g.before / g.after are VGroups of Rectangles; baseline at y = 0."""
    n = len(pairs)
    slot = width / n
    bw = bw or min(0.8, slot * 0.36)
    g = VGroup()
    before, after, bv, av, labs = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
    for i, ((b, a), lab) in enumerate(zip(pairs, labels)):
        xc = -width / 2 + slot * (i + 0.5)
        for v, grp, vals, xo, col in ((b, before, bv, -bw * 0.55, cb[i]), (a, after, av, bw * 0.55, ca[i])):
            h = max(0.03, v / vmax * height)
            r = Rectangle(width=bw, height=h, fill_color=col, fill_opacity=0.92, stroke_width=0)
            r.move_to([xc + xo, h / 2, 0])
            grp.add(r)
            vals.add(T(fmt(v, nd), vsize, INK).next_to(r, UP, buff=0.08))
        labs.add(T(lab, label_size, DIM).move_to([xc, -0.34, 0]))
    axis = Line([-width / 2 - 0.1, 0, 0], [width / 2 + 0.1, 0, 0], stroke_color=FAINT, stroke_width=1.6)
    g.add(axis, before, after, bv, av, labs)
    g.axis, g.before, g.after, g.bv, g.av, g.labs = axis, before, after, bv, av, labs
    return g


def put(g, x, y):
    """Move group so its baseline (g.axis) centre sits at (x, y)."""
    g.shift(np.array([x, y, 0]) - g.axis.get_center())
    return g


def legend(items, size=20, buff=0.5):
    row = VGroup()
    for col, txt in items:
        row.add(VGroup(Rectangle(width=0.26, height=0.26, fill_color=col, fill_opacity=0.95, stroke_width=0),
                       T(txt, size, DIM)).arrange(RIGHT, buff=0.14))
    return row.arrange(RIGHT, buff=buff)


def bracket(xa, xb, y, col):
    return VGroup(Line([xa, y, 0], [xb, y, 0], stroke_color=col, stroke_width=4),
                  Line([xa, y - 0.07, 0], [xa, y + 0.07, 0], stroke_color=col, stroke_width=4),
                  Line([xb, y - 0.07, 0], [xb, y + 0.07, 0], stroke_color=col, stroke_width=4))


class Ch16(BeatScene):
    def construct(self):
        # ───────────────────────── b01: three techniques, four questions ─────────────────────────
        with self.beat('c16_b01'):
            tag = self.intro('c16')
            head = T('Potential techniques', 40, INK, weight='BOLD').move_to([0, 2.95, 0])
            specs = [('auto-gain', 'the EMG changes size', C_S2),
                     ('equal weight per position', 'the loss ignores weak positions', C_GM),
                     ('longer window', 'too little context', C_GL)]
            cards = VGroup()
            for title, sub, col in specs:
                box = rr(4.0, 2.4, col)
                t1 = W(title, 28, col, width=chars_for(3.5, 28), align=ORIGIN, weight='BOLD')
                t2 = W(sub, 22, INK, width=chars_for(3.5, 22), align=ORIGIN)
                VGroup(t1, t2).arrange(DOWN, buff=0.3).move_to(box)
                cards.add(VGroup(box, t1, t2))
            cards.arrange(RIGHT, buff=0.3).move_to([0, 1.3, 0])
            self.play(FadeIn(head, shift=DOWN * 0.15), run_time=0.6)
            self.until(self.cue('There are three'))
            for k, c in enumerate(cards):
                self.play(FadeIn(c, shift=UP * 0.2), run_time=0.7)
                if k == 0:
                    self.until(self.cue('equal weight per position'))
                elif k == 1:
                    self.until(self.cue('and a longer window'))
            self.until(self.cue('For each one'))
            qhead = T('For each technique, four questions', 24, DIM).move_to([0, -0.5, 0])
            self.play(FadeIn(qhead), run_time=0.5)
            qs = ['1  What problem does it solve?', '2  How does it work?', '3  Why might it help?',
                  '4  Could it be cheating?']
            chips = VGroup(*[chip(q, 5.9, 0.78, C_HI if i == 3 else FAINT, 24) for i, q in enumerate(qs)])
            chips.arrange_in_grid(2, 2, buff=(0.3, 0.25)).move_to([0, -1.75, 0])
            cues = ['What problem', 'How does it work', 'Why might it help', 'And could it be cheating']
            for ch, cp in zip(chips, cues):
                self.until(self.cue(cp))
                self.play(FadeIn(ch, shift=UP * 0.1), run_time=0.5)
            self.until(self.cue('By cheating we mean'))
            note = T('cheating = information from the test session sneaking into the model', 22, C_HI)
            note.move_to([0, -3.1, 0])
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.6)

        self.play(FadeOut(VGroup(head, cards, qhead, chips, note)), run_time=0.5)

        # ───────────────────────── b02: the size problem + the radio ─────────────────────────
        with self.beat('c16_b02'):
            cap = T('same push, different EMG size', 32, INK, weight='BOLD').move_to([-2.6, 2.85, 0])
            radio_body = rr(3.3, 2.1, C_S2, fill=PANEL)
            spk = VGroup(*[Line([-1.35, y, 0], [-0.25, y, 0], stroke_color=DIM, stroke_width=3)
                           for y in (-0.55, -0.3, -0.05, 0.2, 0.45)]).move_to(radio_body.get_center() + LEFT * 0.75)
            dial = Rectangle(width=0.95, height=0.3, stroke_color=DIM, stroke_width=2, fill_opacity=0).move_to(
                radio_body.get_center() + UP * 0.6 + RIGHT * 0.85)
            knob = Circle(radius=0.42, stroke_color=C_HI, stroke_width=4, fill_color=BG, fill_opacity=1).move_to(
                radio_body.get_center() + DOWN * 0.35 + RIGHT * 0.85)
            ptr = Line(knob.get_center(), knob.get_center() + UP * 0.38, stroke_color=C_HI, stroke_width=5)
            ant = Line(radio_body.get_corner(UR) + LEFT * 0.35,
                       radio_body.get_corner(UR) + LEFT * 0.35 + UP * 0.9 + RIGHT * 0.6, stroke_color=DIM, stroke_width=4)
            radio = VGroup(ant, radio_body, spk, dial, knob, ptr).move_to([4.2, 1.2, 0])
            rlab = T('auto-gain', 30, C_HI, weight='BOLD').next_to(radio, DOWN, buff=0.35)
            rsub = T('adjusts its own volume', 22, INK).next_to(rlab, DOWN, buff=0.15)
            self.play(FadeIn(radio), run_time=0.7)
            self.play(FadeIn(rlab), run_time=0.4)
            self.play(Rotate(ptr, angle=-1.6, about_point=knob.get_center()), run_time=0.7)
            self.play(Rotate(ptr, angle=2.6, about_point=knob.get_center()), FadeIn(rsub), run_time=0.9)
            self.play(Rotate(ptr, angle=-1.0, about_point=knob.get_center()), run_time=0.6)
            self.until(self.cue('The problem it targets'))
            ratios = F3['size_ratio']['JM'][3]['before']       # JM at -0.36 rad, session 2 / session 1
            names = ['GM', 'GL', 'SOL', 'TA']
            H, VM, base = 3.1, 1.25, -1.5
            slot = 6.0 / 4
            sg = VGroup()
            for i, v in enumerate(ratios):
                h = v / VM * H
                xc = -5.6 + slot * (i + 0.5)
                r = Rectangle(width=0.85, height=h, fill_color=MUSCLE_C[i], fill_opacity=0.92, stroke_width=0)
                r.move_to([xc, base + h / 2, 0])
                if abs(v - 1.0) < 0.2:      # keep the number clear of the dashed 1.0 line: put it inside the bar
                    vl = T(fmt(v), 24, BG, weight='BOLD').next_to(r.get_top(), DOWN, buff=0.1)
                else:
                    vl = T(fmt(v), 24, INK).set_stroke(BG, 6, background=True).next_to(r, UP, buff=0.08)
                sg.add(r, vl, T(names[i], 24, MUSCLE_C[i]).move_to([xc, base - 0.35, 0]))
            axis = Line([-5.6, base, 0], [0.4, base, 0], stroke_color=FAINT, stroke_width=1.6)
            ref = DashedVMobject(Line([-5.6, base + H / VM, 0], [0.4, base + H / VM, 0], stroke_color=C_HI,
                                      stroke_width=2), num_dashes=40)
            reflab = T('1.0 = same size in both sessions', 20, C_HI).move_to([-2.6, base + H / VM + 0.4, 0])
            ttl = T('JM · −0.36 rad · session-2 size ÷ session-1 size', 22, DIM).move_to([-2.6, base - 0.95, 0])
            chart = VGroup(axis, ref, reflab, sg)
            self.play(FadeIn(axis), FadeIn(ref), FadeIn(reflab), FadeIn(ttl), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(m) for m in sg], lag_ratio=0.05), run_time=1.2)
            self.play(FadeIn(cap, shift=DOWN * 0.1), run_time=0.5)
            self.until(self.cue('a fixed session-1 scale'))
            fix = VGroup(T('a fixed session-1 scale', 22, C_BAD), T('cannot follow', 22, C_BAD)).arrange(DOWN, buff=0.1)
            fix.move_to([4.2, -2.4, 0])
            self.play(FadeIn(fix), run_time=0.5)

        self.play(FadeOut(VGroup(radio, rlab, rsub, chart, ttl, cap, fix)), run_time=0.5)

        # ───────────────────────── b03: floor and top ─────────────────────────
        with self.beat('c16_b03'):
            t = X3['jm_t'] - 1.0
            sol, lo, top, agc = X3['jm_base'][:, 2], X3['jm_lo'][:, 2], X3['jm_top'][:, 2], X3['jm_agc'][:, 2]
            sel = t <= 47
            ts, ys, los, tops, ags = t[sel][::5], sol[sel][::5], lo[sel][::5], top[sel][::5], agc[sel][::5]
            ymax = 0.5
            P1 = Plot((0, 47), (0, ymax), 9.6, 2.0, xticks=(0, 10, 20, 30, 40), yticks=(0, 0.2, 0.4),
                      xlabel='time in trial (s)', ylabel='envelope', tick_size=18, yfmt=lambda v: f'{v:.1f}')
            place_plot(P1, -0.7, 0.75)
            ttl = T('JM · session 2 · soleus (SOL)', 22, C_SOL).move_to([-3.2, 3.0, 0])
            cy = P1.curve(ts, ys, C_SOL, 2.0)
            self.play(FadeIn(P1), FadeIn(ttl), run_time=0.5)
            self.play(Create(cy), run_time=1.6)
            tu = F3['jm_example']['t_u'] - 1.0         # a moment at 23.5 s
            self.until(self.cue('looking only backwards'))
            now = P1.vline(tu, C_HI, 2.0, dashed=True)
            fut = P1.band(tu, 47, DIM, 0.14)
            futl = T('future: never used', 20, DIM).move_to(P1.p((tu + 47) / 2, ymax * 0.9))
            ya, yb = 2.5, 2.15
            xl, xr, x0 = P1.p(tu, 0)[0], P1.p(tu - 20, 0)[0], P1.p(0, 0)[0]
            b45 = bracket(x0, xl, ya, TOP_C)
            b20 = bracket(xr, xl, yb, LO_C)
            self.play(Create(now), FadeIn(fut), FadeIn(futl), run_time=0.7)
            self.until(self.cue('A floor'))
            l20 = T('last 20 s → floor (a low point)', 20, LO_C).move_to([xl + 0.2 + 1.7, yb, 0])
            self.play(Create(b20), FadeIn(l20), run_time=0.6)
            cl = P1.curve(ts, los, LO_C, 3.5)
            self.play(Create(cl), run_time=2.0, rate_func=linear)
            self.until(self.cue('And a top'))
            l45 = T('up to 45 s back (here: all of the trial so far) → top', 20, TOP_C).next_to(b45, RIGHT, buff=0.2)
            self.play(Create(b45), FadeIn(l45), run_time=0.6)
            ct = P1.curve(ts, tops, TOP_C, 3.5)
            self.play(Create(ct), run_time=2.0, rate_func=linear)
            self.until(self.cue('Every half second'))
            half = chip('updated every 0.5 s', 3.2, 0.5, C_HI, 20, C_HI).move_to([4.6, 3.1, 0])
            self.play(FadeIn(half), run_time=0.5)
            self.until(self.cue('and the signal is rescaled'))
            P2 = Plot((0, 47), (0, 2), 9.6, 1.5, xticks=(0, 10, 20, 30, 40), yticks=(0, 1, 2),
                      xlabel='time in trial (s)', ylabel='rescaled', tick_size=18)
            place_plot(P2, -0.7, -1.85)
            h0 = P2.hline(0, LO_C, 2.0, dashed=True)
            h1 = P2.hline(1, TOP_C, 2.0, dashed=True)
            lab0 = T('floor → 0', 20, LO_C).next_to(P2.p(47, 0.0), RIGHT, buff=0.15)
            lab1 = T('top → 1', 20, TOP_C).next_to(P2.p(47, 1.0), RIGHT, buff=0.15)
            cag = P2.curve(ts, ags, C_SOL, 2.0)
            self.play(FadeIn(P2), FadeIn(h0), FadeIn(h1), FadeIn(lab0), FadeIn(lab1), run_time=0.6)
            self.play(Create(cag), run_time=1.6)
            self.until(self.cue('This is done to every trial'))
            every = T('applied to every trial, training trials included', 22, C_HI).move_to([0, -3.6, 0])
            self.play(FadeIn(every, shift=UP * 0.1), run_time=0.5)

        self.play(FadeOut(VGroup(P1, ttl, cy, now, fut, futl, b20, b45, l20, l45, cl, ct, half, P2, h0, h1, lab0, lab1,
                                 cag, every)), run_time=0.5)

        # ───────────────────────── b04: weight on the trial's own data + safeguards ─────────────────────────
        with self.beat('c16_b04'):
            sec = np.linspace(0, 45, 181)
            n = sec * 100
            wf = np.minimum(n, 2000) / (np.minimum(n, 2000) + 500)
            wt = np.minimum(n, 4500) / (np.minimum(n, 4500) + 500)
            Pw = Plot((0, 45), (0, 1), 6.4, 2.4, xticks=(0, 5, 20, 45), yticks=(0, 0.5, 1),
                      xlabel='seconds of the trial so far', ylabel='weight on own data', tick_size=18,
                      yfmt=lambda v: f'{v:g}', grid=True)
            place_plot(Pw, -2.2, 1.05)
            gap = Polygon(*[Pw.p(s, w) for s, w in zip(sec, wt)], Pw.p(45, 1.0), Pw.p(0, 1.0), stroke_width=0,
                          fill_color=C_S1, fill_opacity=0.28)
            cf = Pw.curve(sec, wf, LO_C, 3.5)
            ct = Pw.curve(sec, wt, TOP_C, 3.5)
            start = W('At the start of a trial there is no history yet: it starts from session-1 values.', 24, INK,
                      width=chars_for(3.9, 24)).move_to([4.6, 1.9, 0])
            self.play(FadeIn(start, shift=LEFT * 0.1), run_time=0.7)
            self.until(self.cue("The trial's own data"))
            self.play(FadeIn(Pw), run_time=0.5)
            self.play(Create(cf), Create(ct), run_time=1.6)
            half_d = Dot(Pw.p(5, 0.5), radius=0.09, color=C_HI)
            half_l = T('half weight at ≈ 5 s', 20, C_HI).next_to(half_d, DR, buff=0.1)
            self.play(FadeIn(half_d), FadeIn(half_l), run_time=0.5)
            self.until(self.cue('levels off at eighty'))
            sess1 = T('session 1 keeps a small share', 20, C_S1).move_to(Pw.p(17, 0.95))
            c90 = T('0.90', 20, TOP_C).next_to(Pw.p(45, 0.9), RIGHT, buff=0.1).shift(UP * 0.1)
            c80 = T('0.80', 20, LO_C).next_to(Pw.p(45, 0.8), RIGHT, buff=0.1).shift(DOWN * 0.1)
            self.play(FadeIn(gap), FadeIn(c80), FadeIn(c90), FadeIn(sess1), run_time=0.7)
            keyl = VGroup(*[VGroup(Line(LEFT * 0.3, RIGHT * 0.3, stroke_color=c, stroke_width=5), T(n, 22, c)).arrange(RIGHT, buff=0.15)
                            for n, c in (('floor level', LO_C), ('top level', TOP_C))]).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
            keyl.move_to([4.3, 0.1, 0])
            self.play(FadeIn(keyl), run_time=0.4)
            self.until(self.cue('A few safeguards'))
            self.wait(0.3)

            def glyph(kind):
                g = VGroup()
                if kind == 0:     # minimum range: tiny signal, range floor
                    g.add(Line([-0.5, -0.12, 0], [0.5, -0.12, 0], stroke_color=DIM, stroke_width=2),
                          Line([-0.5, 0.12, 0], [0.5, 0.12, 0], stroke_color=DIM, stroke_width=2),
                          VMobject(stroke_color=C_SOL, stroke_width=4).set_points_as_corners(
                              [[-0.5, -0.03, 0], [-0.25, 0.04, 0], [0, -0.04, 0], [0.25, 0.03, 0], [0.5, 0, 0]]))
                elif kind == 1:   # clip 0..2
                    g.add(Line([-0.5, -0.3, 0], [0.5, -0.3, 0], stroke_color=DIM, stroke_width=2),
                          Line([-0.5, 0.3, 0], [0.5, 0.3, 0], stroke_color=DIM, stroke_width=2),
                          VMobject(stroke_color=C_SOL, stroke_width=4).set_points_as_corners(
                              [[-0.5, -0.3, 0], [-0.3, -0.3, 0], [0.0, 0.3, 0], [0.5, 0.3, 0]]))
                else:             # spike replaced by last good value
                    g.add(VMobject(stroke_color=C_BAD, stroke_width=3).set_points_as_corners(
                              [[-0.5, -0.1, 0], [-0.15, -0.1, 0], [-0.05, 0.45, 0], [0.05, -0.1, 0], [0.5, -0.1, 0]]),
                          Line([-0.15, -0.1, 0], [0.05, -0.1, 0], stroke_color=C_GOOD, stroke_width=5))
                return g

            tx = ['minimum range: a nearly silent muscle is not blown up', 'clip to 0 – 2',
                  'spikes replaced by the last good value']
            cds = VGroup()
            for k, s in enumerate(tx):
                box = rr(4.1, 1.5, FAINT)
                gl = glyph(k).scale(0.9).move_to(box.get_left() + RIGHT * 0.75)
                tt = W(s, 20, INK, width=chars_for(2.55, 20)).move_to(box.get_center() + RIGHT * 0.55)
                cds.add(VGroup(box, gl, tt))
            cds.arrange(RIGHT, buff=0.22).move_to([0.0, -2.75, 0])
            for k, c in enumerate(cds):
                self.play(FadeIn(c, shift=UP * 0.15), run_time=0.6)
                if k == 0:
                    self.until(self.cue('a clip between zero and two'))
                elif k == 1:
                    self.until(self.cue('and spikes replaced'))

        self.play(FadeOut(VGroup(Pw, cf, ct, gap, half_d, half_l, c80, c90, sess1, cds, start, keyl)), run_time=0.5)

        # ───────────────────────── b05: does it fix the size problem? ─────────────────────────
        with self.beat('c16_b05'):
            rb = F3['size_ratio']['JM'][3]['before'][:3]
            ra = F3['size_ratio']['JM'][3]['after'][:3]
            ttl = T('JM · −0.36 rad · session-2 size ÷ session-1 size', 24, DIM).move_to([-1.0, 3.0, 0])
            gb = gbars(list(zip(rb, ra)), ['GM', 'GL', 'SOL'], MUSCLE_C[:3], MUSCLE_C[:3], 7.0, 2.8, 1.25, vsize=22,
                       label_size=24, bw=0.85)
            for r in gb.before:
                r.set_fill(opacity=0.38)
            put(gb, -1.8, -1.1)
            ref = DashedVMobject(Line([-5.4, -1.1 + 2.8 / 1.25, 0], [1.8, -1.1 + 2.8 / 1.25, 0], stroke_color=C_HI,
                                      stroke_width=2), num_dashes=40)
            # the label sits to the right of the dashed line so it never crosses the bars or their values
            reflab = VGroup(T('1.0 = same size', 20, C_HI), T('in both sessions', 20, C_HI)).arrange(
                DOWN, buff=0.06, aligned_edge=LEFT).next_to(ref, RIGHT, buff=0.15)
            for i, v in enumerate(rb):      # values near the dashed 1.0 line go inside the bar top
                if abs(v - 1.0) < 0.2:
                    gb.bv[i].next_to(gb.before[i].get_top(), DOWN, buff=0.1)
            for i, v in enumerate(ra):
                if abs(v - 1.0) < 0.2:
                    gb.av[i].become(T(fmt(v), 22, BG, weight='BOLD').next_to(gb.after[i].get_top(), DOWN, buff=0.1))
            lg = VGroup(VGroup(Rectangle(width=0.26, height=0.26, fill_color=C_GM, fill_opacity=0.38, stroke_width=0),
                               T('before', 22, DIM)).arrange(RIGHT, buff=0.14),
                        VGroup(Rectangle(width=0.26, height=0.26, fill_color=C_GM, fill_opacity=0.92, stroke_width=0),
                               T('after auto-gain', 22, INK)).arrange(RIGHT, buff=0.14)).arrange(DOWN, buff=0.25,
                                                                                               aligned_edge=LEFT)
            lg.move_to([4.6, -0.5, 0])
            self.play(FadeIn(ttl), FadeIn(gb.axis), FadeIn(ref), FadeIn(reflab), FadeIn(gb.labs), run_time=0.6)
            self.until(self.cue("session 2's sizes"))
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in gb.before], lag_ratio=0.15), FadeIn(lg[0]), run_time=1.0)
            self.play(FadeIn(gb.bv), run_time=0.3)
            self.until(self.cue('to 0.91'))
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in gb.after], lag_ratio=0.15), FadeIn(lg[1]), run_time=0.8)
            self.play(FadeIn(gb.av), run_time=0.25)
            self.until(self.cue('the share of cases'))      # keep the finished chart up while the sentence is read
            self.play(FadeOut(VGroup(ttl, gb, ref, reflab, lg)), run_time=0.5)
            head = T('5 people · GM, GL, SOL · session-2 size off by more than a quarter', 24, DIM).move_to([0, 2.7, 0])
            cds = VGroup()
            for lab, b, a in (('all positions', 32, 18), ('toes-down positions', 53, 30)):
                box = rr(5.8, 3.4, FAINT)
                t0 = T(lab, 28, C_HI).move_to(box.get_top() + DOWN * 0.6)
                nb = T(f'{b} %', 52, C_BAD, weight='BOLD')
                ar = Arrow(LEFT * 0.4, RIGHT * 0.4, color=DIM, buff=0, stroke_width=5)
                na = T(f'{a} %', 52, C_GOOD, weight='BOLD')
                row = VGroup(nb, ar, na).arrange(RIGHT, buff=0.35).move_to(box.get_center() + DOWN * 0.3)
                cds.add(VGroup(box, t0, row))
            cds.arrange(RIGHT, buff=0.4).move_to([0, 0.5, 0])
            note = T("separate check, last week's settings", 22, DIM).move_to([0, -1.7, 0])
            self.play(FadeIn(head), run_time=0.5)
            self.play(FadeIn(cds[0], shift=UP * 0.15), run_time=0.7)
            self.until(self.cue('and from 53'))
            self.play(FadeIn(cds[1], shift=UP * 0.15), run_time=0.7)
            self.play(FadeIn(note), run_time=0.5)

        self.play(FadeOut(VGroup(head, cds, note)), run_time=0.5)

        # ───────────────────────── b06: is it cheating? ─────────────────────────
        with self.beat('c16_b06'):
            q = T('Is it cheating?', 40, INK, weight='BOLD').move_to([-3.6, 2.9, 0])
            ans = T('no, as far as the code audit could tell', 26, C_GOOD).next_to(q, RIGHT, buff=0.4).align_to(q, DOWN)
            items = ['uses only the past EMG of the trial being predicted', 'never uses torque from session 2',
                     "starting values come from session 1's fit part only",
                     'the on/off decision uses session-1 torque only']
            rows = VGroup(*[check_row(s, 26, 56) for s in items]).arrange(DOWN, buff=0.38, aligned_edge=LEFT)
            rows.move_to([0, 0.4, 0]).align_to(q, LEFT)
            box = rr(11.4, 1.3, C_HI)
            ctxt = W('catch: the envelope underneath still uses the future, so the whole thing stays offline', 24, C_HI,
                     width=chars_for(10.4, 24), align=ORIGIN).move_to(box)
            cav = VGroup(box, ctxt).move_to([0, -2.75, 0])
            self.play(FadeIn(q), run_time=0.5)
            self.play(FadeIn(ans), run_time=0.6)
            self.until(self.cue('At test time'))
            self.play(FadeIn(rows[0], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('never any torque'))
            self.play(FadeIn(rows[1], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('and its starting values'))
            self.play(FadeIn(rows[2], shift=RIGHT * 0.2), run_time=0.6)
            self.wait(0.8)
            self.play(FadeIn(rows[3], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('A real device'))
            self.play(FadeIn(cav, shift=UP * 0.15), run_time=0.7)

        self.play(FadeOut(VGroup(q, ans, rows, cav)), run_time=0.5)

        # ───────────────────────── b07: CH pushes both ways → off ─────────────────────────
        with self.beat('c16_b07'):
            dirs = F3['per_subject']['CH']['push_directions']
            P = Plot((-0.5, 7.5), (-62, 38), 7.4, 4.0, yticks=(-50, -25, 0, 25), xlabel='CH · session-1 trials (8)',
                     ylabel='push torque (Nm)', tick_size=18)
            place_plot(P, -2.0, -0.15)
            zero = P.hline(0, DIM, 1.5, dashed=False)
            ttl = T('CH · direction of each session-1 push', 24, INK).move_to([-2.0, 2.85, 0])
            arrows, vlabs = VGroup(), VGroup()
            for i, v in enumerate(dirs):
                a = Arrow(P.p(i, 0), P.p(i, v), buff=0, color=C_TQ, stroke_width=7, max_tip_length_to_length_ratio=0.35)
                arrows.add(a)
                vlabs.add(T(fmt(v, 0, signed=True), 20, C_TQ).next_to(P.p(i, v), UP if v > 0 else DOWN, buff=0.1))
            self.play(FadeIn(P), FadeIn(zero), FadeIn(ttl), run_time=0.6)
            self.until(self.cue('who pushes in both directions'))
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12), FadeIn(vlabs), run_time=2.0)
            badge = VGroup(chip('pushes both ways', 3.5, 0.7, C_BAD, 24, C_BAD),
                           chip('auto-gain OFF', 3.5, 0.7, C_BAD, 24, C_BAD)).arrange(DOWN, buff=0.15)
            badge.move_to([4.7, 2.0, 0])
            self.play(FadeIn(badge, shift=LEFT * 0.15), run_time=0.6)
            self.until(self.cue('For someone who pushes both ways'))
            why = W("calf or shin? The network probably needs each muscle's real size to tell. Auto-gain would erase it. "
                    'Mechanism not isolated.', 20, INK, width=chars_for(3.3, 20)).move_to([4.7, -0.2, 0])
            self.play(FadeIn(why), run_time=0.6)
            self.until(self.cue('The rule that decides'))
            warn = VGroup(rr(3.6, 1.7, C_HI),
                          W('rule designed after seeing CH fail: a mild form of tuning on the test', 20, C_HI,
                            width=chars_for(3.2, 20)))
            warn[1].move_to(warn[0])
            warn.move_to([4.7, -2.45, 0])
            self.play(FadeIn(warn, shift=UP * 0.15), run_time=0.6)

        self.play(FadeOut(VGroup(P, zero, ttl, arrows, vlabs, badge, why, warn)), run_time=0.5)

        # ───────────────────────── b08: result ─────────────────────────
        with self.beat('c16_b08'):
            td = Y6['x05_group']       # [toes-down|other][R2|MAE][before|after]
            r2b, r2a, mb, ma = td[0, 0, 0], td[0, 0, 1], td[0, 1, 0], td[0, 1, 1]
            ttl = T('toes-down positions · 6 people · seeds 0–2', 24, DIM).move_to([-3.1, 2.85, 0])
            gA = gbars([(r2b, r2a)], ['R²'], [GREY_B], [C_GOOD], 2.0, 2.4, 1.1, label_size=22, vsize=22, bw=0.8)
            gB = gbars([(mb, ma)], ['error (Nm)'], [GREY_B], [C_GOOD], 2.6, 2.4, 1.1, label_size=22, vsize=22, bw=0.8)
            put(gA, -5.0, -0.8)
            put(gB, -2.1, -0.8)
            lg = legend([(GREY_B, 'rebuilt baseline (X00f)'), (C_GOOD, 'auto-gain (X05)')], 20).move_to([-3.3, 2.3, 0])
            self.play(FadeIn(ttl), FadeIn(lg), run_time=0.5)
            for g in (gA, gB):
                self.play(FadeIn(g.axis), FadeIn(g.labs),
                          LaggedStart(*[GrowFromEdge(r, DOWN) for r in (g.before[0], g.after[0])], lag_ratio=0.25),
                          FadeIn(g.bv), FadeIn(g.av), run_time=1.2)
                if g is gA:
                    self.until(self.cue('and the error'))
            self.until(self.cue('The other positions'))
            ot = Y6['x05_group'][1]
            oth = VGroup(T('other positions', 22, DIM),
                         T(f'R² {fmt(ot[0, 0], 3)} → {fmt(ot[0, 1], 3)}   ·   error {fmt(ot[1, 0])} → {fmt(ot[1, 1])} Nm',
                           22, INK)).arrange(DOWN, buff=0.12).move_to([-3.3, -2.6, 0])
            self.play(FadeIn(oth, shift=UP * 0.1), run_time=0.6)
            self.until(self.cue('EG, JM and IES'))
            subj = [str(s) for s in Y6['x05_subjects']]
            rb_, ra_ = Y6['x05_td_r2_before'], Y6['x05_td_r2_after']
            order = ['EG', 'JM', 'IES', 'YES', 'HM', 'CH']
            x0, x1 = 4.4, 6.5
            hdr = T('toes-down R² per person', 22, DIM).move_to([3.9, 3.05, 0])
            axl = Line([x0, 2.4, 0], [x1, 2.4, 0], stroke_color=FAINT, stroke_width=1.6)
            ticks = VGroup()
            for v in (0, 0.5, 1.0):
                xx = x0 + (x1 - x0) * v
                ticks.add(Line([xx, 2.4, 0], [xx, 2.33, 0], stroke_color=FAINT, stroke_width=1.5),
                          T(f'{v:g}', 18, DIM).move_to([xx, 2.1, 0]))
            prow, vals = VGroup(), VGroup()
            for i, s in enumerate(order):
                k = subj.index(s)
                y = 1.45 - i * 0.72
                xb, xa = x0 + (x1 - x0) * rb_[k], x0 + (x1 - x0) * ra_[k]
                # the narration says YES and HM barely change: they get the dimmed style like CH
                up = ra_[k] - rb_[k] > 0.02 and s not in ('YES', 'HM')
                col = C_GOOD if up else DIM
                lab = T(s, 22, INK).move_to([0.9, y, 0])
                dots = VGroup(Line([x0, y, 0], [x1, y, 0], stroke_color=FAINT, stroke_width=1),
                              Dot([xb, y, 0], radius=0.09, color=GREY_B))
                if xa - xb > 0.12:
                    dots.add(Arrow([xb, y, 0], [xa, y, 0], buff=0.1, color=col, stroke_width=4, tip_length=0.15))
                dots.add(Dot([xa, y, 0], radius=0.09, color=col if up else GREY_B))
                prow.add(VGroup(lab, dots))
                if s == 'CH':
                    t_ = T('unchanged (off)', 18, DIM)
                else:
                    t_ = T(f'{fmt(rb_[k])} → {fmt(ra_[k])}', 20, INK if up else DIM)
                vals.add(t_.move_to([1.7 + t_.width / 2, y, 0]))
            self.play(FadeIn(hdr), FadeIn(axl), FadeIn(ticks), run_time=0.5)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(r), FadeIn(v)) for r, v in zip(prow, vals)], lag_ratio=0.5),
                      run_time=max(1.5, min(5.0, self.left() - 1.3)))

        self.play(FadeOut(VGroup(ttl, lg, gA, gB, oth, hdr, axl, ticks, prow, vals)), run_time=0.5)

        # ───────────────────────── b09: caveats ─────────────────────────
        with self.beat('c16_b09'):
            hd = T('Caveats', 38, INK, weight='BOLD').move_to([-5.2, 2.85, 0])
            x05s = Y6['yes_m486_r2_X05']
            x00s = Y6['yes_m486_r2_X00f']
            rows = [
                ('assumes a similar range of effort in both sessions',
                 'if session 2 pushes were gentler, they would be stretched back up'),
                ('can amplify noise where a muscle barely activates',
                 'YES −0.49 rad got worse: auto-gain ' + ', '.join(fmt(v) for v in x05s) + ' vs baseline '
                 + ', '.join(fmt(v) for v in x00s) + ' (R²)'),
                ('removes most size differences between positions',
                 'so the angle has to carry the scale (our reading)'),
                ('failed the pre-written rules (YES + EG)',
                 'gain not clearly above the noise; learning from session 2 instead, toes-down slightly worse'),
            ]
            cds = VGroup()
            for k, (a, b) in enumerate(rows):
                box = rr(12.6, 1.28, C_BAD if k == 3 else FAINT)
                num = T(str(k + 1), 32, C_HI, weight='BOLD').move_to(box.get_left() + RIGHT * 0.5)
                tt = VGroup(T(a, 26, INK, weight='BOLD'), T(b, 20, DIM)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
                tt.move_to(box.get_center()).align_to(box, LEFT).shift(RIGHT * 1.0)
                cds.add(VGroup(box, num, tt))
            cds.arrange(DOWN, buff=0.17).move_to([0, -0.4, 0])
            self.play(FadeIn(hd), run_time=0.4)
            cues = ['Auto-gain assumes', 'Where a muscle barely', 'It also removes', 'And on the two screening']
            for c, cp in zip(cds, cues):
                self.until(self.cue(cp))
                self.play(FadeIn(c, shift=UP * 0.15), run_time=0.7)
