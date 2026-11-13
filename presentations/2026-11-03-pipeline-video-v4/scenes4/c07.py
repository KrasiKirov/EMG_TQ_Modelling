import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, CurvedArrow, Dot, FadeIn, FadeOut, LaggedStart, Line, MoveAlongPath,
                   Rectangle, RoundedRectangle, Transform, VGroup)

from style4 import (C_BAD, C_FIT, C_GOOD, C_HI, C_PRED, C_TQ, C_VAL, DIM, FAINT, INK, PANEL, BeatScene, T, W, card,
                    load)


def sample_row(n, x0, x1, y, color=DIM, r=0.05):
    xs = np.linspace(x0, x1, n)
    return VGroup(*[Dot([x, y, 0], radius=r, color=color) for x in xs]), xs


class Ch07(BeatScene):
    def construct(self):
        z = load('v01')
        with self.beat('c07_b01'):
            tag = self.intro('c07')
            head = T('1 · a muscle model (CEINMS, the earlier ankle project)', 26, C_VAL).move_to([0, 2.7, 0])
            eq = T('u(t) = α·e(t − d) − β₁·u(t − 1) − β₂·u(t − 2)', 32, INK).move_to([0, 1.8, 0])
            self.play(FadeIn(head), FadeIn(eq), run_time=0.8)
            dots, xs = sample_row(26, -6.0, 6.0, 0.2, DIM, 0.06)
            links = VGroup(*[Arrow([xs[k] + 0.06, 0.2, 0], [xs[k + 1] - 0.06, 0.2, 0], buff=0, stroke_width=2,
                                   max_tip_length_to_length_ratio=0.35, color=FAINT) for k in range(len(xs) - 1)])
            lab = T('each sample’s state is computed from the previous ones', 24, DIM).move_to([0, -0.5, 0])
            self.until(self.cue('So the model carries'))
            self.play(FadeIn(dots), FadeIn(links), FadeIn(lab), run_time=0.8)
            state = Dot([xs[0], 0.2, 0], radius=0.13, color=C_VAL)
            self.play(FadeIn(state), run_time=0.2)
            self.play(state.animate.move_to([xs[-1], 0.2, 0]), run_time=2.2)
            rule = card(['the state is carried through the whole recording → it must run in time order'], width=10,
                        size=26, color=C_VAL).move_to([0, -1.9, 0])
            self.play(FadeIn(rule), run_time=0.6)
            self.wait(self.left())
            left = VGroup(head, eq, dots, links, lab, state, rule)

        with self.beat('c07_b02'):
            self.play(left.animate.scale(0.5).move_to([-3.6, 1.4, 0]), run_time=0.8)
            t = z['eg_t']
            m = (t >= 23.5) & (t <= 29.5)
            from style4 import Plot
            from common4 import place
            P = place(Plot((23.5, 29.5), (-35, 2), 6.0, 2.2), 0.8, 0.2)
            curve = P.curve(t[m], z['eg_tq'][m], C_TQ, 2.4)
            head2 = T('2 · our LSTM', 26, C_FIT).next_to(P, UP, buff=0.35)
            te = 26.0
            box = Rectangle(width=abs(P.p(te, 0)[0] - P.p(te - 0.19, 0)[0]), height=2.3, stroke_color=C_HI,
                            stroke_width=2.5, fill_color=C_HI, fill_opacity=0.2).move_to([(P.p(te, 0)[0] + P.p(te - 0.19, 0)[0]) / 2, P.p(te, -16)[1], 0])
            dot = Dot(P.p(te, float(np.interp(te, t, z['eg_tq']))), radius=0.07, color=C_HI)
            l2 = W('each prediction uses only the last 0.2 s · its memory starts empty for every window', 22, INK,
                   width=52).next_to(P, DOWN, buff=0.3)
            self.play(FadeIn(head2), FadeIn(P), FadeIn(curve), run_time=0.7)
            self.play(FadeIn(box), FadeIn(dot), run_time=0.5)
            self.play(box.animate.shift(RIGHT * 0.6), dot.animate.move_to(P.p(te + 0.33, float(np.interp(te + 0.33, t, z['eg_tq'])))),
                      run_time=1.0)
            self.play(FadeIn(l2), run_time=0.6)
            self.until(self.cue('In signal-processing terms'))
            tags = VGroup(card(['carried state', 'like an IIR filter'], width=4.6, size=24, color=C_VAL),
                          card(['fixed look-back (0.2 s)', 'like an FIR filter / impulse response'], width=5.6, size=24,
                               color=C_FIT)).arrange(RIGHT, buff=0.6).move_to([0, -2.8, 0])
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in tags], lag_ratio=0.4), run_time=1.0)
            self.wait(self.left())
