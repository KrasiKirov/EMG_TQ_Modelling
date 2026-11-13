import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, Dot, FadeIn, FadeOut, LaggedStart, Line, Transform, VGroup)

from common3 import place
from style3 import (C_BAD, C_GM, C_GOOD, C_HI, C_PRED, C_TQ, DIM, FAINT, INK, BeatScene, Plot, T, W, card, facts, load)

C_CAL = C_GM        # corrected prediction


class Ch05(BeatScene):
    def construct(self):
        F = facts()['v01']
        ex, grp = F['calibration_example'], F['group']
        z = load('v01')
        t, y, p, pc = (z[k].astype(float) for k in ('cal_t', 'cal_y', 'cal_p', 'cal_pc'))

        def make_plot(width=12.0, height=4.6, left=-5.7, bottom=-2.4):
            return place(Plot((0, 89), (-4, 8), width, height, xticks=[0, 10, 30, 60, 89], yticks=[-4, 0, 4, 8],
                              xfmt=lambda v: f'{v:g} s', ylabel='torque (Nm)'), left, bottom)

        with self.beat('c05_b01'):
            tag = self.intro('c05')
            P = make_plot()
            cy, cp = P.curve(t, y, C_TQ, 2.2), P.curve(t, p, C_PRED, 2.2)
            cap = T('YES · session 2 · −0.381 rad · model trained on session 1 only, used as it is', 22, DIM)\
                .next_to(P, UP, buff=0.12)
            leg = VGroup(T('measured', 20, C_TQ), T('prediction', 20, C_PRED)).arrange(RIGHT, buff=0.4)\
                .next_to(P, DOWN, buff=0.15)
            self.play(FadeIn(P), FadeIn(cap), Create(cy), run_time=1.4)
            self.play(Create(cp), FadeIn(leg), run_time=1.4)
            self.until(self.cue('but it sits about'))
            gap = card([f'about {abs(ex["bias_raw_scored"]):.1f} Nm too low · swings a little too big'], width=6.0, size=22,
                       color=C_BAD).move_to([2.6, 2.6, 0])
            self.play(FadeOut(cap), FadeIn(gap), run_time=0.6)
            self.wait(self.left())

        with self.beat('c05_b02'):
            self.play(FadeOut(gap), run_time=0.4)
            band = P.band(0, 10, C_HI, 0.14)
            bl = T('first 10 s', 20, C_HI).next_to(P.p(5, 8), UP, buff=0.05)
            self.play(FadeIn(band), FadeIn(bl), run_time=0.6)
            whole = VGroup(P, cy, cp, band, bl, leg)
            self.play(whole.animate.scale(0.55).to_edge(LEFT, buff=0.4).shift(DOWN * 0.3), run_time=0.8)
            m = t < 10
            idx = np.linspace(0, m.sum() - 1, 160).astype(int)
            xs, ys = p[m][idx], y[m][idx]
            S = place(Plot((-4, 6), (-1, 8), 5.4, 4.4, xticks=[-4, 0, 4], yticks=[0, 4, 8], xlabel='prediction (Nm)',
                           ylabel='measured (Nm)'), 1.2, -2.2)
            dots = VGroup(*[Dot(S.p(a, b), radius=0.035, color=C_HI) for a, b in zip(xs, ys)])
            self.play(FadeIn(S), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.01), run_time=1.6)
            self.until(self.cue('draw the best straight'))
            a_, b_ = ex['gain'], ex['offset']
            line = Line(S.p(-4, a_ * -4 + b_), S.p(6, a_ * 6 + b_), stroke_color=C_CAL, stroke_width=4)
            eq = T(f'torque ≈ {a_:.2f} × prediction + {b_:.2f} Nm', 26, C_CAL).next_to(S, UP, buff=0.2)
            self.play(Create(line), FadeIn(eq), run_time=1.0)
            self.until(self.cue('a stretch'))
            st = VGroup(T(f'stretch = {a_:.2f}', 24, C_CAL), T(f'shift = {b_:.2f} Nm', 24, C_CAL))\
                .arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to(S.p(-1.6, 6.6))
            self.play(FadeIn(st), run_time=0.6)
            self.wait(self.left())

        with self.beat('c05_b03'):
            whole.remove(bl)
            self.play(*[FadeOut(o) for o in (S, dots, line, eq, st, bl)], run_time=0.5)
            self.play(whole.animate.scale(1 / 0.55).move_to([0.2, -0.05, 0]), run_time=0.8)
            target = P.curve(t, pc, C_CAL, 2.6)
            self.play(Transform(cp, target), run_time=1.6)
            leg2 = VGroup(T('measured', 20, C_TQ), T('corrected prediction', 20, C_CAL)).arrange(RIGHT, buff=0.4)\
                .move_to(leg)
            self.play(Transform(leg, leg2), run_time=0.4)
            self.until(self.cue('To keep the test fair'))
            grey = P.band(0, 30, FAINT, 0.55)
            gl = T('0–30 s: not scored', 20, DIM).next_to(P.p(15, 8), UP, buff=0.05)
            self.play(FadeIn(grey), FadeIn(gl), run_time=0.6)
            self.until(self.cue('Here, the error drops'))
            badge = card([f'error after 30 s: {ex["mae_raw_scored"]:.2f} → {ex["mae_cal_scored"]:.2f} Nm'], width=6.2,
                         size=26, color=C_GOOD).move_to([2.6, 2.75, 0])
            self.play(FadeIn(badge, shift=DOWN * 0.2), run_time=0.6)
            self.wait(self.left())

        with self.beat('c05_b04'):
            self.clear_all(keep=[tag])
            how = T('per angle · first 10 s · two numbers · no retraining', 28, C_HI).move_to([0, 2.3, 0])
            rows = VGroup(
                T('all 6 people × 3 seeds, scored from 30 s', 22, DIM),
                T(f'toes-down angles   {grp["toes_mae"][0]:.2f} → {grp["toes_mae"][1]:.2f} Nm   '
                  f'(R² {grp["toes_r2"][0]:.2f} → {grp["toes_r2"][1]:.2f})', 30, INK),
                T(f'other angles   {grp["other_mae"][0]:.2f} → {grp["other_mae"][1]:.2f} Nm', 30, INK),
            ).arrange(DOWN, buff=0.35).move_to([0, 0.2, 0])
            foot = T('needs the torque sensor for those 10 s, at each angle', 24, DIM).move_to([0, -2.4, 0])
            self.play(FadeIn(how), run_time=0.6)
            self.until(self.cue('Across all six people'))
            self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.15) for r in rows], lag_ratio=0.4), run_time=1.6)
            self.until(self.cue('It needs the torque'))
            self.play(FadeIn(foot), run_time=0.5)
            self.wait(self.left())
