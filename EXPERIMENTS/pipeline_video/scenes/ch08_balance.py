from manim import *

from style import (ANGLE_C, C_HI, C_S1, C_TQ, DIM, FAINT, INK, PANEL, V, BeatScene, CodePanel, Plot, T, chapter_tag,
                   code_at_bottom, facts, load)

FF = facts()
F5, F7 = FF['x05'], FF['x07']
X5 = load('x05')
KN = np.array(F5['knots'])
SD = np.array(F5['sd'])


def lab_angle(a):
    return f'{a:+.2f}'.replace('-', '−')


class Ch08(BeatScene):
    def construct(self):
        tag = chapter_tag(8, 'Balanced positions')
        self.play(FadeIn(tag), run_time=0.4)

        # 8.1 knots
        with self.beat('c08_b01') as d:
            nl = NumberLine(x_range=[-0.8, 0.3, 0.1], length=11.5, include_numbers=False, color=FAINT).move_to([0, 1.6, 0])
            dots = VGroup(*[Dot(nl.n2p(a), radius=0.11, color=c) for a, c in zip(KN, ANGLE_C)])
            labs = VGroup(*[T(lab_angle(a), 16, c).next_to(nl.n2p(a), DOWN, buff=0.25) for a, c in zip(KN, ANGLE_C)])
            ttl = T('EG · session-1 training angles (rad) → 8 knots', 20, INK).next_to(nl, UP, buff=0.6)
            tol = VGroup(Line(nl.n2p(KN[3] - 0.01), nl.n2p(KN[3] + 0.01), color=C_HI, stroke_width=8).shift(UP * 0.3),
                         T('0.02 rad: closer angles would merge', 14, C_HI).next_to(nl.n2p(KN[3]), UP, buff=0.45))
            code = code_at_bottom(CodePanel('knots'))
            self.play(Create(nl), FadeIn(ttl), run_time=0.6)
            self.play(LaggedStart(*[GrowFromCenter(x) for x in dots], lag_ratio=0.1), FadeIn(labs), run_time=1.2)
            self.show_code(code)
            self.mark(code, 35, 36)
            self.play(FadeIn(tol), run_time=0.5)
        self.play(FadeOut(VGroup(tol, code, ttl)), VGroup(nl, dots, labs).animate.shift(DOWN * 1.1), run_time=0.5)

        # 8.2 one standard deviation per knot
        with self.beat('c08_b02') as d:
            bars, vals = VGroup(), VGroup()
            for a, s, c in zip(KN, SD, ANGLE_C):
                r = Rectangle(width=0.6, height=1.7 * s / SD.max(), stroke_width=0, fill_color=c, fill_opacity=0.9)
                r.move_to(nl.n2p(a) + UP * 0.2, aligned_edge=DOWN)
                bars.add(r)
                vals.add(T(f'{s:.2f}', 16, INK).next_to(r, UP, buff=0.06))
            ttl = T('SD of the training targets at each angle (Nm)', 20, INK).move_to([0, 3.0, 0])
            code = code_at_bottom(CodePanel('sd', 144, 150))
            self.play(FadeIn(ttl), run_time=0.4)
            self.show_code(code)
            self.mark(code, 150)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1), FadeIn(vals), run_time=1.6)
            self.wait(max(0.1, d * 0.6 - 3.3))
            sw = T('the dashboard says “scaled by its swing”: in the code it is this standard deviation', 16, C_HI)
            sw.move_to([0, -0.55, 0])
            self.play(FadeIn(sw), run_time=0.6)
        self.play(FadeOut(VGroup(nl, dots, labs, bars, vals, ttl, code, sw)), run_time=0.5)

        # 8.3 divide, multiply back
        with self.beat('c08_b03') as d:
            k_hi = F5['sd_argmax']
            ya, ta = X5['knot0_target'], X5['knot0_t']
            yb, tb = X5[f'knot{k_hi}_target'], X5[f'knot{k_hi}_t']
            ma, mb = ta <= 21, tb <= 21
            P1 = Plot((1, 21), (-34, 4), 4.8, 2.6, yticks=(-30, -20, -10, 0), xticks=(1, 11, 21), tick_size=13,
                      ylabel='target (Nm)', xlabel='time in trial (s)', grid=True).move_to([-3.6, 1.2, 0])
            P2 = Plot((1, 21), (-3.2, 1.2), 4.8, 2.6, yticks=(-3, -2, -1, 0, 1), xticks=(1, 11, 21), tick_size=13,
                      xlabel='time in trial (s)', grid=True).move_to([3.9, 1.2, 0])
            a1, b1 = P1.curve(ta[ma], ya[ma], ANGLE_C[0], 2.0), P1.curve(tb[mb], yb[mb], ANGLE_C[k_hi], 2.0)
            a2 = P2.curve(ta[ma], ya[ma] / SD[0], ANGLE_C[0], 2.0)
            b2 = P2.curve(tb[mb], yb[mb] / SD[k_hi], ANGLE_C[k_hi], 2.0)
            key = VGroup(T(f'{lab_angle(KN[0])} rad', 16, ANGLE_C[0]), T(f'{lab_angle(KN[k_hi])} rad', 16, ANGLE_C[k_hi]))
            key.arrange(RIGHT, buff=0.5).move_to([0.15, 3.15, 0])
            arr = Arrow(P1.get_right(), P2.get_left(), buff=0.25, color=DIM)
            al = T('target ÷ SD of its angle', 16, C_HI).next_to(P2, UP, buff=0.08)
            code = code_at_bottom(CodePanel('sd', 151, 152))
            self.play(FadeIn(P1), FadeIn(key), Create(a1), Create(b1), run_time=1.2)
            self.show_code(code)
            self.mark(code, 152)
            self.play(GrowArrow(arr), FadeIn(al), FadeIn(P2), TransformFromCopy(a1, a2), TransformFromCopy(b1, b2), run_time=1.2)
            self.wait(max(0.1, d * 0.55 - 3.4))
            code2 = code_at_bottom(CodePanel('pred_scale'))
            self.play(FadeOut(code), FadeIn(code2), run_time=0.5)
            self.mark(code2, 183, 184)
        self.play(FadeOut(VGroup(P1, P2, a1, b1, a2, b2, key, arr, al, code2)), run_time=0.5)

        # 8.4 why: the weight of each angle in the loss
        with self.beat('c08_b04') as d:
            f1 = T('Σ over one angle of (y − ŷ)²  =  N · SD² · (1 − R²)', 28, INK).move_to([0, 2.9, 0])
            f2 = T('Σ over one angle of ((y − ŷ) / SD)²  =  N · (1 − R²)', 28, C_HI).move_to([0, 2.9, 0])
            W = (SD / SD.min()) ** 2
            base = np.array([-4.9 + i * 1.4 for i in range(8)])
            bars = VGroup(*[Rectangle(width=0.8, height=2.4 * w / W.max(), stroke_width=0, fill_color=c, fill_opacity=0.9)
                            .move_to([x, -0.6, 0], aligned_edge=DOWN) for x, w, c in zip(base, W, ANGLE_C)])
            vals = VGroup(*[T(f'{w:.1f}×', 15, INK).next_to(b, UP, buff=0.05) for b, w in zip(bars, W)])
            labs = VGroup(*[T(lab_angle(a), 14, c).move_to([x, -0.85, 0]) for x, a, c in zip(base, KN, ANGLE_C)])
            cap = T('weight of each angle in plain mean squared error, for the same R² (relative to −0.74 rad)', 15, DIM)
            cap.move_to([0, 2.25, 0])
            eq = VGroup(*[Rectangle(width=0.8, height=2.4 / W.max() * 4, stroke_width=0, fill_color=c, fill_opacity=0.9)
                          .move_to([x, -0.6, 0], aligned_edge=DOWN) for x, c in zip(base, ANGLE_C)])
            ev = VGroup(*[T('1×', 15, INK).next_to(b, UP, buff=0.05) for b in eq])
            ncap = T(f"after ÷ SD: every angle has {V['n_train']} windows → loss = mean of (1 − R²) over the angles", 15, C_HI)
            ncap.move_to([0, 2.25, 0])
            self.play(FadeIn(f1), run_time=0.8)
            self.play(FadeIn(cap), FadeIn(labs), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08),
                      FadeIn(vals), run_time=1.6)
            self.wait(max(0.1, d * 0.5 - 2.6))
            self.play(FadeOut(f1), FadeOut(cap), ReplacementTransform(bars, eq), ReplacementTransform(vals, ev), run_time=1.2)
            self.play(FadeIn(f2), FadeIn(ncap), run_time=0.6)
        # 8.5 in the trained model
        with self.beat('c08_b05') as d:
            sh = F7['shares']
            plain, scaled = np.array(sh['plain']), np.array(sh['scaled'])
            self.play(FadeOut(VGroup(f2, eq, ev, ncap, labs)), run_time=0.4)
            B = Plot((-0.6, 7.6), (0, 0.36), 11.0, 2.8, yticks=(0, 0.1, 0.2, 0.3), tick_size=13, grid=True,
                     ylabel='share of the training loss', yfmt=lambda v: f'{100 * v:.0f} %').move_to([0.3, 1.0, 0])
            g = VGroup()
            for i in range(8):
                for j, (vals_, op) in enumerate(((plain, 0.35), (scaled, 0.95))):
                    v = float(vals_[i])
                    r = Rectangle(width=0.38, height=max(0.005, B.p(0, v)[1] - B.p(0, 0)[1]), stroke_width=0,
                                  fill_color=ANGLE_C[i], fill_opacity=op).move_to(B.p(i - 0.21 + 0.42 * j, 0), aligned_edge=DOWN)
                    g.add(r)
            nums = VGroup(T(f"{100 * plain[0]:.1f} %", 15, DIM).next_to(g[0], UP, buff=0.05),
                          T(f"{100 * scaled[0]:.1f} %", 16, C_HI).next_to(g[1], UP, buff=0.05))
            xl = VGroup(*[T(lab_angle(a), 13, c).next_to(B.p(i, 0), DOWN, buff=0.12) for i, (a, c) in enumerate(zip(KN, ANGLE_C))])
            key = T('faint: plain squared errors (Nm²) · solid: the scaled loss D01 trains on · trained weights, training windows',
                    14, DIM).next_to(B, UP, buff=0.12)
            self.play(FadeIn(B), FadeIn(xl), FadeIn(key), run_time=0.5)
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in g], lag_ratio=0.05), run_time=1.6)
            self.play(FadeIn(nums), run_time=0.5)
        self.play(FadeOut(VGroup(B, g, nums, xl, key, tag)), run_time=0.6)
