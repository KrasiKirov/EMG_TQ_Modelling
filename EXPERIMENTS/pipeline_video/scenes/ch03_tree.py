from manim import *

from style import (C_ANGLE, C_GM, C_HI, C_PASSIVE, C_PRED, C_S1, C_S2, C_SOL, C_TQ, DIM, FAINT, INK, MUSCLE_C, PANEL, V,
                   BeatScene, CodePanel, Plot, T, chapter_tag, code_at_bottom, envelope_curve, facts, load)

F = facts()
X1, X2, X3, X4, X5, X8 = (load(k) for k in ('x01', 'x02', 'x03', 'x04', 'x05', 'x08'))
BW, BH = 2.05, 1.3


def box(title, col, mini=None):
    r = RoundedRectangle(corner_radius=0.1, width=BW, height=BH, stroke_color=col, stroke_width=2, fill_color=PANEL,
                         fill_opacity=1)
    t = T(title, 14, INK).move_to(r.get_top() + DOWN * 0.2)
    g = VGroup(r, t)
    if mini is not None:
        mini.scale_to_fit_width(BW - 0.3)
        if mini.height > BH - 0.5:
            mini.scale_to_fit_height(BH - 0.5)
        mini.move_to(r.get_center() + DOWN * 0.14)
        g.add(mini)
    return g


def mini_curve(x, y, col, w=1.8, h=0.75, lo=None, hi=None, n=400):
    lo = float(np.min(y)) if lo is None else lo
    hi = float(np.max(y)) if hi is None else hi
    p = Plot((float(x[0]), float(x[-1])), (lo, hi + 1e-9), w, h, axis_color=PANEL)
    return VGroup(p, envelope_curve(p, x, y, col, 1.5, n))


def grid_icon(rows=5, cols=10, col=C_S1):
    g = VGroup(*[Square(0.12, stroke_width=0.6, stroke_color=PANEL, fill_color=c, fill_opacity=0.9)
                 for r, c in zip(range(rows), MUSCLE_C + [C_ANGLE]) for _ in range(cols)])
    g.arrange_in_grid(rows=rows, cols=cols, buff=0.015)
    return g


def lstm_icon(col=INK):
    cells = VGroup(*[RoundedRectangle(corner_radius=0.04, width=0.28, height=0.2, stroke_color=col, stroke_width=1.5)
                     for _ in range(8)]).arrange_in_grid(rows=2, cols=4, buff=(0.12, 0.18))
    return cells


def sd_bars():
    sd = np.array(F['x05']['sd'])
    return VGroup(*[Rectangle(width=0.14, height=0.65 * v / sd.max(), stroke_width=0, fill_color=C_S1, fill_opacity=0.9)
                    for v in sd]).arrange(RIGHT, buff=0.05, aligned_edge=DOWN)


class Ch03(BeatScene):
    def construct(self):
        tag = chapter_tag(3, 'The pipeline in one picture')
        self.play(FadeIn(tag), run_time=0.4)
        xs = [-4.55 + i * 2.45 for i in range(5)]
        yA, yB, yC = 2.05, 0.45, -1.75

        t1, s1sol = X1['raw_t'], X1['raw_sol']
        pe = X2['piece_env']
        sel = (X3['eg_t'] >= 10) & (X3['eg_t'] <= 30)
        maps = F['x04']['maps']['EG']['static']
        A = [box('raw EMG · 1 kHz', C_S1, mini_curve(t1, s1sol, C_SOL)),
             box('envelope', C_S1, mini_curve(X2['piece_t'], pe, C_SOL)),
             box('÷ max · auto-gain', C_S1, mini_curve(X3['eg_t'][sel], X3['eg_agc'][sel, 2], C_SOL)),
             box('0.2 s windows', C_S1, grid_icon()),
             box('LSTM 8 × 2', C_S1, lstm_icon())]
        mp = Plot((-0.8, 0.3), (-11, 4), 1.8, 0.75, axis_color=PANEL)
        mapc = VGroup(mp, VMobject(stroke_color=C_S1, stroke_width=2).set_points_as_corners([mp.p(a, t) for a, t in maps['S1']]))
        selp = X4['p20_t'] <= 20
        B = [box('still holds (passive)', C_S1, mini_curve(X4['p20_t'][selp], X4['p20_tq'][selp], C_PASSIVE, lo=-4.5, hi=-2.5)),
             box('passive map · S1', C_S1, mapc),
             box('measured − passive', C_S1, mini_curve(X4['S1_t'][:3000], X4['S1_target'][:3000], C_TQ)),
             box('÷ SD of the angle', C_S1, sd_bars())]
        mp2 = Plot((-0.8, 0.3), (-11, 4), 1.8, 0.75, axis_color=PANEL)
        mapc2 = VGroup(mp2, VMobject(stroke_color=C_S2, stroke_width=2).set_points_as_corners([mp2.p(a, t) for a, t in maps['S2']]))
        w37 = X8['w37_t'] <= X8['w37_t'][0] + 30
        pred = VGroup(mini_curve(X8['w37_t'][w37], X8['w37_y'][w37], C_TQ, lo=-30, hi=3),
                      mini_curve(X8['w37_t'][w37], X8['w37_p'][w37], C_PRED, lo=-30, hi=3))
        sel2 = (X3['eg2_t'] >= 10) & (X3['eg2_t'] <= 30)
        C = [box('raw EMG · session 2', C_S2, mini_curve(X1['raw2_t'], X1['raw2_sol'], C_SOL)),
             box('same steps, S1 settings', C_S2, mini_curve(X3['eg2_t'][sel2], X3['eg2_agc'][sel2, 2], C_SOL)),
             box('frozen LSTM', C_S2, lstm_icon(DIM)),
             box('× SD → Nm', C_S2, pred),
             box('passive map · S2', C_S2, mapc2)]
        for b, x in zip(A, xs):
            b.move_to([x, yA, 0])
        for b, x in zip(B, xs):
            b.move_to([x, yB, 0])
        for b, x in zip(C, xs):
            b.move_to([x, yC, 0])
        lab1 = T('session 1 · trains', 16, C_S1).rotate(PI / 2).move_to([-6.6, (yA + yB) / 2, 0])
        lab2 = T('session 2 · scored', 16, C_S2).rotate(PI / 2).move_to([-6.6, yC, 0])

        def arrow(a, b, col=DIM):
            return Arrow(a, b, buff=0.06, color=col, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)

        arA = [arrow(A[i].get_right(), A[i + 1].get_left()) for i in range(4)]
        arB = [arrow(B[i].get_right(), B[i + 1].get_left()) for i in range(3)]
        arBA = arrow(B[3].get_right(), A[4].get_bottom() + DOWN * 0.02 + LEFT * 0.4)
        arC = [arrow(C[0].get_right(), C[1].get_left()), arrow(C[1].get_right(), C[2].get_left()),
               arrow(C[2].get_right(), C[3].get_left())]
        ymid = (B[0].get_bottom()[1] + C[0].get_top()[1]) / 2
        x4, x2 = A[4].get_center()[0] + 0.55, C[2].get_center()[0]
        path = VMobject(stroke_color=DIM, stroke_width=2).set_points_as_corners(
            [A[4].get_bottom() + RIGHT * 0.55, [x4, ymid, 0], [x2, ymid, 0], C[2].get_top()])
        frozen = DashedVMobject(path, num_dashes=45)
        froz_lab = T('weights frozen', 13, DIM).move_to([(x4 + x2) / 2, ymid + 0.16, 0])
        score = VGroup(RoundedRectangle(corner_radius=0.08, width=1.9, height=0.5, stroke_color=C_HI, fill_color=PANEL,
                                        fill_opacity=1), T(f"score: R² {V['live_r2']}", 14, C_HI))
        score.move_to([(xs[3] + xs[4]) / 2, yC - 1.05, 0])
        arS = [arrow(C[3].get_bottom(), score.get_left() + UP * 0.05), arrow(C[4].get_bottom(), score.get_right() + UP * 0.05)]

        with self.beat('c03_b01') as d:
            self.play(FadeIn(lab1), run_time=0.3)
            step = max(0.3, (d - 1.5) / 11)
            for i in range(4):
                self.play(FadeIn(A[i], shift=RIGHT * 0.1), GrowArrow(arA[i]), run_time=step)
            for i in range(4):
                self.play(FadeIn(B[i], shift=RIGHT * 0.1), *([GrowArrow(arB[i])] if i < 3 else []), run_time=step)
            self.play(FadeIn(A[4]), GrowArrow(arBA), run_time=step)
        with self.beat('c03_b02') as d:
            self.play(FadeIn(lab2), Create(frozen), FadeIn(froz_lab), run_time=0.8)
            step = max(0.3, (d - 2.0) / 6)
            for i in range(3):
                self.play(FadeIn(C[i], shift=RIGHT * 0.1), GrowArrow(arC[i]), run_time=step)
            self.play(FadeIn(C[3]), run_time=step)
            self.play(FadeIn(C[4]), run_time=step)
            self.play(FadeIn(score), GrowArrow(arS[0]), GrowArrow(arS[1]), run_time=step)
        tree = VGroup(*A, *B, *C, lab1, lab2, *arA, *arB, arBA, *arC, frozen, froz_lab, score, *arS)
        with self.beat('c03_b03') as d:
            self.play(tree.animate.scale(0.62).move_to([0, 1.25, 0]), run_time=0.9)
            code = code_at_bottom(CodePanel('run_exp'))
            self.show_code(code)
            m = self.mark(code, 117, 118)
            self.wait(max(0.1, d * 0.25 - 1.0))
            self.unmark(code, m)
            self.mark(code, 122, 123)
            self.wait(max(0.1, d * 0.2 - 0.7))
            code2 = code_at_bottom(CodePanel('passive_fit_noop'))
            self.play(FadeOut(code), FadeIn(code2), run_time=0.6)
            self.mark(code2, 144, 145)
            note = T('static passive model: fit() returns at once · rest detection unused by D01', 16, C_HI)
            note.next_to(code2, UP, buff=0.15)
            self.play(FadeIn(note), run_time=0.5)
            foot = T('EG and YES: runs stored as C01, identical to D01 (same passive tables)', 15, DIM).next_to(note, UP, buff=0.1)
            self.play(FadeIn(foot), run_time=0.5)
        self.play(FadeOut(VGroup(tree, code2, note, foot, tag)), run_time=0.6)
