from manim import *

from style import (ANGLE_C, C_ANGLE, C_BAD, C_HI, C_S1, C_S2, C_TQ, DIM, FAINT, INK, MUSCLE_C, PANEL, V, BeatScene,
                   CodePanel, Plot, T, chapter_tag, code_at_bottom, facts, load)

FF = facts()
F = FF['x05']
X = load('x05')
NAMES = ['GM', 'GL', 'SOL', 'TA', 'angle']


class Ch07(BeatScene):
    def construct(self):
        tag = chapter_tag(7, 'Windows and splits')
        self.play(FadeIn(tag), run_time=0.4)
        Fe, tgt = X['trial_F'], X['trial_target']
        ex = F['example']
        last = int(ex['last'])
        s0, s1 = last - 160, last + 40                         # 2 s of samples around the example window
        ts = 1.0 + np.arange(s0, s1) / 100.0
        rows = VGroup()
        for k in range(5):
            y = Fe[s0:s1, k]
            lo, hi = float(y.min()), float(y.max())
            pad = 0.1 * (hi - lo + 1e-6)
            pl = Plot((ts[0], ts[-1]), (lo - pad, hi + pad), 5.6, 0.48, axis_color=PANEL).move_to([-3.55, 2.75 - k * 0.56, 0])
            rows.add(VGroup(pl, pl.curve(ts, y, MUSCLE_C[k] if k < 4 else C_ANGLE, 1.8),
                            T(NAMES[k], 14, MUSCLE_C[k] if k < 4 else C_ANGLE).next_to(pl, LEFT, buff=0.12)))
        ty = tgt[s0 - 19:s1 - 19]
        PT = Plot((ts[0], ts[-1]), (float(ty.min()) - 1, float(ty.max()) + 1), 5.6, 0.9, axis_color=FAINT).move_to([-3.55, -0.15, 0])
        ctg = PT.curve(ts, ty, C_TQ, 1.8)
        lt = T('target', 14, C_TQ).next_to(PT, LEFT, buff=0.12)
        x_of = lambda j: rows[0][0].p(1.0 + j / 100.0, 0)[0]

        def frame(j_last):
            x0, x1 = x_of(j_last - 19) - 0.02, x_of(j_last) + 0.02
            top, bot = rows[0][0].get_top()[1] + 0.05, rows[4][0].get_bottom()[1] - 0.05
            box = Rectangle(width=x1 - x0, height=top - bot, stroke_color=C_HI, stroke_width=2.5).move_to([(x0 + x1) / 2, (top + bot) / 2, 0])
            dot = Dot(PT.p(1.0 + j_last / 100.0, float(tgt[j_last - 19])), radius=0.07, color=C_HI)
            return VGroup(box, dot)

        # 7.1 the sliding window
        with self.beat('c07_b01') as d:
            code = code_at_bottom(CodePanel('windows'))
            self.play(FadeIn(VGroup(*[r[0] for r in rows], *[r[2] for r in rows], PT, lt)), run_time=0.5)
            self.play(*[Create(r[1]) for r in rows], Create(ctg), run_time=1.5)
            self.show_code(code)
            fr = frame(last - 60)
            self.play(Create(fr), run_time=0.5)
            self.mark(code, 39, 40)
            j = ValueTracker(last - 60)
            fr.add_updater(lambda m: m.become(frame(int(round(j.get_value())))))
            self.play(j.animate.set_value(last), run_time=min(4.0, d * 0.25), rate_func=linear)
            fr.clear_updaters()
            # the window's 100 real numbers
            W = X['window']
            grid = VGroup()
            for k in range(5):
                for i in range(20):
                    v = W[i, k]
                    if k < 4:
                        col = interpolate_color(ManimColor(PANEL), ManimColor(MUSCLE_C[k]), float(np.clip(v / 1.2, 0, 1)))
                        s = f'{v:.2f}'
                    else:
                        col = ManimColor('#30363D')
                        s = f'{v:.2f}'.replace('-', '−')
                    cell = Rectangle(width=0.3, height=0.44, stroke_width=0.5, stroke_color=PANEL, fill_color=col, fill_opacity=1)
                    cell.move_to([0.75 + i * 0.3, 2.8 - k * 0.47, 0])
                    grid.add(VGroup(cell, T(s, 9 if k < 4 else 7.5, INK, font='Consolas').move_to(cell)))
            gl = T(f"one window: 20 samples × 5 inputs → target {ex['y']:.2f} Nm".replace('-', '−'), 15, C_HI)
            gl.next_to(grid, DOWN, buff=0.18)
            tlab = T('0.2 s →', 13, DIM).next_to(grid, UP, buff=0.06).align_to(grid, RIGHT)
            self.play(TransformFromCopy(fr[0], grid), run_time=1.0)
            self.play(FadeIn(gl), FadeIn(tlab), run_time=0.5)
        # 7.2 how many windows
        with self.beat('c07_b02') as d:
            eq = T(f"{V['n100']} − 20 + 1 = {V['n_windows']} windows per trial", 30, INK)
            eq.scale_to_fit_width(min(eq.width, 5.6)).move_to([3.6, 0.0, 0])
            self.play(FadeIn(eq), run_time=0.6)
        self.play(FadeOut(VGroup(rows, PT, ctg, lt, fr, grid, gl, tlab, eq, code)), run_time=0.5)

        # 7.3 the split inside each session-1 trial
        tr = F['trial']
        n, ntr, nva = tr['n_windows'], tr['n_train'], tr['n_val']
        x0, x1 = -6.2, 6.2
        sx = lambda w: x0 + (x1 - x0) * w / n
        bar_tr = Rectangle(width=sx(ntr) - sx(0), height=0.55, stroke_width=0, fill_color=C_S1, fill_opacity=0.9).move_to(
            [(sx(0) + sx(ntr)) / 2, 2.2, 0])
        bar_va = Rectangle(width=sx(ntr + nva) - sx(ntr), height=0.55, stroke_color=C_S1, stroke_width=1.5, fill_color=C_S1,
                           fill_opacity=0.3).move_to([(sx(ntr) + sx(ntr + nva)) / 2, 2.2, 0])
        drop = Line([sx(n) - 0.015, 1.9, 0], [sx(n) - 0.015, 2.5, 0], color=C_BAD, stroke_width=4)
        l_tr = T(f"training: first 80 % = {V['n_train']} windows", 17, C_S1).next_to(bar_tr, UP, buff=0.12)
        l_va = T(f"validation: {V['n_val']}", 17, C_S1).next_to(bar_va, UP, buff=0.12)
        l_dr = T('last window dropped', 13, C_BAD).next_to(drop, DOWN, buff=0.08).shift(LEFT * 0.6)
        l_t = T(f"EG · one session-1 trial · {V['n_windows']} windows", 16, DIM).next_to(bar_tr, DOWN, buff=0.45).align_to(bar_tr, LEFT)
        with self.beat('c07_b03') as d:
            code = code_at_bottom(CodePanel('split'))
            self.play(FadeIn(l_t), GrowFromEdge(bar_tr, LEFT), FadeIn(l_tr), run_time=1.0)
            self.show_code(code)
            self.mark(code, 68, 69)
            self.play(GrowFromEdge(bar_va, LEFT), FadeIn(l_va), run_time=0.8)
            self.mark(code, 70, color=C_S2)
            self.wait(max(0.1, d * 0.55 - 3.4))
            code2 = code_at_bottom(CodePanel('drop_last'))
            self.play(FadeOut(code), FadeIn(code2), Create(drop), FadeIn(l_dr), run_time=0.7)
            self.mark(code2, 70)
        # 7.4 no gap: the boundary, zoomed
        with self.beat('c07_b04') as d:
            self.play(FadeOut(code2), run_time=0.3)
            b0 = ntr - 6                                     # sample index ~ window index here (window w starts at sample w)
            cells = VGroup()
            for s in range(b0, b0 + 48):
                cells.add(Square(0.22, stroke_width=0.8, stroke_color=PANEL, fill_color=FAINT, fill_opacity=1))
            cells.arrange(RIGHT, buff=0.02).move_to([0, 0.75, 0])
            lab = T('samples of the trial (100 Hz)', 14, DIM).next_to(cells, DOWN, buff=0.12)
            sl = lambda s: cells[s - b0]
            w_tr = Rectangle(width=sl(ntr - 1 + 19).get_right()[0] - sl(ntr - 1).get_left()[0], height=0.16, stroke_width=0,
                             fill_color=C_S1, fill_opacity=0.95).next_to(cells, UP, buff=0.1)
            w_tr.align_to(sl(ntr - 1), LEFT)
            w_va = Rectangle(width=sl(ntr + 19).get_right()[0] - sl(ntr).get_left()[0], height=0.16, stroke_color=C_S1,
                             stroke_width=1.5, fill_color=C_S1, fill_opacity=0.35).next_to(w_tr, UP, buff=0.08).align_to(sl(ntr), LEFT)
            t_tr = T('last training window', 13, C_S1).next_to(w_tr, LEFT, buff=0.1)
            t_va = T('first validation window', 13, C_S1).next_to(w_va, LEFT, buff=0.1)
            over = VGroup(*[sl(s).copy().set_fill(C_HI, 0.85) for s in range(ntr, ntr + 19)])
            o_l = T('19 input samples shared · targets differ', 15, C_HI).next_to(w_va, UP, buff=0.12)
            bx = sx(ntr)
            zoom = VGroup(Line([bx - 0.05, 1.92, 0], cells.get_corner(UL) + UP * 0.5, color=FAINT),
                          Line([bx + 0.05, 1.92, 0], cells.get_corner(UR) + UP * 0.5, color=FAINT))
            self.play(FadeOut(VGroup(l_va, drop, l_dr, l_t)), Create(zoom), FadeIn(cells), run_time=0.8)
            self.play(FadeIn(lab), GrowFromEdge(w_tr, LEFT), FadeIn(t_tr), run_time=0.7)
            self.play(GrowFromEdge(w_va, LEFT), FadeIn(t_va), run_time=0.7)
            self.play(FadeIn(over), FadeIn(o_l), run_time=0.8)
            self.wait(max(0.1, d * 0.6 - 3.4))
            lw = VGroup(T('last week (pipeline.py):', 16, DIM),
                        T('gap_s = 1.0 between the blocks, each block filtered on its own and trimmed 1 s at both ends', 15, INK))
            lw.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([0, -0.55, 0])
            code = code_at_bottom(CodePanel('lastweek_blocks'))
            self.play(FadeIn(lw), FadeIn(code), run_time=0.7)
            self.mark(code, 82, 84)
        self.play(FadeOut(Group(*[m for m in self.mobjects if m is not tag and isinstance(m, Mobject) and not isinstance(m, ValueTracker)])), run_time=0.5)

        # 7.5 the arrays
        with self.beat('c07_b05') as d:
            arrs = VGroup()
            for name, nwin, col, op in (('X_train', V['x_train'], C_S1, 0.9), ('X_val', V['x_val'], C_S1, 0.35),
                                        ('X_test', V['x_test'], C_S2, 0.9)):
                box = RoundedRectangle(corner_radius=0.1, width=3.9, height=1.5, stroke_color=col, fill_color=col,
                                       fill_opacity=op * 0.25)
                txt = VGroup(T(name, 24, col, font='Consolas'), T(f'({nwin}, 20, 5)', 22, INK, font='Consolas'))
                arrs.add(VGroup(box, txt.arrange(DOWN, buff=0.12).move_to(box)))
            arrs.arrange(RIGHT, buff=0.45).move_to([0, 1.3, 0])
            subs = VGroup(T('8 session-1 trials × 7,112', 15, DIM), T('8 session-1 trials × 1,778', 15, DIM),
                          T('8 session-2 trials × 8,891 (every window)', 15, DIM))
            for s, a in zip(subs, arrs):
                s.next_to(a, DOWN, buff=0.15)
            self.play(LaggedStart(*[FadeIn(a, shift=UP * 0.15) for a in arrs], lag_ratio=0.3), run_time=1.5)
            self.play(FadeIn(subs), run_time=0.6)
        # 7.6 what shuffle does
        with self.beat('c07_b06') as d:
            self.play(arrs.animate.scale(0.6).move_to([0, 2.65, 0]), FadeOut(subs), run_time=0.6)
            code = code_at_bottom(CodePanel('fit_call'))
            self.show_code(code)
            self.mark(code, 181)
            B = FF.get('x07', {}).get('batches', {}).get('knot', [[i % 8 for i in range(8)]] * 3)
            cards = VGroup()
            for b in range(3):
                for i in range(8):
                    k = B[b][i]
                    cards.add(RoundedRectangle(corner_radius=0.05, width=0.42, height=0.6, stroke_width=1,
                                               stroke_color=PANEL, fill_color=ANGLE_C[k], fill_opacity=0.95))
            order = sorted(range(24), key=lambda q: (B[q // 8][q % 8], q))
            for pos, q in enumerate(order):
                cards[q].move_to([-5.6 + pos * 0.48, 1.1, 0])
            lab0 = T('training windows, ordered by trial (angle colours)', 15, DIM).move_to([0, 0.55, 0])
            self.play(FadeIn(cards), FadeIn(lab0), run_time=0.6)
            targets = []
            for q in range(24):
                b, i = divmod(q, 8)
                targets.append(np.array([-4.6 + b * 3.6 + i * 0.42, 0.0, 0]))
            self.play(*[cards[q].animate.move_to(targets[q]) for q in range(24)], run_time=1.6)
            bl = VGroup(*[T(f'batch {b + 1}', 15, INK).move_to([-4.6 + b * 3.6 + 3.5 * 0.42, -0.5, 0]) for b in range(3)])
            lab1 = T('each epoch: a new random order, then batches of 8 (the first 3 batches of epoch 1, real angles)', 15, INK)
            lab1.move_to([0, 0.55, 0])
            self.play(FadeIn(bl), ReplacementTransform(lab0, lab1), run_time=0.6)
            ang = VGroup(*[VGroup(Square(0.18, fill_color=c, fill_opacity=1, stroke_width=0), T(f'{a:+.2f}'.replace('-', '−'), 11, DIM)).arrange(RIGHT, buff=0.05)
                           for c, a in zip(ANGLE_C, F['knots'])]).arrange(RIGHT, buff=0.22).move_to([0, 2.0, 0])
            self.play(FadeIn(ang), run_time=0.5)
            self.wait(max(0.1, d * 0.5 - 4.0))
            code2 = code_at_bottom(CodePanel('shuffle_tf'))
            self.play(FadeOut(code), FadeIn(code2), run_time=0.5)
            self.mark(code2, 133, 134)
            note = T('validation: never shuffled, never trained on · session 2: not used at all', 15, C_HI).move_to([0, 1.55, 0])
            self.play(FadeIn(note), run_time=0.5)
        self.play(FadeOut(VGroup(arrs, cards, bl, lab1, ang, code2, note, tag)), run_time=0.6)
