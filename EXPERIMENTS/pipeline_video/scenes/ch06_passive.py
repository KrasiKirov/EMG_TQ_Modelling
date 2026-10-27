from manim import *

from style import (C_ANGLE, C_BAD, C_GOOD, C_HI, C_PASSIVE, C_S1, C_S2, C_TQ, DIM, FAINT, INK, PANEL, V, BeatScene,
                   CodePanel, Plot, T, chapter_tag, code_at_bottom, facts, load)

F = facts()['x04']
X = load('x04')


class Ch06(BeatScene):
    def construct(self):
        tag = chapter_tag(6, 'Passive map and target')
        self.play(FadeIn(tag), run_time=0.4)
        t, pos, tq = X['p20_t'], X['p20_pos'], X['p20_tq']
        PA = Plot((0, 90.1), (-0.08, -0.02), 8.0, 0.9, yticks=(-0.07, -0.03), tick_size=13, ylabel='rad',
                  yfmt=lambda v: f'{v:.2f}').move_to([-2.3, 2.55, 0])
        PT = Plot((0, 90.1), (-80, 45), 8.0, 2.4, xticks=range(0, 91, 10), yticks=(-60, -30, 0, 30), tick_size=13,
                  xlabel='time in passive trial (s)', ylabel='torque (Nm)', grid=True).move_to([-2.3, 0.45, 0])
        ttl = T('EG · passive trial before p7 · session 1', 18, C_S1).next_to(PA, UP, buff=0.05).align_to(PA, LEFT)
        ca = PA.curve(t, pos, C_ANGLE, 2.0, 2000)
        ct = PT.curve(t, np.clip(tq, -80, 45), C_TQ, 1.2, 3000)

        # 6.1 a passive trial
        with self.beat('c06_b01') as d:
            self.play(FadeIn(PA), FadeIn(PT), FadeIn(ttl), run_time=0.5)
            self.play(Create(ca), Create(ct), run_time=min(4.0, d * 0.4))
            p1 = F['p20_pieces'][0]
            hold = Rectangle(width=PT.p(p1['seconds'], 0)[0] - PT.p(0, 0)[0], height=3.6, stroke_color=C_GOOD,
                             stroke_width=2, fill_opacity=0.06, fill_color=C_GOOD)
            hold.move_to([(PT.p(0, 0)[0] + PT.p(p1['seconds'], 0)[0]) / 2, (PA.get_top()[1] + PT.p(0, -80)[1]) / 2 + 0.05, 0])
            hl = VGroup(T('still hold', 17, C_GOOD), T(f"{V['hold_s']} s, torque SD {p1['tq_sd']:.2f} Nm", 15, C_GOOD)).arrange(DOWN, buff=0.06).move_to([4.6, 2.3, 0])
            self.play(Create(hold), FadeIn(hl), run_time=0.8)
            sw = VGroup(T('then a step to another angle;', 15, DIM), T('the torque swings by tens of Nm', 15, DIM)).arrange(DOWN, buff=0.06).move_to([4.6, 1.1, 0])
            self.play(FadeIn(sw), run_time=0.6)
        # 6.2 the stillness test
        with self.beat('c06_b02') as d:
            code = code_at_bottom(CodePanel('still_test'))
            self.play(FadeOut(sw), FadeOut(hl), run_time=0.3)
            self.show_code(code)
            pcs = VGroup()
            for p in F['p20_pieces']:
                x0, x1 = p['start'] / 1000.0, p['stop'] / 1000.0
                col = C_GOOD if p['keep'] else C_BAD
                br = Line(PT.p(x0, -72), PT.p(x1, -72), color=col, stroke_width=6)
                lab = VGroup(T(f"{'✓ kept' if p['keep'] else '✗ dropped'} · {x0:.1f}–{x1:.1f} s", 16, col, weight='BOLD'),
                             T(f"{p['seconds']:.1f} s long", 14, col), T(f"angle SD {p['angle_sd_mrad']:.2f} mrad", 14, col),
                             T(f"torque SD {p['tq_sd']:.2f} Nm", 14, col)).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
                lab.move_to([4.6, 2.3 if p['keep'] else 0.75, 0])
                pcs.add(VGroup(br, lab))
            skip = T('1 s skipped after the step', 12, DIM).next_to(PT.p(21, -72), UP, buff=0.08)
            self.mark(code, 69)
            self.play(LaggedStart(*[FadeIn(p) for p in pcs], lag_ratio=0.4), FadeIn(skip), run_time=1.5)
            self.wait(max(0.1, d * 0.6 - 3.0))
            nb = T('no EMG check: "relaxed" = a still piece of a trial labelled passive', 17, C_HI).move_to([2.7, 3.62, 0])
            self.play(FadeOut(ttl), FadeIn(nb), run_time=0.6)
        self.play(FadeOut(VGroup(PA, PT, ca, ct, hold, pcs, skip, nb, code)), run_time=0.5)

        # 6.3 the map
        with self.beat('c06_b03') as d:
            maps = F['maps']['EG']['static']
            M = Plot((-0.8, 0.3), (-11.5, 4), 7.0, 2.85, xticks=(-0.7, -0.5, -0.3, -0.1, 0.1, 0.3), yticks=(-10, -5, 0),
                     xlabel='ankle angle (rad)', ylabel='passive torque (Nm)', tick_size=14, grid=True).move_to([-2.2, 1.35, 0])
            code = code_at_bottom(CodePanel('map_merge'))
            kept = [p for p in F['eg_pieces'] if p['keep']]
            dots = VGroup()
            for p in kept:
                col = C_S1 if p['session'] == 'test' else C_S2
                dots.add(Dot(M.p(p['angle'], p['tq_mean']), radius=0.07, color=col))
            lines = VGroup(*[VMobject(stroke_color=c, stroke_width=2.2).set_points_as_corners([M.p(a, b) for a, b in maps[s]])
                             for s, c in (('S1', C_S1), ('S2', C_S2))])
            cnt = T(f"{V['eg_kept']} of {V['eg_pieces']} pieces kept: the first {V['hold_s']} s of every passive trial", 16, INK)
            cnt.scale_to_fit_width(4.9).move_to([4.45, 2.3, 0])
            n1 = VGroup(T('one point = mean angle, mean torque of a hold', 15, DIM),
                        T('holds within 0.05 rad are averaged', 15, DIM),
                        T('not an average over every moment at that angle', 15, C_HI),
                        T('active trials contribute nothing', 15, C_HI),
                        T('session 2 has its own map', 15, C_S2)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            n1.next_to(cnt, DOWN, buff=0.35).align_to(cnt, LEFT)
            self.play(FadeIn(M), FadeIn(cnt), run_time=0.6)
            self.show_code(code)
            self.play(LaggedStart(*[GrowFromCenter(x) for x in dots], lag_ratio=0.08), run_time=1.6)
            self.mark(code, 75, 77)
            self.play(Create(lines), FadeIn(n1[:2]), run_time=1.2)
            self.wait(max(0.1, d * 0.55 - 4.5))
            self.play(FadeIn(n1[2:]), run_time=0.8)
        self.play(FadeOut(VGroup(M, dots, lines, cnt, n1, code)), run_time=0.5)

        # 6.4 three exceptions
        with self.beat('c06_b04') as d:
            jl = dict(F['maps']['JM']['legacy']['S1'])
            bad = [a for a, v in F['maps']['JM']['legacy']['S1'] if not any(abs(a - b) < 1e-9 for b, _ in F['maps']['JM']['static']['S1'])]
            cards = VGroup()
            for title, lines, col in (
                    ('YES', ['map = hand-entered table', 'in ML/config.py, both rules'], C_S1),
                    ('IES', ['no passive trials', 'empty map → nothing subtracted', 'target = measured torque'], C_HI),
                    ('JM', [f"old rule kept {jl[bad[0]]:+.1f} Nm at {bad[0]:+.2f} rad".replace('-', '−'),
                            'from a movement recording;', 'the stillness test drops it'], C_S2)):
                box = RoundedRectangle(corner_radius=0.12, width=4.0, height=2.0, stroke_color=col, fill_color=PANEL,
                                       fill_opacity=1)
                txt = VGroup(T(title, 28, col, weight='BOLD'), *[T(l, 16, INK) for l in lines]).arrange(DOWN, buff=0.12)
                cards.add(VGroup(box, txt.move_to(box)))
            cards.arrange(RIGHT, buff=0.35).move_to([0, 1.2, 0])
            code = code_at_bottom(CodePanel('empty_map'))
            self.play(FadeIn(cards[0], shift=UP * 0.1), run_time=0.6)
            c1 = code_at_bottom(CodePanel('yes_table'))
            self.show_code(c1)
            self.wait(max(0.1, d * 0.25 - 1.2))
            self.play(FadeIn(cards[1], shift=UP * 0.1), FadeOut(c1), FadeIn(code), run_time=0.6)
            self.mark(code, 47, 48)
            self.wait(max(0.1, d * 0.3 - 1.0))
            self.play(FadeIn(cards[2], shift=UP * 0.1), run_time=0.6)
        self.play(FadeOut(VGroup(cards, code)), run_time=0.5)

        # 6.5 the target
        with self.beat('c06_b05') as d:
            code = code_at_bottom(CodePanel('targets'))
            groups = VGroup()
            for k, (s, col) in enumerate((('S1', C_S1), ('S2', C_S2))):
                tt, y, P, tgt = X[f'{s}_t'], X[f'{s}_tq'], X[f'{s}_P'], X[f'{s}_target']
                m = tt <= 31
                pl = Plot((1, 31), (-33, 6), 5.6, 2.9, xticks=(1, 11, 21, 31), yticks=(-30, -20, -10, 0), tick_size=13,
                          xlabel='time in trial (s)', ylabel='Nm' if k == 0 else '', grid=True).move_to([-3.4 + 6.9 * k, 1.05, 0])
                cm = pl.curve(tt[m], y[m], '#6E7681', 1.2)
                cp = DashedVMobject(pl.curve(tt[m], P[m], '#B0B8C1', 3.0), num_dashes=60)
                cg = pl.curve(tt[m], tgt[m], C_TQ, 2.0)
                lab = T(f"session {k + 1}: passive ≈ {V['p_s1'] if k == 0 else V['p_s2']} Nm", 17, col).next_to(pl, UP, buff=0.05)
                groups.add(VGroup(pl, cm, cp, cg, lab))
            key = VGroup(T('measured', 15, '#6E7681'), T('- - passive map at the sample angle', 15, '#B0B8C1'),
                         T('target = measured − passive', 15, C_TQ)).arrange(RIGHT, buff=0.5).move_to([0, -0.95, 0])
            self.play(FadeIn(groups[0][0]), FadeIn(groups[0][4]), Create(groups[0][1]), run_time=1.0)
            self.show_code(code)
            self.play(Create(groups[0][2]), FadeIn(key), run_time=0.8)
            self.mark(code, 71)
            self.play(TransformFromCopy(groups[0][1], groups[0][3]), run_time=1.0)
            self.wait(max(0.1, d * 0.6 - 4.0))
            code2 = code_at_bottom(CodePanel('map_by_session', 100, 102))
            self.play(FadeIn(groups[1]), FadeOut(code), FadeIn(code2), run_time=1.0)
            self.mark(code2, 102)
        self.play(FadeOut(VGroup(groups, key, code2, tag)), run_time=0.6)
