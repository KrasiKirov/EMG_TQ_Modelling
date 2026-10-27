from manim import *

from style import (BG, C_ANGLE, C_HI, C_S1, C_S2, C_TQ, DIM, FAINT, INK, MUSCLE_C, PANEL, V, BeatScene, CodePanel,
                   Plot, T, chapter_tag, code_at_bottom, envelope_curve, facts, load)

F = facts()['x01']
D = load('x01')


class Ch01(BeatScene):
    def construct(self):
        # 1.0 title
        with self.beat('c01_b00') as d:
            title = T('EMG → ankle torque', 60, INK, weight='BOLD').shift(UP * 0.9)
            sub = T('what the code actually does', 32, DIM).next_to(title, DOWN, buff=0.3)
            badge = VGroup(RoundedRectangle(corner_radius=0.12, width=6.6, height=0.62, stroke_color='#2EC4B6',
                                            fill_color=PANEL, fill_opacity=1),
                           T('D01 · “Balanced positions” in the dashboard', 22, '#2EC4B6'))
            badge.next_to(sub, DOWN, buff=0.55)
            self.play(FadeIn(title, shift=UP * 0.2), run_time=1.0)
            self.play(FadeIn(sub), FadeIn(badge), run_time=1.0)
            self.wait(max(0.1, d - 3.4))
            self.play(FadeOut(VGroup(title, sub, badge)), run_time=0.8)
        tag = chapter_tag(1, 'The data')
        self.play(FadeIn(tag), run_time=0.4)

        # 1.1 six people
        with self.beat('c01_b01') as d:
            codes = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']
            cards = VGroup()
            for c in codes:
                box = RoundedRectangle(corner_radius=0.14, width=1.7, height=1.15, stroke_color=FAINT, fill_color=PANEL,
                                       fill_opacity=1)
                lab = T(c, 34, INK, weight='BOLD').move_to(box.get_center() + UP * 0.12)
                f = T('1 recording file', 13, DIM).next_to(lab, DOWN, buff=0.12)
                cards.add(VGroup(box, lab, f))
            cards.arrange(RIGHT, buff=0.35).move_to([0, 1.3, 0])
            code = code_at_bottom(CodePanel('data_load'))
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.12), run_time=1.6)
            self.show_code(code)
            self.wait(max(0.1, d * 0.55 - 2.2))
            eg = cards[1]
            self.play(eg[0].animate.set_stroke(C_HI, 3), eg[1].animate.set_color(C_HI), run_time=0.6)
        self.play(FadeOut(cards), FadeOut(code), run_time=0.5)

        # 1.2 the 40 trials of EG's file
        with self.beat('c01_b02') as d:
            inv = F['eg_trials']
            w, gapx = 0.29, 0.04
            blocks = VGroup()
            for t in inv:
                col = '#6E7681' if t['kind'] == 'mvc' else (C_S1 if t['session'] == 'test' else C_S2)
                h = 0.5 + 0.9 * (t['seconds'] / 90.1)
                r = Rectangle(width=w, height=h, stroke_color=col, stroke_width=2,
                              fill_color=col, fill_opacity=0.85 if t['kind'] in ('active', 'mvc') else 0.0)
                blocks.add(r)
            blocks.arrange(RIGHT, buff=gapx, aligned_edge=DOWN).move_to([0, 0.9, 0])
            lab_mvc = T(f"{V['eg_n_mvc']} MVC × {V['eg_mvc_s']} s", 18, DIM).next_to(blocks[0:8], UP, buff=0.25)
            s1 = VGroup(*[b for b, t in zip(blocks, inv) if t['session'] == 'test'])
            s2 = VGroup(*[b for b, t in zip(blocks, inv) if t['session'] == 'retest'])
            lab_s1 = T('session 1 · “test”', 20, C_S1).next_to(s1, UP, buff=0.25)
            lab_s2 = T('session 2 · “retest”', 20, C_S2).next_to(s2, UP, buff=0.25)
            plabs = VGroup()
            for b, t in zip(blocks, inv):
                if t['kind'] == 'active':
                    plabs.add(T(t['pos'], 12, DIM).next_to(b, DOWN, buff=0.08))
            legend = VGroup(
                VGroup(Rectangle(width=0.25, height=0.25, stroke_color=INK, stroke_width=2), T('passive trial', 16, DIM)).arrange(RIGHT, buff=0.12),
                VGroup(Rectangle(width=0.25, height=0.25, stroke_color=INK, fill_color=INK, fill_opacity=0.85),
                       T('active trial (90.1 s)', 16, DIM)).arrange(RIGHT, buff=0.12),
                VGroup(Rectangle(width=0.25, height=0.25, stroke_color='#6E7681', fill_color='#6E7681', fill_opacity=0.85),
                       T('maximal contraction (unused)', 16, DIM)).arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.6)
            legend.next_to(blocks, DOWN, buff=0.55)
            code = code_at_bottom(CodePanel('parse_comment'))
            self.play(LaggedStart(*[FadeIn(b) for b in blocks[:8]], lag_ratio=0.05), FadeIn(lab_mvc), run_time=1.2)
            self.play(LaggedStart(*[FadeIn(b) for b in s1], lag_ratio=0.04), FadeIn(lab_s1), run_time=1.6)
            self.show_code(code)
            self.play(FadeIn(legend), FadeIn(plabs[:8]), run_time=0.8)
            m1 = self.mark(code, 88, 93)
            self.wait(max(0.1, d * 0.6 - 4.6))
            self.unmark(code, m1)
            self.play(LaggedStart(*[FadeIn(b) for b in s2], lag_ratio=0.04), FadeIn(lab_s2), FadeIn(plabs[8:]),
                      run_time=1.6)
            self.mark(code, 95, 100)
        timeline = VGroup(blocks, lab_mvc, lab_s1, lab_s2, plabs, legend)
        self.play(FadeOut(timeline), FadeOut(code), run_time=0.5)

        # 1.3 six channels, raw 1 kHz
        with self.beat('c01_b03') as d:
            t = D['raw_t']
            t0, t1 = float(t[0]), float(t[-1])
            rows = [('raw_gm', 'GM', MUSCLE_C[0]), ('raw_gl', 'GL', MUSCLE_C[1]), ('raw_sol', 'SOL', MUSCLE_C[2]),
                    ('raw_ta', 'TA', MUSCLE_C[3]), ('raw_position', 'angle (rad)', C_ANGLE), ('raw_torque', 'torque (Nm)', C_TQ)]
            group = VGroup()
            for k, (key, name, col) in enumerate(rows):
                y = D[key].astype(float)
                lo, hi = float(np.min(y)), float(np.max(y))
                pad = 0.08 * (hi - lo + 1e-9)
                pl = Plot((t0, t1), (lo - pad, hi + pad), 9.6, 0.62, axis_color=BG)
                pl.move_to([0.1, 2.75 - k * 0.74, 0])
                c = envelope_curve(pl, t, y, col, 1.4, 1600)
                lab = T(name, 18, col).next_to(pl, LEFT, buff=0.25)
                fmt = (lambda v: f'{v:+.3f}') if k < 5 else (lambda v: f'{v:+.1f}')
                rng = T(f'{fmt(lo)} … {fmt(hi)}'.replace('-', '−'), 13, DIM).next_to(pl, RIGHT, buff=0.18)
                group.add(VGroup(pl, c, lab, rng))
            axis = Line([-4.7, -1.65, 0], [4.9, -1.65, 0], stroke_color=FAINT)
            tl = VGroup(*[T(f'{v:.1f} s', 14, DIM).move_to([-4.7 + (v - t0) / (t1 - t0) * 9.6, -1.85, 0])
                          for v in np.arange(np.ceil(t0 * 2) / 2, t1, 0.5)])
            note = T('1000 samples per second · EMG in recorded units · range on the right', 16, DIM).move_to([0.1, -2.25, 0])
            code = CodePanel('canonical_order', max_font=18).move_to([0, -3.2, 0])
            for g in group:
                self.play(Create(g[1]), FadeIn(g[2]), FadeIn(g[3]), run_time=min(0.9, d / 9))
            self.play(Create(axis), FadeIn(tl), FadeIn(note), FadeIn(code), run_time=0.8)
        self.play(FadeOut(VGroup(group, axis, tl, note, code)), run_time=0.5)

        # 1.4 one whole active trial
        with self.beat('c01_b04') as d:
            tt, tq, pos = D['trial_t'], D['trial_tq'], D['trial_pos']
            pa = Plot((0, 91), (-0.045, -0.031), 11.6, 0.8, yticks=(-0.04, -0.035), ylabel='angle (rad)', tick_size=14,
                      yfmt=lambda v: f'{v:.3f}').move_to([0.6, 2.35, 0])
            pt = Plot((0, 91), (-36, 4), 11.6, 3.3, xticks=range(0, 91, 10), yticks=(-30, -20, -10, 0),
                      xlabel='time in trial (s)', ylabel='torque (Nm)', tick_size=14, grid=True).move_to([0.6, -0.55, 0])
            ca = pa.curve(tt, pos, C_ANGLE, 1.6, 1500)
            ct = pt.curve(tt, tq, C_TQ, 1.8, 2000)
            ttl = T(f"EG · session 1 · {V['eg_angle']} rad", 22, C_S1).move_to([0.6, 3.2, 0])
            sdl = T(f"angle SD {V['eg_angle_sd']} mrad", 18, C_ANGLE).next_to(pa, RIGHT, buff=-2.6).shift(UP * 0.55)
            self.play(FadeIn(ttl), FadeIn(pa), FadeIn(pt), run_time=0.8)
            self.play(Create(ca), FadeIn(sdl), run_time=1.0)
            self.play(Create(ct), run_time=min(4.0, d * 0.35))
            ticks = VGroup()
            for k, (a, b) in enumerate(D['push_runs']):
                x = float(tt[a])
                p = pt.p(x, 2.2)
                ticks.add(Triangle(color=C_HI, fill_opacity=1).scale(0.07).rotate(PI).move_to(p))
            cnt = T(f"{V['eg_pushes']} pushes", 20, C_HI).next_to(pt.p(91, 4), UP, buff=0.12).shift(LEFT * 0.6)
            self.play(LaggedStart(*[FadeIn(x) for x in ticks], lag_ratio=0.08), FadeIn(cnt), run_time=1.6)
            arr = VGroup()
            why = T('pushing the toes down → negative torque', 17, DIM).move_to([4.7, 3.62, 0])
            self.play(FadeIn(why), run_time=0.8)
        self.play(FadeOut(VGroup(ttl, pa, pt, ca, ct, sdl, ticks, cnt, arr, why)), run_time=0.5)

        # 1.5 the eight angles
        with self.beat('c01_b05') as d:
            angs = F['eg_S1_angles']
            peaks = F['eg_S1_peak_abs_tq_Nm']
            joint = np.array([-4.2, 0.6, 0])

            def foot(th, col, op=1.0, sw=4):
                shank = Polygon(joint + np.array([-0.22, 1.9, 0]), joint + np.array([0.18, 1.9, 0]),
                                joint + np.array([0.16, 0.05, 0]), joint + np.array([-0.2, 0.05, 0]),
                                stroke_color=col, stroke_width=sw * 0.5, stroke_opacity=op, fill_color=col,
                                fill_opacity=0.25 * op)
                pts = [[-0.32, 0.08], [-0.42, -0.25], [-0.3, -0.42], [1.45, -0.42], [1.55, -0.3], [1.2, -0.18],
                       [0.35, 0.02], [0.15, 0.12]]
                f = Polygon(*[joint + np.array([x, y, 0]) for x, y in pts], stroke_color=col, stroke_width=sw * 0.5,
                            stroke_opacity=op, fill_color=col, fill_opacity=(0.5 if op == 1.0 else 0.0))
                f.rotate(th, about_point=joint)
                return VGroup(shank, f, Dot(joint, radius=0.05, color=col).set_opacity(op))
            ghosts = VGroup(*[foot(a, DIM, 0.35, 2) for a in angs])
            live = foot(angs[0], INK)
            lab_dn = T('toes down', 16, DIM).move_to(joint + np.array([1.6, -1.25, 0]))
            lab_up = T('toes up', 16, DIM).move_to(joint + np.array([1.7, 0.55, 0]))
            pb = Plot((-0.5, 7.5), (0, 44), 6.6, 3.5, yticks=(0, 10, 20, 30, 40), ylabel='largest |torque| (Nm)',
                      tick_size=14, grid=True).move_to([2.9, 0.5, 0])
            bars, labs = VGroup(), VGroup()
            for k, (a, p) in enumerate(zip(angs, peaks)):
                r = Rectangle(width=0.55, height=max(0.01, pb.p(0, p)[1] - pb.p(0, 0)[1]), stroke_width=0,
                              fill_color=C_S1, fill_opacity=0.85)
                r.move_to(pb.p(k, 0), aligned_edge=DOWN)
                bars.add(r)
                labs.add(T(f'{a:+.2f}'.replace('-', '−'), 14, DIM).next_to(pb.p(k, 0), DOWN, buff=0.12))
            vals = VGroup(*[T(f'{p:.1f}', 13, INK).next_to(b, UP, buff=0.06) for b, p in zip(bars, peaks)])
            xl = T('ankle angle (rad), session 1', 15, DIM).next_to(labs, DOWN, buff=0.15)
            self.play(FadeIn(ghosts), FadeIn(live), FadeIn(lab_dn), FadeIn(lab_up), FadeIn(pb), FadeIn(labs), FadeIn(xl),
                      run_time=0.8)
            step = max(0.25, min(0.7, (d - 2.5) / 8))
            for k, a in enumerate(angs):
                new = foot(a, INK)
                self.play(Transform(live, new), GrowFromEdge(bars[k], DOWN), FadeIn(vals[k]), run_time=step)
        self.play(FadeOut(VGroup(ghosts, live, lab_dn, lab_up, pb, bars, labs, vals, xl)), run_time=0.5)

        # 1.6 two sessions
        with self.beat('c01_b06') as d:
            s1 = RoundedRectangle(corner_radius=0.15, width=5.2, height=2.0, stroke_color=C_S1, fill_color=C_S1,
                                  fill_opacity=0.12)
            s2 = s1.copy().set_stroke(C_S2).set_fill(C_S2, 0.12)
            VGroup(s1, s2).arrange(RIGHT, buff=1.4).move_to([0, 0.9, 0])
            t1 = VGroup(T('session 1 · test', 26, C_S1, weight='BOLD'), T('8 angles · 8 active trials', 18, DIM),
                        T('the model learns here', 18, INK)).arrange(DOWN, buff=0.15).move_to(s1)
            t2 = VGroup(T('session 2 · retest', 26, C_S2, weight='BOLD'), T('the same 8 angles', 18, DIM),
                        T('the model is scored here', 18, INK)).arrange(DOWN, buff=0.15).move_to(s2)
            arr = Arrow(s1.get_right(), s2.get_left(), color=DIM, buff=0.15)
            note = T('same recording file, same day · electrodes and tissue can change in between', 18, DIM)
            note.next_to(VGroup(s1, s2), DOWN, buff=0.5)
            self.play(FadeIn(s1), FadeIn(t1), run_time=0.8)
            self.play(GrowArrow(arr), FadeIn(s2), FadeIn(t2), run_time=0.8)
            self.play(FadeIn(note), run_time=0.6)
        self.play(FadeOut(VGroup(s1, s2, t1, t2, arr, note, tag)), run_time=0.6)
