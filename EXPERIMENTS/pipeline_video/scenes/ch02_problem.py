from manim import *

from style import (C_ANGLE, C_HI, C_PASSIVE, C_S1, C_S2, C_TQ, DIM, FAINT, INK, MUSCLE_C, PANEL, V, BeatScene, Plot, T,
                   chapter_tag, facts, load)

F = facts()
X3, X4, X5 = load('x03'), load('x04'), load('x05')


def small_traces(win, width=2.6, height=2.2):
    """The 20 x 5 window as five little traces (GM, GL, SOL, TA, angle)."""
    g = VGroup()
    cols = MUSCLE_C + [C_ANGLE]
    for k in range(5):
        y = win[:, k]
        lo, hi = float(y.min()), float(y.max())
        pl = Plot((0, 19), (lo - 0.05 * (hi - lo + 1e-6), hi + 0.05 * (hi - lo + 1e-6)), width, height / 5.6, axis_color=PANEL)
        pl.move_to([0, -k * height / 5, 0])
        g.add(VGroup(pl, pl.curve(np.arange(20), y, cols[k], 2.2)))
    return g


class Ch02(BeatScene):
    def construct(self):
        tag = chapter_tag(2, 'The problem')
        self.play(FadeIn(tag), run_time=0.4)

        # 2.1 the task
        with self.beat('c02_b01') as d:
            win = X5['window']
            tr = small_traces(win).move_to([-4.3, 0.8, 0])
            names = VGroup(*[T(n, 15, c).next_to(tr[k], LEFT, buff=0.15) for k, (n, c) in
                             enumerate(zip(['GM', 'GL', 'SOL', 'TA', 'angle'], MUSCLE_C + [C_ANGLE]))])
            box_in = SurroundingRectangle(VGroup(tr, names), color=FAINT, buff=0.2, corner_radius=0.1)
            lab_in = T('last 0.2 s · 20 samples', 18, DIM).next_to(box_in, UP, buff=0.15)
            model = VGroup(RoundedRectangle(corner_radius=0.15, width=2.4, height=1.3, stroke_color=INK, fill_color=PANEL,
                                            fill_opacity=1), T('model', 28, INK)).move_to([0, 0.8, 0])
            out = VGroup(T(f"{F['x05']['example']['y']:.1f} Nm".replace('-', '−'), 40, C_TQ, weight='BOLD'),
                         T('active torque, now', 18, DIM)).arrange(DOWN, buff=0.15).move_to([4.2, 0.8, 0])
            a1 = Arrow(box_in.get_right(), model.get_left(), color=DIM, buff=0.15)
            a2 = Arrow(model.get_right(), out.get_left(), color=DIM, buff=0.15)
            p1 = VGroup(RoundedRectangle(corner_radius=0.2, width=4.2, height=0.6, stroke_color=C_S1, fill_color=C_S1,
                                         fill_opacity=0.15), T('learn on session 1', 20, C_S1))
            p2 = VGroup(RoundedRectangle(corner_radius=0.2, width=4.9, height=0.6, stroke_color=C_S2, fill_color=C_S2,
                                         fill_opacity=0.15), T('score on session 2 (never seen)', 20, C_S2))
            VGroup(p1, p2).arrange(RIGHT, buff=0.6).move_to([0, -1.6, 0])
            self.play(FadeIn(box_in), FadeIn(lab_in), LaggedStart(*[Create(x[1]) for x in tr], lag_ratio=0.15),
                      FadeIn(VGroup(*[x[0] for x in tr])), FadeIn(names), run_time=1.6)
            self.play(GrowArrow(a1), FadeIn(model), run_time=0.7)
            self.play(GrowArrow(a2), FadeIn(out), run_time=0.7)
            self.wait(max(0.1, d * 0.55 - 3.0))
            self.play(FadeIn(p1), run_time=0.5)
            self.play(FadeIn(p2), run_time=0.5)
        self.play(FadeOut(VGroup(tr, names, box_in, lab_in, model, out, a1, a2, p1, p2)), run_time=0.5)

        # 2.2 EMG size changes between sessions (JM, -0.36 rad)
        with self.beat('c02_b02') as d:
            hdr = T(f"1 · EMG size changes · JM, {V['jm_angle']} rad, soleus ÷ session-1 max", 22, INK).move_to([0, 2.95, 0])
            sel = (X3['jm_s1_t'] >= 1) & (X3['jm_s1_t'] <= 41)
            sel2 = (X3['jm_t'] >= 1) & (X3['jm_t'] <= 41)
            ymax = float(max(X3['jm_s1_base'][sel, 2].max(), X3['jm_base'][sel2, 2].max())) * 1.05
            pe1 = Plot((1, 41), (0, ymax), 5.6, 1.6, yticks=(0, round(ymax / 2, 2)), ylabel='SOL', tick_size=13,
                       yfmt=lambda v: f'{v:.2f}').move_to([-3.4, 1.35, 0])
            pe2 = Plot((1, 41), (0, ymax), 5.6, 1.6, yticks=(0, round(ymax / 2, 2)), ylabel='SOL', tick_size=13,
                       yfmt=lambda v: f'{v:.2f}').move_to([3.4, 1.35, 0])
            tqlo = float(min(X3['jm_s1_tq'][sel].min(), X3['jm_tq'][sel2].min()))
            tqhi = float(max(X3['jm_s1_tq'][sel].max(), X3['jm_tq'][sel2].max()))
            pt1 = Plot((1, 41), (tqlo, tqhi), 5.6, 1.1, xticks=(1, 11, 21, 31, 41), ylabel='Nm', tick_size=13,
                       xlabel='time in trial (s)').move_to([-3.4, -0.45, 0])
            pt2 = Plot((1, 41), (tqlo, tqhi), 5.6, 1.1, xticks=(1, 11, 21, 31, 41), ylabel='Nm', tick_size=13,
                       xlabel='time in trial (s)').move_to([3.4, -0.45, 0])
            c1 = pe1.curve(X3['jm_s1_t'][sel], X3['jm_s1_base'][sel, 2], C_S1, 1.8)
            c2 = pe2.curve(X3['jm_t'][sel2], X3['jm_base'][sel2, 2], C_S2, 1.8)
            q1 = pt1.curve(X3['jm_s1_t'][sel], X3['jm_s1_tq'][sel], C_TQ, 1.5)
            q2 = pt2.curve(X3['jm_t'][sel2], X3['jm_tq'][sel2], C_TQ, 1.5)
            l1 = T('session 1', 20, C_S1).next_to(pe1, UP, buff=0.1)
            l2 = T('session 2', 20, C_S2).next_to(pe2, UP, buff=0.1)
            self.play(FadeIn(hdr), FadeIn(VGroup(pe1, pe2, pt1, pt2, l1, l2)), run_time=0.8)
            self.play(Create(c1), Create(q1), run_time=1.6)
            self.play(Create(c2), Create(q2), run_time=1.6)
            jm = next(r for r in F['x03']['size_ratio']['JM'] if abs(r['angle'] + 0.36) < 0.005)
            ratio = VGroup(*[T(f"{n}: {v:.2f}×", 22, c) for n, v, c in zip(['GM', 'GL', 'SOL'], jm['before'][:3], MUSCLE_C)])
            ratio.arrange(RIGHT, buff=0.7).move_to([0, -2.0, 0])
            cap = T('session-2 EMG size per Nm of push, relative to session 1', 17, DIM).next_to(ratio, DOWN, buff=0.18)
            self.play(FadeIn(ratio), FadeIn(cap), run_time=0.8)
        self.play(FadeOut(VGroup(hdr, pe1, pe2, pt1, pt2, c1, c2, q1, q2, l1, l2, ratio, cap)), run_time=0.5)

        # 2.3 passive torque
        with self.beat('c02_b03') as d:
            hdr = T('2 · Passive torque · EG, relaxed', 22, INK).move_to([0, 2.95, 0])
            maps = F['x04']['maps']['EG']['static']
            pm = Plot((-0.8, 0.3), (-11.5, 4), 8.0, 4.0, xticks=(-0.7, -0.5, -0.3, -0.1, 0.1, 0.3), yticks=(-10, -5, 0),
                      xlabel='ankle angle (rad)   ← toes down · toes up →', ylabel='passive torque (Nm)', tick_size=15,
                      grid=True).move_to([0, 0.1, 0])
            g = VGroup()
            for s, col in (('S1', C_S1), ('S2', C_S2)):
                pts = maps[s]
                line = VMobject(stroke_color=col, stroke_width=2.5).set_points_as_corners([pm.p(a, t) for a, t in pts])
                dots = VGroup(*[Dot(pm.p(a, t), radius=0.06, color=col) for a, t in pts])
                g.add(VGroup(line, dots))
            lab1 = T(f"{V['eg_map_low']} Nm", 18, C_S1).next_to(pm.p(*maps['S1'][0]), UP, buff=0.15)
            lab2 = T(f"{V['eg_map_high']} Nm", 18, C_S1).next_to(pm.p(*maps['S1'][-1]), LEFT, buff=0.15)
            lab3 = T(f"{V['eg_map_high_s2']} Nm", 18, C_S2).next_to(pm.p(*maps['S2'][-1]), UP, buff=0.12)
            leg = VGroup(T('session 1', 18, C_S1), T('session 2', 18, C_S2)).arrange(RIGHT, buff=0.5).next_to(pm, DOWN, buff=0.7)
            self.play(FadeIn(hdr), FadeIn(pm), run_time=0.6)
            self.play(Create(g[0]), FadeIn(lab1), FadeIn(lab2), run_time=1.5)
            self.play(Create(g[1]), FadeIn(lab3), FadeIn(leg), run_time=1.5)
        self.play(FadeOut(VGroup(hdr, pm, g, lab1, lab2, lab3, leg)), run_time=0.5)

        # 2.4 small pushes at the toes-down end
        with self.beat('c02_b04') as d:
            hdr = T('3 · Small pushes toes-down · EG, session 1, active torque', 22, INK).move_to([0, 2.95, 0])
            k_hi = F['x05']['sd_argmax']
            ya, yb = X5['knot0_target'], X5[f'knot{k_hi}_target']
            ta, tb = X5['knot0_t'], X5[f'knot{k_hi}_t']
            sa, sb = (ta >= 1) & (ta <= 41), (tb >= 1) & (tb <= 41)
            lo = float(min(ya[sa].min(), yb[sb].min())) - 1
            hi = float(max(ya[sa].max(), yb[sb].max())) + 1
            pa = Plot((1, 41), (lo, hi), 4.8, 3.2, xticks=(1, 21, 41), yticks=(-30, -20, -10, 0), ylabel='active torque (Nm)',
                      xlabel='time in trial (s)', tick_size=14, grid=True).move_to([-3.9, 0.15, 0])
            pb = Plot((1, 41), (lo, hi), 4.8, 3.2, xticks=(1, 21, 41), yticks=(-30, -20, -10, 0), xlabel='time in trial (s)',
                      tick_size=14, grid=True).move_to([1.9, 0.15, 0])
            ca, cb = pa.curve(ta[sa], ya[sa], C_TQ, 1.6), pb.curve(tb[sb], yb[sb], C_TQ, 1.6)
            la = T(f"{V['sd_min_angle']} rad", 20, INK).next_to(pa, UP, buff=0.1)
            lb = T(f"{V['sd_max_angle']} rad", 20, INK).next_to(pb, UP, buff=0.1)
            sd = F['x05']['sd']
            bars = VGroup(Rectangle(width=0.55, height=3.0 * sd[0] / max(sd), fill_color=C_S1, fill_opacity=0.85, stroke_width=0),
                          Rectangle(width=0.55, height=3.0 * max(sd) / max(sd), fill_color=C_S1, fill_opacity=0.85, stroke_width=0))
            bars.arrange(RIGHT, buff=0.5, aligned_edge=DOWN).move_to([5.8, -0.1, 0])
            bl = VGroup(T(f"{V['sd_min']} Nm", 16, INK).next_to(bars[0], UP, buff=0.08),
                        T(f"{V['sd_max']} Nm", 16, INK).next_to(bars[1], UP, buff=0.08))
            bt = T('standard deviation', 15, DIM).next_to(bars, DOWN, buff=0.2)
            self.play(FadeIn(hdr), FadeIn(pa), FadeIn(pb), FadeIn(la), FadeIn(lb), run_time=0.6)
            self.play(Create(ca), Create(cb), run_time=2.0)
            self.play(GrowFromEdge(bars[0], DOWN), GrowFromEdge(bars[1], DOWN), FadeIn(bl), FadeIn(bt), run_time=1.0)
            err = T('1 Nm of error: minor here', 17, DIM).next_to(pb, DOWN, buff=0.75)
            err2 = T('1 Nm of error: half the signal here', 17, C_HI).next_to(pa, DOWN, buff=0.75)
            self.wait(max(0.1, d * 0.5 - 4.2))
            self.play(FadeIn(err), FadeIn(err2), run_time=0.8)
        self.play(FadeOut(VGroup(hdr, pa, pb, ca, cb, la, lb, bars, bl, bt, err, err2, tag)), run_time=0.6)
