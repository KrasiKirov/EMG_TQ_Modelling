from manim import *

from style import (C_BAD, C_GOOD, C_HI, C_S1, C_S2, C_SOL, C_TQ, DIM, FAINT, INK, MUSCLE_C, PANEL, V, BeatScene,
                   CodePanel, Plot, T, chapter_tag, code_at_bottom, facts, load)

F = facts()['x03']
X = load('x03')
NAMES = ['GM', 'GL', 'SOL', 'TA']


class Ch05(BeatScene):
    def construct(self):
        tag = chapter_tag(5, 'EMG scaling and auto-gain')
        self.play(FadeIn(tag), run_time=0.4)

        # 5.1 step one: ÷ the session-1 maximum
        with self.beat('c05_b01') as d:
            t, base = X['eg_t'], X['eg_base']
            sel = (t >= 1) & (t <= 31)
            P = Plot((1, 31), (0, 1.0), 8.6, 3.0, xticks=(1, 11, 21, 31), yticks=(0, 0.5, 1.0), xlabel='time in trial (s)',
                     ylabel='envelope ÷ session-1 max', tick_size=14, grid=True).move_to([-1.3, 1.15, 0])
            curves = VGroup(*[P.curve(t[sel], base[sel, k], MUSCLE_C[k], 1.6) for k in range(4)])
            mx = F['per_subject']['EG']['train_max']
            table = VGroup(T('divided by (recorded units)', 16, DIM),
                           *[T(f'{n}  {m:.4f}', 20, c, font='Consolas') for n, m, c in zip(NAMES, mx, MUSCLE_C)])
            table.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([5.0, 1.3, 0])
            note = VGroup(T('max of the 1 kHz envelope', 15, DIM), T('over whole session-1 trials', 15, DIM),
                          T('(training + validation parts)', 15, DIM)).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
            note.next_to(table, DOWN, buff=0.35).align_to(table, LEFT)
            ttl = T('EG · session 1 · −0.038 rad', 18, C_S1).next_to(P, UP, buff=0.08).align_to(P, LEFT)
            code = code_at_bottom(CodePanel('trainmax'))
            self.play(FadeIn(P), FadeIn(ttl), run_time=0.5)
            self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.2), FadeIn(table), run_time=2.0)
            self.show_code(code)
            self.mark(code, 86)
            self.play(FadeIn(note), run_time=0.5)
            self.wait(max(0.1, d * 0.6 - 4.4))
            self.mark(code, 91, color=C_S2)
        self.play(FadeOut(VGroup(P, curves, table, note, ttl, code)), run_time=0.5)

        # 5.2 who gets auto-gain: push directions
        with self.beat('c05_b02') as d:
            groups = VGroup()
            for k, (s, col) in enumerate((('EG', C_S1), ('CH', C_S2))):
                dirs = F['per_subject'][s]['push_directions']
                pl = Plot((-0.5, 7.5), (-60, 32), 5.2, 3.0, yticks=(-50, -25, 0, 25), tick_size=13, grid=False,
                          ylabel='Nm').move_to([-3.4 + 6.8 * k, 1.4, 0])
                lines = VGroup(*[DashedLine(pl.p(-0.5, v), pl.p(7.5, v), color=C_HI, dash_length=0.06, stroke_width=1.5)
                                 for v in (2, -2)])
                bars = VGroup()
                for i, v in enumerate(dirs):
                    r = Rectangle(width=0.42, height=abs(pl.p(0, v)[1] - pl.p(0, 0)[1]), stroke_width=0,
                                  fill_color=col, fill_opacity=0.85)
                    r.move_to(pl.p(i, 0), aligned_edge=DOWN if v > 0 else UP)
                    bars.add(r)
                ttl = T(f'{s} · 8 session-1 trials', 18, col).next_to(pl, UP, buff=0.08)
                verdict = T('auto-gain ON' if s == 'EG' else 'auto-gain OFF (pushes both ways)', 18,
                            C_GOOD if s == 'EG' else C_BAD).next_to(pl, DOWN, buff=0.25)
                groups.add(VGroup(pl, lines, bars, ttl, verdict))
            expl = T('torque in the top 20 % of EMG − torque in the bottom 20 %   ·   rule: one > +2 Nm and one < −2 Nm → off',
                     15, DIM).move_to([0, -0.72, 0])
            code = code_at_bottom(CodePanel('push_dirs', 145, 151))
            self.play(FadeIn(groups[0][0]), FadeIn(groups[0][3]), Create(groups[0][1]), run_time=0.6)
            self.show_code(code)
            self.mark(code, 147, 150)
            self.play(LaggedStart(*[GrowFromEdge(b, UP) for b in groups[0][2]], lag_ratio=0.1), run_time=1.2)
            self.play(FadeIn(groups[0][4]), FadeIn(expl), run_time=0.5)
            self.wait(max(0.1, d * 0.6 - 3.6))
            code2 = code_at_bottom(CodePanel('agc_auto'))
            self.play(FadeOut(code), FadeIn(code2), FadeIn(groups[1][0]), FadeIn(groups[1][3]), Create(groups[1][1]),
                      run_time=0.6)
            self.mark(code2, 98)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN if v > 0 else UP) for b, v in
                                    zip(groups[1][2], F['per_subject']['CH']['push_directions'])], lag_ratio=0.1),
                      run_time=1.2)
            self.play(FadeIn(groups[1][4]), run_time=0.5)
        self.play(FadeOut(VGroup(groups, expl, code2)), run_time=0.5)

        # 5.3 running floor and top (JM, session 2, soleus)
        ex = F['jm_example']
        t, y, lo, top = X['jm_t'], X['jm_base'][:, 2], X['jm_lo'][:, 2], X['jm_top'][:, 2]
        sel = t <= 48
        ymax = float(np.max(y[sel])) * 1.08
        P = Plot((1, 48), (0, ymax), 10.8, 2.9, xticks=(1, 10, 20, 30, 40), yticks=(0, round(ymax / 2, 2)), tick_size=14,
                 xlabel='time in trial (s)', ylabel='SOL ÷ S1 max', yfmt=lambda v: f'{v:.2f}').move_to([0.3, 1.25, 0])
        ttl = T(f"JM · session 2 · {V['jm_angle']} rad · soleus", 18, C_S2).next_to(P, UP, buff=0.08).align_to(P, LEFT)
        cy = P.curve(t[sel], y[sel], C_SOL, 1.4, 2400)
        clo = P.curve(t[sel], lo[sel], '#B0B8C1', 2.5)
        ctop = P.curve(t[sel], top[sel], C_TQ, 2.5)
        with self.beat('c05_b03') as d:
            code = code_at_bottom(CodePanel('causal_q', 43, 51))
            self.play(FadeIn(P), FadeIn(ttl), Create(cy), run_time=1.2)
            self.show_code(code)
            tu = ex['t_u']
            w20 = Rectangle(width=P.p(tu, 0)[0] - P.p(max(1, tu - 20), 0)[0], height=2.9, stroke_width=0, fill_color='#B0B8C1',
                            fill_opacity=0.13).move_to([(P.p(tu, 0)[0] + P.p(max(1, tu - 20), 0)[0]) / 2, P.p(0, ymax / 2)[1], 0])
            w45 = Rectangle(width=P.p(tu, 0)[0] - P.p(1, 0)[0], height=2.9, stroke_color=C_TQ, stroke_width=1.2,
                            fill_opacity=0).move_to([(P.p(tu, 0)[0] + P.p(1, 0)[0]) / 2, P.p(0, ymax / 2)[1], 0])
            l20 = T('shaded: past 20 s → 10th percentile = floor', 14, '#B0B8C1').move_to([3.3, P.get_top()[1] + 0.42, 0])
            l45 = T('outlined: past 45 s (here since the start) → 95th percentile = top', 14, C_TQ).move_to([3.3, P.get_top()[1] + 0.16, 0])
            self.mark(code, 44)
            self.play(FadeIn(w20), FadeIn(l20), Create(w45), FadeIn(l45), run_time=1.0)
            self.play(Create(clo), Create(ctop), run_time=2.0)
            leg = VGroup(T('floor', 16, '#B0B8C1'), T('top', 16, C_TQ)).arrange(RIGHT, buff=0.4).next_to(ttl, RIGHT, buff=0.6)
            self.play(FadeIn(leg), run_time=0.4)
        # 5.4 the prior and its weight
        with self.beat('c05_b04') as d:
            code2 = code_at_bottom(CodePanel('causal_q', 45, 51))
            self.play(FadeOut(code), FadeIn(code2), FadeOut(VGroup(w20, l20, w45, l45)), run_time=0.5)
            self.mark(code2, 48, 49)
            W = Plot((0, 48), (0, 1.0), 4.0, 2.0, xticks=(0, 5, 20, 45), yticks=(0, 0.5, 1.0), tick_size=12,
                     xlabel='seconds of past data', ylabel='data weight', grid=True).move_to([4.25, 1.55, 0])
            sec = np.linspace(0, 48, 200)
            wf = np.minimum(sec * 100, 2000) / (np.minimum(sec * 100, 2000) + 500)
            wt = np.minimum(sec * 100, 4500) / (np.minimum(sec * 100, 4500) + 500)
            cf = W.curve(sec, wf, '#B0B8C1', 2.5)
            ct = W.curve(sec, wt, C_TQ, 2.5)
            self.play(VGroup(P, cy, clo, ctop, ttl, leg).animate.scale(0.62).move_to([-2.6, 1.35, 0]), FadeIn(W), run_time=0.8)
            self.play(Create(cf), Create(ct), run_time=1.5)
            caps = VGroup(T('0.80', 12, '#B0B8C1').next_to(W.p(48, 0.8), RIGHT, buff=0.06).shift(DOWN * 0.07),
                          T('0.90', 12, C_TQ).next_to(W.p(48, 0.9), RIGHT, buff=0.06).shift(UP * 0.07))
            self.add(caps)
            lab = VGroup(T('w = n / (n + 500)', 18, INK, font='Consolas'),
                         T('floor: n ≤ 2000 → w ≤ 0.80', 15, '#B0B8C1'), T('top:   n ≤ 4500 → w ≤ 0.90', 15, C_TQ))
            lab.arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(W, DOWN, buff=0.5).align_to(W, LEFT)
            self.play(FadeIn(lab), run_time=0.6)
        # 5.5 one rescale, worked through
        with self.beat('c05_b05') as d:
            code3 = code_at_bottom(CodePanel('agc_fn'))
            self.play(FadeOut(VGroup(W, cf, ct, lab, caps, code2)), FadeIn(code3),
                      VGroup(P, cy, clo, ctop, ttl, leg).animate.scale(0.8).move_to([-4.35, 1.55, 0]), run_time=0.7)
            dot = Dot(P.p(ex['t'], ex['x']), radius=0.07, color=C_HI)
            self.play(FadeIn(dot), run_time=0.4)
            rows = [f"t = {V['wx_t']} s   (window refreshed at {V['wx_t_u']} s)",
                    f"floor = {V['wx_w_lo']} × {V['wx_p10']}  +  {1 - float(V['wx_w_lo']):.2f} × {V['wx_prior_lo']}  =  {V['wx_floor']}",
                    f"top   = {V['wx_w_hi']} × {V['wx_p95']}  +  {1 - float(V['wx_w_hi']):.3f} × {V['wx_prior_hi']}  =  {V['wx_top']}",
                    f"range = max({V['wx_top']} − {V['wx_floor']},  0.35 × {V['wx_prior_range']} = {V['wx_rmin']})  =  {V['wx_range']}",
                    f"a = clip(({V['wx_x']} − {V['wx_floor']}) / {V['wx_range']}, 0, 2)  =  {V['wx_a']}"]
            eq = VGroup(*[T(r, 19, INK, font='Consolas') for r in rows]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            eq.scale_to_fit_width(min(eq.width, 7.4)).move_to([2.95, 1.25, 0])
            eq[-1].set_color(C_HI)
            self.mark(code3, 70)
            for r in eq:
                self.play(FadeIn(r, shift=RIGHT * 0.1), run_time=max(0.4, min(1.2, (d - 2.5) / 6)))
            chk = T('= the value in the array (checked)', 15, C_GOOD).next_to(eq, DOWN, buff=0.2).align_to(eq, LEFT)
            self.play(FadeIn(chk), run_time=0.4)
        # 5.6 despike
        with self.beat('c05_b06') as d:
            code4 = code_at_bottom(CodePanel('despike'))
            self.play(FadeOut(code3), FadeIn(code4), FadeOut(VGroup(eq, chk, dot)), run_time=0.5)
            self.mark(code4, 69)
            s = T(f"EG: {V['eg_despiked']} samples held (16 trials × 4 muscles)", 20, C_HI).move_to([3.4, 1.3, 0])
            self.play(FadeIn(s), run_time=0.5)
        self.play(FadeOut(VGroup(P, cy, clo, ctop, ttl, leg, s, code4)), run_time=0.5)

        # 5.7 before vs after
        with self.beat('c05_b07') as d:
            jm = next(r for r in F['size_ratio']['JM'] if abs(r['angle'] + 0.36) < 0.005)
            B = Plot((-0.5, 2.5), (0, 1.3), 6.4, 3.1, yticks=(0, 0.5, 1.0), tick_size=14, grid=True,
                     ylabel='session 2 ÷ session 1 (size per Nm)').move_to([-2.4, 1.0, 0])
            one = DashedLine(B.p(-0.5, 1), B.p(2.5, 1), color=C_HI, dash_length=0.08)
            bars, labs = VGroup(), VGroup()
            for i in range(3):
                for j, (key, op) in enumerate((('before', 0.35), ('after', 0.95))):
                    v = jm[key][i]
                    r = Rectangle(width=0.36, height=B.p(0, v)[1] - B.p(0, 0)[1], stroke_width=0, fill_color=MUSCLE_C[i],
                                  fill_opacity=op).move_to(B.p(i - 0.2 + 0.4 * j, 0), aligned_edge=DOWN)
                    bars.add(r)
                    labs.add(T(f'{v:.2f}', 14, INK).next_to(r, UP, buff=0.05))
            names = VGroup(*[T(n, 16, c).next_to(B.p(i, 0), DOWN, buff=0.15) for i, (n, c) in enumerate(zip(NAMES[:3], MUSCLE_C))])
            key = T('faint = before, solid = after auto-gain', 15, DIM).next_to(B, UP, buff=0.08)
            ttl = T(f"JM · {V['jm_angle']} rad", 20, INK).next_to(key, UP, buff=0.08)
            side = VGroup(T('EG, all 8 angles, GM GL SOL', 18, DIM), T('typical mismatch', 18, DIM),
                          T(f"×{V['eg_mis_before']} → ×{V['eg_mis_after']}", 34, C_HI, weight='BOLD'),
                          T(f"JM: ×{V['jm_mis_before']} → ×{V['jm_mis_after']}", 18, DIM))
            side.arrange(DOWN, buff=0.15).move_to([4.2, 1.1, 0])
            self.play(FadeIn(B), Create(one), FadeIn(names), FadeIn(key), FadeIn(ttl), run_time=0.6)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15), FadeIn(labs), run_time=2.0)
            self.wait(max(0.1, d * 0.6 - 2.6))
            self.play(FadeIn(side), run_time=0.8)
        self.play(FadeOut(VGroup(B, one, bars, labs, names, key, ttl, side)), run_time=0.5)

        # 5.8 what step one still does inside auto-gain
        with self.beat('c05_b08') as d:
            msg = VGroup(T('(x − floor) / range  is the same whatever x was divided by …', 22, INK),
                         T('… except for one fixed number in the spike detector:  lo > 0.1', 22, C_HI),
                         T(f"differs on {V['cancel_pct']} % of EG's samples, in {V['cancel_trials']} trials", 20, DIM))
            msg.arrange(DOWN, buff=0.3).move_to([0, 1.2, 0])
            code = code_at_bottom(CodePanel('despike', 66, 69))
            self.play(FadeIn(msg[0]), run_time=0.6)
            self.show_code(code)
            self.mark(code, 69)
            self.play(FadeIn(msg[1]), run_time=0.6)
            self.play(FadeIn(msg[2]), run_time=0.6)
        self.play(FadeOut(VGroup(msg, code, tag)), run_time=0.6)
