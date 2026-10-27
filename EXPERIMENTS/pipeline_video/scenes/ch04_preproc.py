from manim import *

from style import (C_HI, C_SOL, DIM, FAINT, INK, PANEL, V, BeatScene, CodePanel, Plot, T, chapter_tag, code_at_bottom,
                   envelope_curve, facts, load)

F = facts()['x02']
X = load('x02')


class Ch04(BeatScene):
    def construct(self):
        tag = chapter_tag(4, 'Preprocessing, step by step')
        self.play(FadeIn(tag), run_time=0.4)
        t = X['piece_t']
        t0, t1 = float(t[0]), float(t[-1])
        A = float(np.max(np.abs(X['piece_band']))) * 1.1
        raw_lo, raw_hi = float(X['piece_raw'].min()), float(X['piece_raw'].max())
        lo = min(-A, raw_lo) * 1.05
        hi = max(A, raw_hi) * 1.05
        P = Plot((t0, t1), (lo, hi), 8.4, 3.1, xticks=np.round(np.arange(np.ceil(t0 * 2) / 2, t1, 0.5), 1),
                 xlabel='time in trial (s)', ylabel='soleus (recorded units)', tick_size=14,
                 xfmt=lambda v: f'{v:.1f}').move_to([-1.75, 1.2, 0])
        zero = DashedLine(P.p(t0, 0), P.p(t1, 0), color=FAINT, dash_length=0.06)
        title = T('EG · session 1 · −0.038 rad · soleus, 3 s', 18, DIM).move_to([3.9, 3.62, 0])

        def cur(y, col=C_SOL, w=1.2):
            return envelope_curve(P, t, y, col, w, 2400)

        # 4.1 demean
        with self.beat('c04_b01') as d:
            raw = cur(X['piece_raw'], '#B0B8C1')
            mean = F['mean_offset']
            mline = DashedLine(P.p(t0, mean), P.p(t1, mean), color=C_HI, dash_length=0.08)
            mlab = T(f"trial mean {V['env_mean']}", 16, C_HI).next_to(P, RIGHT, buff=0.3).set_y(P.p(t1, mean)[1])
            code = code_at_bottom(CodePanel('demean'))
            self.play(FadeIn(P), FadeIn(title), FadeIn(zero), run_time=0.6)
            self.play(Create(raw), run_time=1.4)
            self.show_code(code)
            self.play(Create(mline), FadeIn(mlab), run_time=0.6)
            self.mark(code, 124)
            dem = cur(X['piece_demeaned'], '#B0B8C1')
            self.play(Transform(raw, dem), mline.animate.move_to(P.p((t0 + t1) / 2, 0)), FadeOut(mlab), run_time=1.0)
        # 4.2 band-pass
        with self.beat('c04_b02') as d:
            code2 = code_at_bottom(CodePanel('bandpass'))
            self.play(FadeOut(code), FadeIn(code2), FadeOut(mline), run_time=0.5)
            g = Plot((0, np.log10(499)), (0, 1.05), 2.8, 1.25, xticks=(np.log10(10), np.log10(30), np.log10(100),
                                                                        np.log10(300)), tick_size=12,
                     xfmt=lambda v: f'{10 ** v:.0f}').move_to([4.9, 1.9, 0])
            gl = T('gain × gain (forward + backward), Hz', 12, DIM).next_to(g, DOWN, buff=0.32)
            gc = g.curve(np.log10(X['f']), X['gain_bp'], C_HI, 2)
            band = cur(X['piece_band'])
            self.mark(code2, 129, 130)
            self.play(Transform(raw, band), run_time=1.2)
            self.play(FadeIn(g), Create(gc), FadeIn(gl), run_time=1.0)
        # 4.3 rectify
        with self.beat('c04_b03') as d:
            code3 = code_at_bottom(CodePanel('rectify'))
            self.play(FadeOut(code2), FadeIn(code3), FadeOut(VGroup(g, gc, gl)), run_time=0.5)
            rect = cur(X['piece_rect'])
            self.play(Transform(raw, rect), run_time=min(1.2, d * 0.5))
        # 4.4 low-pass: the envelope, and how far it looks ahead
        with self.beat('c04_b04') as d:
            code4 = code_at_bottom(CodePanel('lowpass'))
            self.play(FadeOut(code3), FadeIn(code4), run_time=0.5)
            env = cur(X['piece_env'], INK, 3.0)
            self.play(raw.animate.set_stroke(opacity=0.35), run_time=0.4)
            self.mark(code4, 136, 137)
            self.play(Create(env), run_time=1.5)
            ip = Plot((-1.0, 1.0), (-0.1, 1.05), 3.4, 1.3, xticks=(-1, -0.5, 0, 0.5, 1), tick_size=12,
                      xfmt=lambda v: f'{v:+.1f}' if v else '0').move_to([4.75, 2.05, 0])
            sel = np.abs(X['imp_t']) <= 1.0
            band90 = Rectangle(width=ip.p(F['imp90'][1], 0)[0] - ip.p(F['imp90'][0], 0)[0], height=1.3,
                               fill_color=C_HI, fill_opacity=0.15, stroke_width=0).move_to(ip.p(0, 0.475))
            ic = ip.curve(X['imp_t'][sel], X['imp_h'][sel], INK, 2)
            il = T(f"90 % of the weight within ±{V['imp90']} s", 12, C_HI).next_to(ip, DOWN, buff=0.32)
            past = T('past', 11, DIM).next_to(ip.p(-0.75, 0.8), UP, buff=0)
            fut = T('future', 11, DIM).next_to(ip.p(0.75, 0.8), UP, buff=0)
            self.play(FadeIn(ip), FadeIn(band90), Create(ic), FadeIn(il), FadeIn(past), FadeIn(fut), run_time=1.2)
            self.wait(max(0.1, d * 0.55 - 4.0))
            note = VGroup(T('zero-phase: uses samples', 16, C_HI), T('from the future →', 16, C_HI),
                          T('not real-time as written', 16, C_HI)).arrange(DOWN, buff=0.06).move_to([4.75, -0.1, 0])
            self.play(FadeIn(note), run_time=0.6)
        # 4.5 trim and decimate
        with self.beat('c04_b05') as d:
            code5 = code_at_bottom(CodePanel('trim_decimate'))
            self.play(FadeOut(VGroup(code4, ip, band90, ic, il, past, fut, note)), FadeIn(code5), run_time=0.6)
            # zoom on 0.4 s of the envelope: 1 kHz line vs 100 Hz samples
            z0 = float(X['piece_t100'][len(X['piece_t100']) // 2])
            z1 = z0 + 0.4
            m1 = (t >= z0) & (t <= z1)
            m2 = (X['piece_t100'] >= z0 - 1e-9) & (X['piece_t100'] <= z1 + 1e-9)
            e = X['piece_env']
            zl, zh = float(e[m1].min()) * 0.9, float(e[m1].max()) * 1.1
            Z = Plot((z0, z1), (zl, zh), 8.4, 3.1, xticks=np.round(np.arange(z0, z1 + 1e-6, 0.1), 2), tick_size=14,
                     xlabel='time in trial (s)', ylabel='soleus envelope', xfmt=lambda v: f'{v:.2f}').move_to([-1.75, 1.2, 0])
            zc = Z.curve(t[m1], e[m1], INK, 2.5)
            dots = VGroup(*[Dot(Z.p(a, b), radius=0.06, color=C_HI) for a, b in zip(X['piece_t100'][m2], X['piece_env100'][m2])])
            self.play(FadeOut(VGroup(P, raw, env, zero, title)), FadeIn(Z), run_time=0.6)
            self.play(Create(zc), run_time=0.8)
            self.mark(code5, 95, 96)
            self.play(LaggedStart(*[GrowFromCenter(x) for x in dots], lag_ratio=0.1), run_time=1.4)
            lab = VGroup(T('1000 → 100 samples', 17, C_HI), T('per second', 17, C_HI), T('FIR anti-aliasing,', 15, DIM),
                         T('zero phase', 15, DIM)).arrange(DOWN, buff=0.06).move_to([4.75, 2.0, 0])
            trim = VGroup(T('first: the first 1000 raw', 15, DIM), T('samples (1 s) are dropped', 15, DIM)).arrange(DOWN, buff=0.06).next_to(lab, DOWN, buff=0.35)
            self.play(FadeIn(lab), FadeIn(trim), run_time=0.6)
        # 4.6 the result
        with self.beat('c04_b06') as d:
            self.play(FadeOut(VGroup(Z, zc, dots, lab, trim, code5)), run_time=0.5)
            rows = VGroup(T(f"env_raw   ({V['n100']}, 4)   GM GL SOL TA envelopes", 26, INK, font='Consolas'),
                          T(f"pos       ({V['n100']},)     ankle angle, rad", 26, INK, font='Consolas'),
                          T(f"tq        ({V['n100']},)     measured torque, Nm", 26, INK, font='Consolas'))
            rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([0, 1.2, 0])
            sub = T(f"{V['n100']} samples at 100 Hz = {V['sec100']} s per active trial", 20, C_HI).next_to(rows, DOWN, buff=0.45)
            code6 = code_at_bottom(CodePanel('env_max1k'))
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.2), run_time=1.2)
            self.play(FadeIn(sub), FadeIn(code6), run_time=0.6)
        self.play(FadeOut(VGroup(rows, sub, code6, tag)), run_time=0.6)
