from manim import *

from style import (ANGLE_C, C_BAD, C_GOOD, C_HI, C_PASSIVE, C_PRED, C_S2, C_TQ, DIM, FAINT, INK, PANEL, V, BeatScene,
                   CodePanel, Plot, T, chapter_tag, code_at_bottom, facts, load)

FF = facts()
F8 = FF['x08']
X8 = load('x08')
SUBJ = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']


def r2col(v, lo=0.85):
    x = float(np.clip((v - lo) / (1 - lo), 0, 1))
    return interpolate_color(ManimColor('#3B2A5A'), ManimColor('#4FD39B'), x)


class Ch11(BeatScene):
    def construct(self):
        tag = chapter_tag(11, 'Evaluation, every number')
        self.play(FadeIn(tag), run_time=0.4)
        live = F8['eg_live']
        i = live['trial']
        t, y, p = X8[f'w{i}_t'], X8[f'w{i}_y'], X8[f'w{i}_p']

        # 11.1 what is compared
        with self.beat('c11_b01') as d:
            parts = VGroup(T('measured torque', 22, INK), T('−', 30, INK), T('passive map of the tested session', 22, C_PASSIVE),
                           T('=', 30, INK), T('answer key (active torque)', 22, C_TQ)).arrange(RIGHT, buff=0.3).move_to([0, 2.4, 0])
            sub = T(f"read at every window's last sample: {V['live_n']} per session-2 trial · IES: nothing subtracted", 17, DIM)
            sub.next_to(parts, DOWN, buff=0.3)
            vs = T('compared with: the prediction (active torque, Nm), never with total torque', 18, C_PRED).next_to(sub, DOWN, buff=0.3)
            code = code_at_bottom(CodePanel('pm_key'))
            self.play(LaggedStart(*[FadeIn(x) for x in parts], lag_ratio=0.15), run_time=1.2)
            self.show_code(code)
            self.mark(code, 74, 76)
            self.play(FadeIn(sub), run_time=0.5)
            self.wait(max(0.1, d * 0.5 - 2.9))
            code2 = code_at_bottom(CodePanel('pm_sample'))
            self.play(FadeOut(code), FadeIn(code2), FadeIn(vs), run_time=0.6)
            self.mark(code2, 111, 112)
        self.play(FadeOut(VGroup(parts, sub, vs, code2)), run_time=0.5)

        # 11.2 R² live: SS_res and SS_tot as areas
        P = Plot((1, 90.1), (-31, 4), 6.6, 2.4, xticks=(1, 30, 60, 90), yticks=(-30, -20, -10, 0), tick_size=12,
                 xlabel='time in trial (s)', ylabel='Nm', grid=True).move_to([-3.2, 1.55, 0])
        cy, cp = P.curve(t, y, C_TQ, 1.1, 1600), P.curve(t, p, C_PRED, 1.1, 1600)
        ttl = T(f"EG · session 2 · {V['eg_angle']} rad · seed 0", 17, C_S2).next_to(P, UP, buff=0.06).align_to(P, LEFT)
        mean = DashedLine(P.p(1, live['mean']), P.p(90.1, live['mean']), color=C_HI, dash_length=0.06)
        with self.beat('c11_b02') as d:
            code = code_at_bottom(CodePanel('pm_metrics'))
            self.play(FadeIn(P), FadeIn(ttl), Create(cy), Create(cp), run_time=1.2)
            self.show_code(code)
            side = 2.6
            sq_tot = Square(side, stroke_color=C_TQ, fill_color=C_TQ, fill_opacity=0.12).move_to([3.9, 1.6, 0])
            r = np.sqrt(live['ss_res'] / live['ss_tot'])
            sq_res = Square(side * r, stroke_color=C_BAD, fill_color=C_BAD, fill_opacity=0.8).move_to(sq_tot.get_corner(DL), aligned_edge=DL)
            l_res = T(f"Σ (prediction − key)² = {V['live_ss_res']} Nm²", 16, C_BAD).next_to(sq_tot, DOWN, buff=0.15)
            l_tot = T(f"Σ (key − its mean)² = {V['live_ss_tot']} Nm²", 16, C_TQ).next_to(sq_tot, UP, buff=0.12)
            self.mark(code, 82)
            self.play(GrowFromPoint(sq_res, sq_res.get_corner(DL)), FadeIn(l_res), run_time=1.0)
            self.wait(max(0.1, d * 0.3 - 2.5))
            self.mark(code, 83, color=C_S2)
            self.play(Create(mean), FadeIn(sq_tot), FadeIn(l_tot), run_time=1.0)
            r2 = T(f"R² = 1 − {V['live_ss_res']} / {V['live_ss_tot']} = {V['live_r2']}", 22, C_HI).move_to([3.4, -0.35, 0])
            self.wait(max(0.1, d * 0.25 - 1.0))
            self.mark(code, 85)
            self.play(FadeIn(r2), run_time=0.8)
        # 11.3 MAE and peak
        with self.beat('c11_b03') as d:
            ip = int(np.argmax(np.abs(y)))
            pk = VGroup(Dot(P.p(float(t[ip]), float(y[ip])), radius=0.07, color=C_HI),
                        T(f"peak target |{float(y[ip]):.1f}| = {V['live_peak']} Nm".replace('-', '−'), 15, C_HI)
                        .next_to(P.p(float(t[ip]), float(y[ip])), RIGHT, buff=0.15))
            mae = T(f"MAE = mean |prediction − key| = {V['live_mae']} Nm", 20, INK).next_to(r2, DOWN, buff=0.2)
            note = T('the "peak" shown next to the MAE is a torque, not an error', 15, DIM).next_to(mae, DOWN, buff=0.12)
            self.play(FadeIn(mae), run_time=0.6)
            self.mark(code, 87, color=C_GOOD)
            self.play(FadeIn(pk), FadeIn(note), run_time=0.7)
        self.play(FadeOut(VGroup(P, cy, cp, ttl, mean, sq_tot, sq_res, l_res, l_tot, r2, mae, pk, note, code)), run_time=0.5)

        # 11.4 EG's 8 x 10 table
        tab = F8['eg_table']
        R = np.array(tab['r2'])
        with self.beat('c11_b04') as d:
            hd = VGroup(T('angle', 14, DIM).move_to([-6.0, 3.0, 0]),
                        *[T(f'seed {s}', 13, DIM).move_to([-4.85 + j * 1.0, 3.0, 0]) for j, s in enumerate(tab['seeds'])],
                        T('median', 14, C_HI).move_to([5.45, 3.0, 0]))
            rows = VGroup()
            for k, a in enumerate(tab['positions']):
                cells = [T(f'{a:+.3f}'.replace('-', '−'), 15, ANGLE_C[k], font='Consolas').move_to([-6.0, 2.55 - k * 0.42, 0])]
                for j in range(R.shape[1]):
                    v = R[k, j]
                    cells.append(T(f'{max(v, 0):.3f}' if v >= 0 else '0 (<0)', 14, INK if v >= 0 else C_BAD,
                                   font='Consolas').move_to([-4.85 + j * 1.0, 2.55 - k * 0.42, 0]))
                cells.append(T(f"{tab['median_r2'][k]:.3f}", 15, C_HI, font='Consolas').move_to([5.45, 2.55 - k * 0.42, 0]))
                rows.add(VGroup(*cells))
            cap = T('R², learn session 1 → test session 2; negative values would be shown as 0 before the median', 15, DIM)
            cap.move_to([0, -0.95, 0])
            code = code_at_bottom(CodePanel('dash_floor'))
            self.play(FadeIn(hd), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(r[:-1]) for r in rows], lag_ratio=0.2), run_time=min(3.2, d * 0.45))
            self.show_code(code)
            self.mark(code, 34)
            self.play(LaggedStart(*[FadeIn(r[-1]) for r in rows], lag_ratio=0.1), FadeIn(cap), run_time=1.0)
        self.play(FadeOut(VGroup(hd, rows, cap, code)), run_time=0.5)

        # 11.5 across people: 48 cells
        ag = F8['aggregate']['S1toS2']
        cells = ag['cells_r2']
        with self.beat('c11_b05') as d:
            grid = VGroup()
            toes = VGroup()
            for r_, s in enumerate(SUBJ):
                keys = sorted((k for k in cells if k.startswith(s + '|')), key=lambda k: float(k.split('|')[1]))
                grid.add(T(s, 18, INK).move_to([-6.25, 2.6 - r_ * 0.5, 0]))
                for c_, k in enumerate(keys):
                    v = cells[k]
                    box = Rectangle(width=0.86, height=0.42, stroke_width=1, stroke_color=PANEL, fill_color=r2col(v),
                                    fill_opacity=1).move_to([-5.35 + c_ * 0.9, 2.6 - r_ * 0.5, 0])
                    g = VGroup(box, T(f'{v:.3f}', 13, INK, font='Consolas').move_to(box))
                    grid.add(g)
                    if c_ < 2:
                        toes.add(box.copy().set_fill(opacity=0).set_stroke(C_HI, 2.5))
            hdr = T('median R² over seeds per person and angle (R² < 0 → 0) · columns: angles, toes-down first', 15, DIM)
            hdr.move_to([0, 3.15, 0])
            res = VGroup(T(f"all {V['cells']} cells", 17, INK), T(f"R² {V['head_r2']} · MAE {V['head_mae']} Nm", 22, INK, weight='BOLD'),
                         T('toes-down (12 cells)', 16, C_HI), T(f"R² {V['toes_r2']} · MAE {V['toes_mae']} Nm", 18, C_HI),
                         T(f"other ({V['other_cells']} cells)", 16, DIM), T(f"R² {V['other_r2']} · MAE {V['other_mae']} Nm", 18, DIM))
            res.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.95, 1.4, 0])
            code = code_at_bottom(CodePanel('dash_score'))
            self.play(FadeIn(hdr), LaggedStart(*[FadeIn(g) for g in grid], lag_ratio=0.01), run_time=1.6)
            self.show_code(code)
            self.mark(code, 32, 33)
            self.play(FadeIn(res[0:2]), run_time=0.6)
            self.wait(max(0.1, d * 0.5 - 3.2))
            self.play(Create(toes), FadeIn(res[2:4]), run_time=0.8)
            self.play(FadeIn(res[4:6]), run_time=0.6)
        self.play(FadeOut(VGroup(grid, toes, hdr, res, code)), run_time=0.5)

        # 11.6 the bands
        with self.beat('c11_b06') as d:
            box = tab['box']
            B = Plot((-0.6, 7.6), (0.9, 1.0), 9.0, 2.7, yticks=(0.9, 0.95, 1.0), tick_size=12, grid=True, ylabel='R²',
                     yfmt=lambda v: f'{v:.2f}').move_to([-1.5, 1.45, 0])
            g = VGroup()
            for k, a in enumerate(tab['positions']):
                q = box[f'{a:+.3f}']
                clip = lambda v: float(np.clip(v, 0.9, 1.0))
                rect = Rectangle(width=0.5, height=max(0.01, B.p(0, clip(q['75']))[1] - B.p(0, clip(q['25']))[1]),
                                 stroke_color=ANGLE_C[k], fill_color=ANGLE_C[k], fill_opacity=0.3).move_to(
                    B.p(k, (clip(q['75']) + clip(q['25'])) / 2))
                wh = Line(B.p(k, clip(q['5'])), B.p(k, clip(q['95'])), color=ANGLE_C[k], stroke_width=2)
                med = Line(B.p(k - 0.25, clip(q['50'])), B.p(k + 0.25, clip(q['50'])), color=INK, stroke_width=3)
                dots = VGroup(*[Dot(B.p(k + 0.33, clip(max(v, 0))), radius=0.03, color=INK) for v in R[k]])
                g.add(VGroup(wh, rect, med, dots))
            xl = VGroup(*[T(f'{a:+.2f}'.replace('-', '−'), 13, ANGLE_C[k]).next_to(B.p(k, 0.9), DOWN, buff=0.1)
                          for k, a in enumerate(tab['positions'])])
            key = T('EG · box = 25–75 %, whiskers = 5–95 % over the 10 seeds, dots = each seed (axis from 0.90)', 14, DIM)
            key.next_to(B, UP, buff=0.08)
            kp = ag['toes']['kpi_r2']
            kpi = VGroup(T('overview cards, toes-down group:', 15, DIM),
                         T(f"centre {kp['centre']:.3f} (median of cell medians)", 15, INK),
                         T(f"5–95 %: {kp['p5']:.3f} – {kp['p95']:.3f}", 15, C_HI),
                         T('= percentiles over the 10 seeds of', 14, DIM), T('each seed\'s median across the cells', 14, DIM))
            kpi.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([5.05, 1.35, 0])
            code = code_at_bottom(CodePanel('dash_quantile'))
            self.play(FadeIn(B), FadeIn(xl), FadeIn(key), LaggedStart(*[FadeIn(x) for x in g], lag_ratio=0.1), run_time=1.6)
            self.show_code(code)
            self.mark(code, 7, 10)
            self.wait(max(0.1, d * 0.55 - 2.6))
            code2 = code_at_bottom(CodePanel('dash_group'))
            self.play(FadeOut(code), FadeIn(code2), FadeIn(kpi), run_time=0.7)
            self.mark(code2, 54, 56)
        self.play(FadeOut(VGroup(B, g, xl, key, kpi, code2, tag)), run_time=0.6)
