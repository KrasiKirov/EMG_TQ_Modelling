from manim import *

from style import (ANGLE_C, C_HI, C_PRED, C_S2, C_TQ, DIM, FAINT, INK, PANEL, V, BeatScene, CodePanel, Plot, T,
                   chapter_tag, code_at_bottom, facts, load)

FF = facts()
F8, F5 = FF['x08'], FF['x05']
X8 = load('x08')


class Ch10(BeatScene):
    def construct(self):
        tag = chapter_tag(10, 'Prediction')
        self.play(FadeIn(tag), run_time=0.4)

        # 10.1 one pass, then × SD
        with self.beat('c10_b01') as d:
            specs = [(f"X_test\n({V['x_test']}, 20, 5)", C_S2), ('frozen model\ndropout off', INK),
                     (f"ŷ\n({V['x_test']},)", INK), ('× SD of the\nnearest knot', C_HI), ('active torque\nin Nm', C_PRED)]
            boxes = VGroup()
            for txt, col in specs:
                r = RoundedRectangle(corner_radius=0.1, width=2.2, height=1.15, stroke_color=col, fill_color=PANEL, fill_opacity=1)
                lines = VGroup(*[T(l, 17, col if i == 0 else INK) for i, l in enumerate(txt.split('\n'))]).arrange(DOWN, buff=0.06)
                boxes.add(VGroup(r, lines.move_to(r)))
            boxes.arrange(RIGHT, buff=0.45).move_to([0, 2.0, 0])
            arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.04, color=DIM,
                                    max_tip_length_to_length_ratio=0.3) for i in range(4)])
            knots, sd = np.array(F5['knots']), np.array(F5['sd'])
            s2 = sorted(a for a, _ in F8['eg_wave_trials'])
            pairs = VGroup(*[T(f"{a:+.3f} → knot {knots[np.argmin(abs(knots - a))]:+.3f} → × {sd[np.argmin(abs(knots - a))]:.2f} Nm"
                               .replace('-', '−'), 14, ANGLE_C[int(np.argmin(abs(knots - a)))], font='Consolas') for a in s2])
            pairs.arrange_in_grid(rows=4, cols=2, buff=(0.6, 0.12), col_alignments='ll').move_to([0, 0.15, 0])
            pl = T('session-2 trial angle → nearest training knot → multiplier', 15, DIM).next_to(pairs, UP, buff=0.15)
            code = code_at_bottom(CodePanel('pred_scale'))
            self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.2), LaggedStart(*[GrowArrow(a) for a in arrows],
                                                                                         lag_ratio=0.2), run_time=1.6)
            self.show_code(code)
            self.mark(code, 182)
            self.wait(max(0.1, d * 0.4 - 2.6))
            self.mark(code, 183, 184, color=C_PRED)
            self.play(FadeIn(pl), FadeIn(pairs), run_time=0.8)
        self.play(FadeOut(VGroup(boxes, arrows, pairs, pl, code)), run_time=0.5)

        # 10.2 every session-2 trial of EG, seed 0
        with self.beat('c10_b02') as d:
            panels = VGroup()
            curves = []
            for k, (a, i) in enumerate(F8['eg_wave_trials']):
                t, y, p = X8[f'w{i}_t'], X8[f'w{i}_y'], X8[f'w{i}_p']
                lo, hi = float(min(y.min(), p.min())) - 1, float(max(y.max(), p.max())) + 1
                r, c = k // 2, k % 2
                pl_ = Plot((1, 90.1), (lo, hi), 6.1, 1.05, axis_color=FAINT, yticks=(round(lo + 1), round(hi - 1)), tick_size=11,
                           yfmt=lambda v: f'{v:.0f}').move_to([-3.45 + 7.0 * c, 2.55 - r * 1.62, 0])
                sc = F8['eg_seed0'][f'{a:+.3f}']
                lab = T(f"{a:+.2f} rad · R² {sc['r2']:.3f} · MAE {sc['mae']:.2f} Nm".replace('-', '−'), 14, ANGLE_C[k])
                lab.next_to(pl_, UP, buff=0.04).align_to(pl_, LEFT)
                cy, cp = pl_.curve(t, y, C_TQ, 1.1, 1400), pl_.curve(t, p, C_PRED, 1.1, 1400)
                panels.add(VGroup(pl_, lab))
                curves.append((cy, cp))
            tl = T('time in trial (s), 1 to 90 · Nm on the left · white = measured active torque, green = prediction', 14, DIM)
            tl.move_to([0, -3.75, 0])
            self.play(FadeIn(panels), FadeIn(tl), run_time=0.6)
            step = max(0.4, (d + 14 - 1.0) / 8)
            for cy, cp in curves:
                self.play(Create(cy), Create(cp), run_time=step)
        self.play(FadeOut(VGroup(panels, tl, *[c for pair in curves for c in pair], tag)), run_time=0.6)
