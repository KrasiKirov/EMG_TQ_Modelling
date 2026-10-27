from manim import *

from style import (C_GOOD, C_HI, C_PRED, C_S1, C_S2, C_SOL, C_TQ, DIM, FAINT, INK, PANEL, V, BeatScene, CodePanel, Plot, T,
                   chapter_tag, code_at_bottom, facts, load)

FF = facts()
F8, F9 = FF['x08'], FF['x09']
X8, X9 = load('x08'), load('x09')
C_LW = '#E3A15B'


class Ch12(BeatScene):
    def construct(self):
        tag = chapter_tag(12, "Last week's pipeline, and recap")
        self.play(FadeIn(tag), run_time=0.4)
        cfg = F9['config']

        # 12.1 the contrast
        with self.beat('c12_b01') as d:
            rows = [('input window', f"{cfg['window'] / 100:.1f} s ({cfg['window']} samples)", '0.2 s (20 samples)'),
                    ('training windows', f"every {cfg['train_stride']}th", 'every one'),
                    ('split', f"{cfg['gap']:.0f} s gap · parts filtered separately", 'no gap · one pass per trial'),
                    ('EMG scaling', '÷ training maximum', '÷ S1 max, then auto-gain (not CH)'),
                    ('target scaling', 'one mean and SD for all angles', '÷ SD of each angle'),
                    ('training', f"batch {cfg['batch']} · ≤ {cfg['epochs']} epochs · patience {cfg['patience']}",
                     'batch 8 · ≤ 200 epochs · patience 25'),
                    ('EG −0.038 rad, seed 0', f"R² {V['lw_r2']} · MAE {V['lw_mae']} Nm", f"R² {V['live_r2']} · MAE {V['live_mae']} Nm")]
            hd = VGroup(T("last week (0.5 s window + retest map)", 17, C_LW).move_to([-0.6, 3.15, 0]),
                        T('D01 · balanced positions', 17, C_GOOD).move_to([4.2, 3.15, 0]))
            tab = VGroup()
            for k, (a, b, c) in enumerate(rows):
                y = 2.7 - k * 0.42
                tab.add(VGroup(T(a, 15, DIM).move_to([-4.9, y, 0]), T(b, 15, INK).move_to([-0.6, y, 0]),
                               T(c, 15, INK if k < 6 else C_HI).move_to([4.2, y, 0])))
            tab[-1][1].set_color(C_LW)
            code = code_at_bottom(CodePanel('lw_config'))
            self.play(FadeIn(hd), run_time=0.4)
            self.show_code(code)
            for k, r in enumerate(tab[:-1]):
                self.play(FadeIn(r), run_time=max(0.35, min(1.6, (d * 0.7) / 7)))
                if k == 0:
                    self.mark(code, 23)
                if k == 2:
                    self.mark(code, 26, color=C_S2)
            code2 = code_at_bottom(CodePanel('lw_train'))
            self.play(FadeOut(code), FadeIn(code2), FadeIn(tab[-1]), run_time=0.7)
            self.mark(code2, 69, 72)
        self.play(FadeOut(VGroup(code2)), tab.animate.shift(UP * 0.0), run_time=0.3)
        # overlay on the same trial (shown under the table)
        lw = next(k for k, v in F9['trials'].items() if v['position'] == -0.038)
        tl, yl, pl = X9[f't{lw}_t'], X9[f't{lw}_y'], X9[f't{lw}_p']
        i = F8['eg_live']['trial']
        t, y, p = X8[f'w{i}_t'], X8[f'w{i}_y'], X8[f'w{i}_p']
        P = Plot((20, 50), (-31, 4), 11.0, 2.2, xticks=(20, 30, 40, 50), yticks=(-30, -15, 0), tick_size=12,
                 xlabel='time in trial (s) · EG · session 2 · −0.038 rad', ylabel='Nm', grid=True).move_to([0.4, -1.75, 0])
        m, ml = (t >= 20) & (t <= 50), (tl >= 20) & (tl <= 50)
        cy, cp, cl = P.curve(t[m], y[m], C_TQ, 1.6), P.curve(t[m], p[m], C_PRED, 1.6), P.curve(tl[ml], pl[ml], C_LW, 1.6)
        key = VGroup(T('measured active torque', 14, C_TQ), T('D01', 14, C_PRED), T('last week', 14, C_LW)).arrange(RIGHT, buff=0.5)
        key.next_to(P, UP, buff=0.05)
        self.play(FadeIn(P), FadeIn(key), Create(cy), Create(cl), Create(cp), run_time=2.0)
        self.wait(1.5)
        self.play(FadeOut(VGroup(hd, tab, P, cy, cp, cl, key)), run_time=0.5)

        # 12.2 recap
        with self.beat('c12_b02') as d:
            steps = [('smooth EMG envelopes', 'band-pass · rectify · 2 Hz', C_SOL), ('rescale from the recent past', 'auto-gain', C_SOL),
                     ('subtract a passive map', 'still holds, per session', DIM), ('0.2 s windows', '20 × 5 inputs', C_S1),
                     ('small LSTM, angles balanced', '÷ SD per angle', C_S1), ('score on a new session', 'every person and angle', C_S2)]
            boxes = VGroup()
            for a, b, col in steps:
                r = RoundedRectangle(corner_radius=0.12, width=3.9, height=1.3, stroke_color=col, fill_color=PANEL, fill_opacity=1)
                boxes.add(VGroup(r, VGroup(T(a, 19, INK), T(b, 14, DIM)).arrange(DOWN, buff=0.1).move_to(r)))
            boxes.arrange_in_grid(rows=2, cols=3, buff=(0.45, 0.6)).move_to([0, 0.6, 0])
            step = max(0.4, (d - 1.0) / 6)
            for b in boxes:
                self.play(FadeIn(b, shift=UP * 0.15), run_time=step)
            end = T(f"D01 on 6 people × 8 angles: median R² {V['head_r2']}, MAE {V['head_mae']} Nm (session 1 → 2)", 20, C_HI)
            end.next_to(boxes, DOWN, buff=0.6)
            self.play(FadeIn(end), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(VGroup(boxes, end, tag)), run_time=1.0)
