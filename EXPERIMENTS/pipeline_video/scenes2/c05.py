"""c05: One scale per muscle (divide by the session-1 maximum; session-2 size change; two small notes)."""
from manim import *

from style2 import *

F = facts()
D = load('y02')
NAMES = ['GM', 'GL', 'SOL', 'TA']
KEYS = ['gm', 'gl', 'sol', 'ta']
EG_I = [str(s) for s in D['emg_max_subjects']].index('EG')
EG_MAX = [float(v) for v in D['emg_max'][EG_I]]                    # session-1 maxima of EG's four channels (raw units)
PEAKS = F['y02']['eg_S1_trial']['norm_env_max_in_trial']              # trial 21 peaks after dividing
JM_RATIO = [r for r in F['x03']['size_ratio']['JM'] if abs(r['angle'] + 0.36) < 1e-6][0]['before']
VAL_PAIRS = {tuple(s.split()) for s in F['y02']['max_in_validation_list']}


def legend_row():
    row = VGroup()
    for n, c in zip(NAMES, MUSCLE_C):
        row.add(VGroup(Rectangle(width=0.28, height=0.28, fill_color=c, fill_opacity=1, stroke_width=0), T(n, 22, c))
                .arrange(RIGHT, buff=0.14))
    return row.arrange(RIGHT, buff=0.55)


def bump(cx, base, width, height, color, sw=4.0):
    xs = np.linspace(-1, 1, 60)
    ys = np.exp(-(xs * 2.6) ** 2)
    pts = [np.array([cx + x * width / 2, base + y * height, 0]) for x, y in zip(xs, ys)]
    return VMobject(stroke_color=color, stroke_width=sw).set_points_as_corners(pts)


class Ch05(BeatScene):
    def construct(self):
        # ── b01: divide each channel by its session-1 maximum ───────────────────────────────────────────────
        with self.beat('c05_b01'):
            tag = self.intro('c05')
            t = D['s1_t']
            lp = Plot((0, 90), (0, 0.14), 4.6, 3.2, xticks=[0, 30, 60, 90], yticks=[0, 0.05, 0.10],
                      xlabel='time in trial (s)', ylabel='EMG envelope (mV)', yfmt=lambda v: f'{v:.2f}',
                      tick_size=18).move_to([-3.4, 0.35, 0])
            rp = Plot((0, 90), (0, 1.15), 4.6, 3.2, xticks=[0, 30, 60, 90], yticks=[0, 0.5, 1],
                      xlabel='time in trial (s)', ylabel='EMG envelope ÷ session-1 max', yfmt=lambda v: f'{v:g}',
                      tick_size=18).move_to([3.6, 0.35, 0])
            ttl_l = T('Raw sizes', 26, INK, weight='BOLD').move_to([-3.4, 2.85, 0])
            ttl_r = T('After dividing', 26, INK, weight='BOLD').move_to([3.6, 2.85, 0])
            raw = [envelope_curve(lp, t, D[f's1_raw_{k}'], c, 2.0, 700) for k, c in zip(KEYS, MUSCLE_C)]
            nrm = [envelope_curve(rp, t, D[f's1_{k}'], c, 2.0, 700) for k, c in zip(KEYS, MUSCLE_C)]
            legend = legend_row().move_to([0, -2.95, 0])
            self.play(FadeIn(lp), FadeIn(ttl_l), run_time=0.5)
            self.play(*[Create(c) for c in raw], FadeIn(legend), run_time=1.5)
            self.until(self.cue('So that the network'))
            arrow = VGroup(Arrow([-0.95, 0.35, 0], [0.2, 0.35, 0], buff=0, color=INK, stroke_width=4, max_tip_length_to_length_ratio=0.35),
                           T('÷ max', 22, INK).move_to([-0.37, 0.8, 0]))
            self.play(FadeIn(rp), FadeIn(ttl_r), FadeIn(arrow), run_time=0.8)
            self.until(self.cue('the baseline divides'))
            refs = VGroup(*[lp.hline(m, c, 2.0, True) for m, c in zip(EG_MAX, MUSCLE_C)])
            note = T('dashed: largest value over all 8 session-1 trials', 20, DIM).move_to([-3.4, -2.35, 0])
            foot = T('in the code this division happens at 1000 samples/s, before the trim and the downsampling', 18,
                     DIM).move_to([0, -3.5, 0])
            self.play(Create(refs), FadeIn(note), run_time=1.2)
            self.play(FadeIn(foot), run_time=0.6)
            self.until(self.cue('Across session 1'))
            one = rp.hline(1.0, INK, 2.0, True)
            one_l = T('= session-1 maximum', 18, INK).next_to(rp.p(90, 1.0), UP, buff=0.07).align_to(rp.p(90, 1.0), RIGHT)
            self.play(Create(one), FadeIn(one_l), run_time=0.8)
            self.until(self.cue('A single trial'))
            self.play(*[TransformFromCopy(a, b) for a, b in zip(raw, nrm)], run_time=1.6)
            pk = VGroup(T("this trial's peaks:", 20, DIM),
                        *[T(f'{n} {fmt(v)}', 20, c) for n, v, c in zip(NAMES, PEAKS, MUSCLE_C)])
            pk.arrange(RIGHT, buff=0.22).move_to([3.2, -2.35, 0])
            if pk.get_right()[0] > 6.6:
                pk.shift(LEFT * (pk.get_right()[0] - 6.6))
            self.play(FadeIn(pk), run_time=0.6)

        self.clear_all(keep=[tag])

        # ── b02: session 2 is divided by the same numbers; the size can change ──────────────────────────────
        with self.beat('c05_b02'):
            chip = card([T('session-1 maximum, per muscle', 24, C_S1)], width=5.0, color=C_S1, pad=0.22)
            chip.move_to([0, 2.4, 0])
            b1 = card([T('Session 1 EMG', 24, C_S1, weight='BOLD'), T('÷ session-1 maximum', 24)], width=4.6,
                      color=C_S1).move_to([-3.4, 0.5, 0])
            b2 = card([T('Session 2 EMG', 24, C_S2, weight='BOLD'), T('÷ the same numbers', 24)], width=4.6,
                      color=C_S2).move_to([3.4, 0.5, 0])
            a1 = Arrow(chip.get_bottom(), b1.get_top(), buff=0.1, color=C_S1, stroke_width=4)
            a2 = Arrow(chip.get_bottom(), b2.get_top(), buff=0.1, color=C_S2, stroke_width=4)
            honest = T('nothing from the session being tested is used', 22, DIM).next_to(b2, DOWN, buff=0.35)
            self.play(FadeIn(chip), run_time=0.5)
            self.play(FadeIn(b1), FadeIn(b2), Create(a1), Create(a2), run_time=1.0)
            self.until(self.cue('That is the honest'))
            self.play(FadeIn(honest), run_time=0.6)
            self.until(self.cue('For the same push'))
            self.play(FadeOut(VGroup(chip, b1, b2, a1, a2, honest)), run_time=0.5)
            # same push, different signal size
            panels = VGroup()
            curves = VGroup()
            for cx, col, name, eh in ((-3.4, C_S1, 'Session 1', 1.3), (3.4, C_S2, 'Session 2', 0.5)):
                panels.add(T(name, 26, col, weight='BOLD').move_to([cx, 2.55, 0]),
                           Line([cx - 2.5, 0.55, 0], [cx + 2.5, 0.55, 0], stroke_color=FAINT, stroke_width=1.5),
                           Line([cx - 2.5, -1.6, 0], [cx + 2.5, -1.6, 0], stroke_color=FAINT, stroke_width=1.5))
                curves.add(bump(cx, 0.55, 3.2, 1.3, C_TQ), bump(cx, -1.6, 3.2, eh, C_GM))
            lab_t = T('torque', 20, C_TQ).move_to([-5.9, 1.9, 0], aligned_edge=LEFT)
            lab_e = T('EMG size', 20, C_GM).move_to([-5.9, -0.3, 0], aligned_edge=LEFT)
            ref = DashedVMobject(Line([0.9, -1.6 + 1.3, 0], [5.9, -1.6 + 1.3, 0], stroke_color=C_S1, stroke_width=2),
                                 num_dashes=24)
            ref_l = T('session-1 size', 18, C_S1).next_to(ref, UP, buff=0.12).align_to(ref, RIGHT)
            illus = T('illustration', 18, DIM).move_to([6.2, 2.55, 0], aligned_edge=RIGHT)
            self.play(FadeIn(panels), FadeIn(illus), Create(curves[0]), Create(curves[1]), FadeIn(lab_t), FadeIn(lab_e), run_time=1.0)
            self.play(Create(curves[2]), Create(curves[3]), Create(ref), FadeIn(ref_l), run_time=1.0)
            nk = T('same push, different EMG size: the model has no way of knowing', 24, C_HI).move_to([0, -2.8, 0])
            self.play(FadeIn(nk), run_time=0.6)
            self.until(self.cue('Each bar here'))
            self.play(FadeOut(VGroup(panels, curves, lab_t, lab_e, ref, ref_l, nk, illus)), run_time=0.5)
            # the bar chart
            ch = bars(JM_RATIO, NAMES, MUSCLE_C, width=7.2, height=3.4, vmax=1.2, label_size=24, value_size=28)
            ch.shift(DOWN * 2.1)
            for v, b in zip(ch.vals, ch.bars):
                v.set_color(BG).move_to(b.get_top() + DOWN * 0.3)
            title = T('EMG size per newton metre of push', 28, INK, weight='BOLD').move_to([0, 3.0, 0])
            sub = T('session 2 ÷ session 1 · JM · position −0.36 rad', 24, DIM).move_to([0, 2.5, 0])
            y1 = ch.axis.get_center()[1] + 1.0 / 1.2 * 3.4
            one = DashedVMobject(Line([-3.8, y1, 0], [3.8, y1, 0], stroke_color=INK, stroke_width=2), num_dashes=40)
            one_l = T('1 = same as session 1', 22, INK).next_to([3.8, y1, 0], UP, buff=0.08)
            one_l.shift(RIGHT * (3.8 - one_l.get_right()[0]))
            self.play(FadeIn(title), FadeIn(sub), FadeIn(ch.axis), FadeIn(ch.labels), run_time=0.7)
            self.play(Create(one), FadeIn(one_l), run_time=0.6)
            self.play(*[GrowFromEdge(b, DOWN) for b in ch.bars], FadeIn(ch.vals), run_time=1.4)
            self.until(self.cue('For JM at'))
            boxes = VGroup(*[SurroundingRectangle(ch.bars[i], color=C_HI, buff=0.08, stroke_width=3)
                             for i in (0, 2)])
            thirds = VGroup(*[T('≈ 1/3', 22, BG, weight='BOLD').move_to(ch.bars[i].get_top() + DOWN * 0.7) for i in (0, 2)])
            self.play(Create(boxes), FadeIn(thirds), run_time=0.8)
            self.until(self.cue('GL drops less'))
            fifth = T('≈ 4/5', 22, BG, weight='BOLD').move_to(ch.bars[1].get_top() + DOWN * 0.7)
            gl_box = SurroundingRectangle(ch.bars[1], color=C_HI, buff=0.08, stroke_width=3)
            self.play(Create(gl_box), FadeIn(fifth), run_time=0.8)

        self.clear_all(keep=[tag])

        # ── b03: two small notes ────────────────────────────────────────────────────────────────────────────
        with self.beat('c05_b03'):
            # note 1: one single sample
            k = F['y02']['eg_S1_trial']
            sp = Plot((0, 90), (0, 0.1), 5.0, 1.8, xticks=[0, 30, 60, 90], yticks=[0, 0.05], xlabel='time in trial (s)',
                      ylabel='envelope (mV)', yfmt=lambda v: f'{v:g}', tick_size=18).move_to([-3.8, 1.35, 0])
            smax = sp.hline(EG_MAX[2], C_SOL, 2.0, True)          # session-1 maximum of EG's soleus (trial 17, 23.65 s)
            smax_l = T('session-1 maximum (one sample, another trial)', 18, C_SOL
                       ).next_to(sp.p(90, EG_MAX[2]), UP, buff=0.07).align_to(sp.p(90, EG_MAX[2]), RIGHT)
            cs = envelope_curve(sp, D['s1_t'], D['s1_raw_sol'], C_SOL, 2.0, 700)
            ipk = int(np.argmax(D['s1_raw_sol']))
            dot = Dot(sp.p(D['s1_t'][ipk], D['s1_raw_sol'][ipk]), radius=0.1, color=C_HI)
            head1 = T('Each maximum is one single sample', 26, INK, weight='BOLD').move_to([-3.5, 2.95, 0])
            ex = T('example: one trial of EG, soleus', 18, DIM).move_to([-3.8, -0.3, 0])
            dot_l = T("this trial's peak, below the maximum", 18, C_HI).next_to(dot, UP, buff=0.07)
            self.play(FadeIn(head1), FadeIn(sp), run_time=0.6)
            self.play(Create(cs), run_time=1.0)
            self.play(FadeIn(dot, scale=2.5), FadeIn(ex), FadeIn(dot_l), Create(smax), FadeIn(smax_l), run_time=0.6)
            self.until(self.cue('And for five'))
            # note 2: the 6 x 4 grid, 5 cells in the validation part
            subs = [str(s) for s in D['emg_max_subjects']]
            cw, chh = 0.95, 0.46
            gx, gy = 3.0, 2.25
            grid = VGroup()
            cells = {}
            for j, n in enumerate(NAMES):
                grid.add(T(n, 20, MUSCLE_C[j], weight='BOLD').move_to([gx + j * cw, gy + 0.5, 0]))
            for i, s in enumerate(subs):
                grid.add(T(s, 20, INK).move_to([gx - 0.95 - 0.1, gy - i * (chh + 0.06), 0], aligned_edge=RIGHT))
                for j, kk in enumerate(KEYS):
                    inval = (s, kk) in VAL_PAIRS
                    r = Rectangle(width=cw - 0.08, height=chh, stroke_width=0, fill_color=C_VAL if inval else PANEL,
                                  fill_opacity=0.95 if inval else 1).move_to([gx + j * cw, gy - i * (chh + 0.06), 0])
                    cells[(s, kk)] = r
                    grid.add(r)
            head2 = T('Where the maximum lies', 26, INK, weight='BOLD').move_to([3.6, 3.15, 0])
            self.play(FadeIn(head2), FadeIn(grid), run_time=0.8)
            vals = [cells[(s, kk)] for (s, kk) in sorted(VAL_PAIRS, key=lambda p: (subs.index(p[0]), KEYS.index(p[1])))]
            self.play(*[Indicate(v, color=C_HI, scale_factor=1.15) for v in vals], run_time=1.2)
            leg = VGroup(Rectangle(width=0.3, height=0.3, fill_color=C_VAL, fill_opacity=1, stroke_width=0),
                         T('in the validation part: 5 of 24 pairs', 20, C_VAL)).arrange(RIGHT, buff=0.15)
            leg.move_to([3.4, -1.15, 0])
            self.play(FadeIn(leg), run_time=0.5)
            self.until(self.cue('And the fifth input'))
            chips = VGroup()
            for n, c in zip(NAMES, MUSCLE_C):
                chips.add(card([T(n, 24, c, weight='BOLD'), T('÷ max', 22)], width=1.8, color=c, pad=0.15))
            chips.add(card([T('angle', 24, C_ANGLE, weight='BOLD'), T('raw, in rad', 22)], width=2.2, color=C_ANGLE, pad=0.15))
            chips.arrange(RIGHT, buff=0.3).move_to([0, -2.55, 0])
            chips_l = T('the five inputs of the network', 20, DIM).next_to(chips, UP, buff=0.2)
            self.play(FadeIn(chips, shift=UP * 0.2), FadeIn(chips_l), run_time=0.8)
