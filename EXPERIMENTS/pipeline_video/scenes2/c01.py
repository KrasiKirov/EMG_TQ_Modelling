"""c01 | A puzzle: a good-looking prediction with a negative R², the road map and the ground rule."""
from manim import *
from style2 import *

D3 = load('y03')
F3 = facts()['y03']
F1 = facts()['y01']

SEEDS = list(D3['seeds'])
J_YES, J_POS = 5, 0                                     # YES, position -0.486 rad
SEED = 1
R2 = F3['YES_minus0p486_seed1']['r2']                   # -0.2626
MEAN_TRUTH = F3['YES_minus0p486_seed1']['mean_truth_nm']
N_NEG = int(D3['n_seeds_negative'][J_YES][J_POS])       # 9 of 10
N_SEEDS = len(SEEDS)
T0, T1 = [float(v) for v in D3['window_yes_m486_start_end_s']]
_tw = D3['t_window_s']
_w = (_tw >= T0) & (_tw <= T1)
TW = D3['t_trial_s'][_w]                                  # time since trial start (window index x 0.01 s + 0.19 s)
T0T, T1T = float(TW[0]), float(TW[-1])
KEY = D3['yes_m486_key'][_w]
PRED = D3['yes_m486_pred'][SEEDS.index(SEED)][_w]
POS_YES = float(D3['yes_m486_position'])
GS = F1['B_group_summaries']['X00_baseline']
R2_DOWN = GS['toes_down_12']['r2_raw_median']           # 0.77
R2_OTHER = GS['other_36']['r2_raw_median']              # 0.97


class Ch01(BeatScene):
    def construct(self):
        # ── b01: the prediction that looks good ──────────────────────────────
        with self.beat('c01_b01'):
            t1 = T('EMG → ankle torque:', 54, INK, weight='BOLD')
            t2 = T('what the baseline really does', 54, INK, weight='BOLD')
            sub = T('and why it struggles with the toes down', 32, DIM)
            card_t = VGroup(t1, t2, sub).arrange(DOWN, buff=0.3)
            card_t[2].shift(DOWN * 0.15)
            self.play(FadeIn(card_t, shift=UP * 0.2), run_time=0.7)
            self.wait(0.6)
            self.play(FadeOut(card_t), run_time=0.5)
            tag = self.tag_ref = self.intro('c01', hold=0.4)

            plot = Plot((T0T, T1T), (-4.4, 2.0), 10.6, 4.1, xticks=range(15, 45, 5), yticks=(-4, -3, -2, -1, 0),
                        xlabel='time in trial (s)', ylabel='torque (Nm)', tick_size=18, grid=True)
            plot.move_to([0.5, -0.65, 0])
            title = M(f'<b>YES</b> · <span foreground="{C_S2}">session 2</span> · position '
                      f'{fmt(POS_YES, 3)} rad (toes down)', 26).move_to([0.5, 2.85, 0])
            key_c = envelope_curve(plot, TW, KEY, C_TQ, 2.6, 1500)
            pred_c = envelope_curve(plot, TW, PRED, C_PRED, 2.6, 1500)

            def legend_row(color, text, dashed=False):
                ln = Line(LEFT * 0.3, RIGHT * 0.3, stroke_color=color, stroke_width=4)
                if dashed:
                    ln = DashedVMobject(ln, num_dashes=5)
                return VGroup(ln, T(text, 20, INK)).arrange(RIGHT, buff=0.2)
            leg_key = legend_row(C_TQ, 'answer key')
            leg_pred = legend_row(C_PRED, f'baseline prediction, seed {SEED}')
            leg_flat = legend_row(C_HI, 'flat line at the true average', dashed=True)
            legend = VGroup(leg_key, leg_pred, leg_flat).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
            legend.move_to(plot.p(T0T, 2.0) + RIGHT * 0.35 + DOWN * 0.1, aligned_edge=UL)
            self.legend, self.plot, self.title_c01 = legend, plot, title

            self.play(FadeIn(title), FadeIn(plot), run_time=0.7)
            self.until(self.cue('The white line'))
            self.play(Create(key_c), FadeIn(leg_key, shift=RIGHT * 0.2), run_time=3.2, rate_func=linear)
            self.until(self.cue('The green line'))
            self.play(Create(pred_c), FadeIn(leg_pred, shift=RIGHT * 0.2), run_time=3.2, rate_func=linear)
            # one push, one relaxed stretch
            i_push = int(np.argmin(PRED[: len(PRED) // 2]))
            self.until(self.cue('It moves with every push'))
            push_lab = T('push', 22, C_HI)
            push_lab.next_to(plot.p(TW[i_push], KEY[i_push]), DOWN, buff=0.3)
            push_arr = Arrow(push_lab.get_top(), plot.p(TW[i_push], KEY[i_push]) + DOWN * 0.07, buff=0.04,
                             stroke_width=3, color=C_HI, max_tip_length_to_length_ratio=0.35)
            self.play(FadeIn(push_lab), GrowArrow(push_arr), run_time=0.6)
            self.until(self.cue('comes back when they relax'))
            lo = int(np.argmax(PRED[len(PRED) // 3: 2 * len(PRED) // 3])) + len(PRED) // 3
            rel_lab = T('relax', 22, C_HI)
            rel_lab.next_to(plot.p(TW[lo], PRED[lo]), UP, buff=0.3)
            rel_arr = Arrow(rel_lab.get_bottom(), plot.p(TW[lo], PRED[lo]) + UP * 0.07, buff=0.04, stroke_width=3,
                            color=C_HI, max_tip_length_to_length_ratio=0.35)
            self.play(FadeIn(rel_lab), GrowArrow(rel_arr), run_time=0.6)
            self.until(self.cue('It looks pretty good'))
            self.play(FadeOut(VGroup(push_lab, push_arr, rel_lab, rel_arr)), run_time=0.5)
            self.keep = VGroup(key_c, pred_c)

        # ── b02: R² is negative ──────────────────────────────────────────────
        with self.beat('c01_b02'):
            plot, legend = self.plot, self.legend
            flat = plot.hline(MEAN_TRUTH, C_HI, 2.4)
            self.until(self.cue('worse than a flat line'))
            self.play(Create(flat), FadeIn(legend[2], shift=RIGHT * 0.2), run_time=1.0)
            badge_box = RoundedRectangle(corner_radius=0.16, width=4.3, height=1.4, fill_color=PANEL,
                                         fill_opacity=1, stroke_color=C_BAD, stroke_width=3)
            badge_main = T(f'R² = {fmt(R2)}', 40, C_BAD, weight='BOLD')
            badge_sub = T('whole trial, this run', 20, INK)
            badge = VGroup(badge_box, VGroup(badge_main, badge_sub).arrange(DOWN, buff=0.14)).move_to(
                plot.p(T1T, 2.0) + LEFT * 2.4 + DOWN * 0.85)
            badge[1].move_to(badge_box)
            self.until(self.cue('Its R² is negative'))
            self.play(FadeIn(badge, shift=DOWN * 1.2), run_time=0.8, rate_func=rush_from)
            below = T(f'{N_NEG} of the {N_SEEDS} runs at this position are below zero', 20, INK)
            below.next_to(badge, DOWN, buff=0.15).align_to(badge, RIGHT)
            self.until(self.cue('and lower still in most other runs'))
            self.play(FadeIn(below), run_time=0.6)

            # toes-down vs the rest
            self.until(self.cue('Low scores like this turn up'))
            self.clear_all(0.6, keep=(self.tag_ref,))
            vals = [R2_DOWN, R2_OTHER]
            ch = bars(vals, ['toes-down positions', 'all other positions'], [ANGLE_C[0], ANGLE_C[4]], width=5.4,
                      height=2.7, vmax=1.0, value_fmt=lambda v: fmt(v), label_size=22, value_size=34, gap=0.8)
            ch.move_to([-2.3, -0.2, 0])
            ttl = T('typical R² of the baseline', 28, INK).move_to([-2.3, 2.55, 0])
            self.play(FadeIn(ttl), run_time=0.4)
            self.play(*[GrowFromEdge(b, DOWN) for b in ch.bars], FadeIn(ch.axis), FadeIn(ch.labels),
                      LaggedStart(*[FadeIn(v) for v in ch.vals], lag_ratio=0.4), run_time=1.6)
            self.until(self.cue('By the end of this video'))
            q = T('?', 190, C_HI, weight='BOLD').move_to([3.9, 0.1, 0])
            why = T('why?', 34, C_HI).next_to(q, DOWN, buff=0.1)
            self.play(FadeIn(q, scale=0.6), FadeIn(why), run_time=0.8)

        # ── b03: the road map ────────────────────────────────────────────────
        with self.beat('c01_b03'):
            self.clear_all(0.5, keep=(self.tag_ref,))
            blocks = [
                ('1', 'The experiment and the data', None),
                ('2', 'Inputs: cleaning, cutting, splitting, feeding', None),
                ('3', 'Training and scoring', None),
                ('4', 'Why toes-down is hard', 'bugs · potential techniques · the papers · next steps'),
            ]
            cols = [FAINT, FAINT, FAINT, FAINT]
            rows = VGroup()
            for (n, head, subt), col in zip(blocks, cols):
                h = 1.35 if subt else 1.0
                box = RoundedRectangle(corner_radius=0.16, width=11.4, height=h, fill_color=PANEL, fill_opacity=1,
                                       stroke_color=col, stroke_width=2.5)
                circ = VGroup(Circle(radius=0.32, fill_color=C_HI, fill_opacity=1, stroke_width=0),
                              T(n, 30, BG, weight='BOLD'))
                txt = VGroup(T(head, 30, INK, weight='BOLD'))
                if subt:
                    txt.add(T(subt, 24, DIM))
                txt.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
                circ.move_to(box.get_left() + RIGHT * 0.65)
                txt.next_to(circ, RIGHT, buff=0.35)
                rows.add(VGroup(box, circ, txt))
            rows.arrange(DOWN, buff=0.3).move_to([0, -0.25, 0])
            cues = ['First, the experiment', 'Second, how', 'Third, how', 'And fourth']
            for row, c in zip(rows, cues):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.4), run_time=0.7)
            self.until(self.cue('That last block also covers'))
            self.play(rows[3][0].animate.set_stroke(C_HI, 4), run_time=0.5)

        # ── b04: the ground rule ─────────────────────────────────────────────
        with self.beat('c01_b04'):
            self.clear_all(0.5, keep=(self.tag_ref,))
            box = RoundedRectangle(corner_radius=0.18, width=12.4, height=6.2, fill_color=PANEL, fill_opacity=1,
                                   stroke_color=C_HI, stroke_width=2.5).move_to([0, -0.3, 0])
            head = T('Ground rule', 34, C_HI, weight='BOLD')
            head.move_to(box.get_top() + DOWN * 0.55).align_to(box, LEFT).shift(RIGHT * 0.5)
            line1 = VGroup(M('<b>The baseline</b> = last semester\'s model,', 30),
                           T('the reference for everything else', 30)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            line1.next_to(head, DOWN, buff=0.4).align_to(head, LEFT)
            b2 = bullets(['Code as found in the working copy, frozen, rerun ten times with fixed seeds',
                          'Most numbers come from those saved runs; when one does not, we say so',
                          'Where the code does something odd, we say so'], size=26, chars=64)
            b2.next_to(line1, DOWN, buff=0.45).align_to(head, LEFT)
            chips = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.07, width=0.56, height=0.5,
                                                     fill_color=BG, fill_opacity=1, stroke_color=FAINT, stroke_width=2),
                                    T(str(int(s)), 22, INK)) for s in SEEDS]).arrange(RIGHT, buff=0.14)
            for c in chips:
                c[1].move_to(c[0])
            seed_lab = T('seeds', 22, DIM)
            seeds_row = VGroup(seed_lab, chips).arrange(RIGHT, buff=0.3)
            seeds_row.move_to(box.get_bottom() + UP * 0.6)
            self.until(self.cue('One ground rule'))
            self.play(FadeIn(box), FadeIn(head), run_time=0.6)
            self.play(FadeIn(line1, shift=UP * 0.15), run_time=0.7)
            self.until(self.cue("Until the potential techniques"))
            self.play(FadeIn(b2[0], shift=UP * 0.15), run_time=0.6)
            self.until(self.cue('ten times with fixed seeds'))
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in chips], lag_ratio=0.08), FadeIn(seed_lab),
                      run_time=1.2)
            self.until(self.cue('Most numbers on screen'))
            self.play(FadeIn(b2[1], shift=UP * 0.15), run_time=0.6)
            self.until(self.cue('And where the code does something odd'))
            self.play(FadeIn(b2[2], shift=UP * 0.15), run_time=0.6)
