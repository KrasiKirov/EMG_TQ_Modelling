"""Chapter 14: Why toes-down is hard (five reasons)."""
from manim import *
from style2 import *

DEG = np.pi / 180


def reason_header(n, title):
    circ = Circle(radius=0.27, fill_color=C_HI, fill_opacity=1, stroke_width=0)
    num = T(str(n), 26, BG, weight='BOLD').move_to(circ)
    ttl = T(title, 34, INK, weight='BOLD')
    row = VGroup(VGroup(circ, num), ttl).arrange(RIGHT, buff=0.25)
    row.align_to([-6.5, 0, 0], LEFT).set_y(2.8)
    pips = VGroup(*[Circle(radius=0.11, fill_color=C_HI if i + 1 == n else FAINT, fill_opacity=1, stroke_width=0)
                    for i in range(5)]).arrange(RIGHT, buff=0.18)
    pips.align_to([6.5, 0, 0], RIGHT).set_y(2.8)
    return VGroup(row, pips)


def tick_mark(size=0.3, color=C_GOOD):
    pts = [np.array([-0.5, 0.0, 0]) * size, np.array([-0.12, -0.42, 0]) * size, np.array([0.55, 0.5, 0]) * size]
    return VMobject(stroke_color=color, stroke_width=8).set_points_as_corners(pts)


def runs(mask):
    """Start/stop index pairs of the True runs in a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def yes_panel(sess, w, h, ylabel=True):
    """YES at -0.486 rad: measured torque over the trial, the passive-table value and the relaxed level."""
    d = load('y05')
    disp = facts()['y05']['display']['yes_m049']
    table, relaxed = disp['table'][sess - 1], disp['relaxed_rest_median'][sess - 1]
    t, tq, mask = d[f'yesS{sess}_t'], d[f'yesS{sess}_tq'], d[f'yesS{sess}_rest_mask']
    pl = Plot((0, 90), (2, 12), w, h, xticks=(0, 30, 60, 90), yticks=(2, 4, 6, 8, 10, 12),
              xlabel='time in trial (s)', ylabel='torque (Nm)' if ylabel else '', tick_size=18)
    gap = pl.band(0, 90, C_BAD, 0.30, y0=relaxed, y1=table)
    trace = envelope_curve(pl, t, tq, C_TQ, 1.6)
    rest = VGroup(*[pl.curve(t[a:b], tq[a:b], C_GOOD, 4.5) for a, b in runs(mask) if b - a > 5])
    tline = pl.hline(table, C_PASSIVE, 3.0)
    rline = pl.hline(relaxed, C_GOOD, 3.0, dashed=False)
    tlab = T(f'{table:.1f}', 20, C_PASSIVE, weight='BOLD').next_to(pl.p(90, table), RIGHT, buff=0.12)
    rlab = T(f'{relaxed:.1f}', 20, C_GOOD, weight='BOLD').next_to(pl.p(90, relaxed), RIGHT, buff=0.12)
    g = VGroup(pl, gap, trace, rest, tline, rline, tlab, rlab)
    g.pl, g.gap, g.trace, g.rest, g.tline, g.rline, g.tlab, g.rlab = pl, gap, trace, rest, tline, rline, tlab, rlab
    g.table, g.relaxed = table, relaxed
    return g


def leg_toes_down(cx, top):
    """Simple side view: shank, foot pointing down, calf muscle drawn short (dashed = its length at rest)."""
    shank = RoundedRectangle(corner_radius=0.2, width=0.5, height=1.9, fill_color='#3A4350', fill_opacity=1,
                             stroke_width=0).move_to([cx, top - 0.95, 0])
    ankle = np.array([cx, top - 1.9, 0])
    d = np.array([np.cos(-55 * DEG), np.sin(-55 * DEG), 0])
    foot = RoundedRectangle(corner_radius=0.15, width=1.5, height=0.32, fill_color='#3A4350', fill_opacity=1,
                            stroke_width=0).rotate(-55 * DEG).move_to(ankle + d * 0.7)
    knee = Circle(radius=0.2, fill_color='#3A4350', fill_opacity=1, stroke_width=0).move_to([cx, top, 0])
    ghost = DashedVMobject(Ellipse(width=0.6, height=1.5, stroke_color=DIM, stroke_width=2), num_dashes=24) \
        .move_to([cx - 0.42, top - 0.85, 0])
    calf = Ellipse(width=0.6, height=0.8, fill_color=C_GM, fill_opacity=0.95, stroke_width=0) \
        .move_to([cx - 0.42, top - 0.85, 0])
    achilles = Line(calf.get_bottom(), ankle + LEFT * 0.12 + DOWN * 0.05, stroke_color=C_GM, stroke_width=3)
    return VGroup(ghost, shank, knee, foot, achilles, calf), calf, foot


class Ch14(BeatScene):
    def construct(self):
        F = facts()
        y3 = load('y03')
        sd, r2, td = y3['sd_truth'], y3['r2_med'], y3['toes_down']
        subj, pos = list(y3['subjects']), y3['positions_rad']

        # ── b01: the scatter ────────────────────────────────────────────────
        with self.beat('c14_b01'):
            tag = self.intro('c14')
            title = T('Five reasons the baseline struggles', 34, INK, weight='BOLD').align_to([-6.5, 0, 0], LEFT).set_y(2.8)
            self.play(FadeIn(title), run_time=0.5)
            self.until(self.cue('There are five'))
            pips = VGroup(*[VGroup(Circle(radius=0.22, fill_color=FAINT, fill_opacity=1, stroke_width=0),
                                   T(str(i + 1), 22, INK, weight='BOLD')) for i in range(5)])
            for p in pips:
                p[1].move_to(p[0])
            pips.arrange(RIGHT, buff=0.2).align_to([6.5, 0, 0], RIGHT).set_y(2.8)
            self.play(LaggedStart(*[FadeIn(p, scale=0.6) for p in pips], lag_ratio=0.15), run_time=0.9)

            lo, hi = np.log10(0.3), np.log10(40)
            pl = Plot((lo, hi), (-1, 1.05), 9.0, 4.0, xticks=[np.log10(v) for v in (0.5, 1, 2, 5, 10, 20)],
                      yticks=(-1, -0.5, 0, 0.5, 1), xlabel='spread of the true torque in the cell (Nm, log scale)',
                      ylabel='baseline R² (median over seeds)', tick_size=18, xfmt=lambda v: f'{10 ** v:.2g}',
                      yfmt=lambda v: f'{v:g}')
            pl.move_to([0.45, -0.45, 0])
            base = VGroup(pl.ax, pl.frame, pl.ticks)
            zero = pl.hline(0, FAINT, 1.5)
            band = pl.band(lo, np.log10(2.1), C_BAD, 0.12)
            dots_o, dots_t = VGroup(), VGroup()
            jm_pt = None
            for i in range(6):
                for j in range(8):
                    x, y = np.log10(sd[i, j]), r2[i, j]
                    if y < -1:
                        jm_pt = (x, i, j)
                        continue
                    if td[i, j]:
                        dots_t.add(Dot(pl.p(x, y), radius=0.1, color=C_HI, stroke_color=BG, stroke_width=1.5))
                    else:
                        dots_o.add(Dot(pl.p(x, y), radius=0.07, color=C_ANGLE, fill_opacity=0.75))
            self.until(self.cue('First, a picture'))
            self.play(FadeIn(base), Create(zero), run_time=0.6)
            self.until(self.cue('Every dot here'))
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots_o], lag_ratio=0.03),
                      LaggedStart(*[FadeIn(d, scale=0.5) for d in dots_t], lag_ratio=0.06), run_time=1.6)
            # JM arrow off the bottom
            jx = jm_pt[0]
            arr = Arrow(pl.p(jx, -0.5), pl.p(jx, -1.0), buff=0, color=C_BAD, stroke_width=5, max_tip_length_to_length_ratio=0.35)
            jm_lab = T(f'JM {pos[jm_pt[1], jm_pt[2]]:.3f} rad: R² {fmt(r2[jm_pt[1], jm_pt[2]], 1)}'.replace('-', '−'),
                       18, C_BAD).next_to(arr, RIGHT, buff=0.12).shift(UP * 0.1)
            self.play(Create(arr), FadeIn(jm_lab), run_time=0.6)
            leg = VGroup(VGroup(Dot(radius=0.1, color=C_HI), T('toes-down cells (12)', 20, INK)).arrange(RIGHT, buff=0.15),
                         VGroup(Dot(radius=0.07, color=C_ANGLE), T('other cells (36)', 20, INK)).arrange(RIGHT, buff=0.15)
                         ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to(pl.p(np.log10(8), -0.45))
            self.play(FadeIn(leg), run_time=0.4)
            self.until(self.cue('with the spread'))
            self.play(FadeIn(pl.xlabel), run_time=0.5)
            self.until(self.cue('up the side'))
            self.play(FadeIn(pl.ylabel), run_time=0.5)
            self.until(self.cue('The low scores'))
            lab = T('little to predict', 22, C_BAD).next_to(band, UP, buff=0.05, aligned_edge=LEFT).shift(RIGHT * 0.1)
            lab.move_to([pl.p(lo, 1.05)[0] + 1.2, pl.p(lo, 1.05)[1] + 0.25, 0])
            sp = F['y03']['scatter_summary']['spearman_sd_vs_r2_all']
            rho = T(f'wider spread, higher score: rank correlation {sp:.2f}', 20, DIM).move_to([0.45, -3.5, 0])
            self.play(FadeIn(band), FadeIn(lab), FadeIn(rho), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b02: reason 1, little to predict ────────────────────────────────
        with self.beat('c14_b02'):
            hdr = reason_header(1, 'Little to predict')
            self.play(FadeIn(hdr), run_time=0.5)
            c = lambda n, i: y3['sd_truth'][subj.index(n), i]
            iyes = int(np.argmin(np.abs(pos[subj.index('YES')] + 0.486)))
            ijm, ieg = 0, 0
            vals = [sd[subj.index('YES'), iyes], sd[subj.index('JM'), ijm], sd[subj.index('EG'), ieg],
                    F['y03']['groups']['other_36']['sd_truth_median_nm'] if 'other_36' in F['y03']['groups'] else 7.0,
                    F['y03']['scatter_summary']['sd_truth_median_of_each_persons_strongest_cell']]
            fm = [lambda v: f'{v:.2f} Nm', lambda v: f'{v:.2f} Nm', lambda v: f'{v:.1f} Nm',
                  lambda v: f'{v:.1f} Nm', lambda v: f'≈ {v:.0f} Nm']
            labs = [('YES', fmt(pos[subj.index("YES"), iyes], 3) + ' rad'), ('JM', fmt(pos[subj.index("JM"), ijm], 3) + ' rad'),
                    ('EG', fmt(pos[subj.index("EG"), ieg], 3) + ' rad'), ('typical', 'other position'),
                    ('strongest', 'per person (3–28)')]
            cols = [ANGLE_C[0]] * 3 + [C_PASSIVE, DIM]
            width, bh, vmax = 6.2, 2.4, 13.0
            bw = (width - 0.28 * 4) / 5
            bx0, base_y = -3.35, -1.4
            bars_g, vals_g, labs_g = VGroup(), VGroup(), VGroup()
            for k in range(5):
                x = bx0 - width / 2 + bw / 2 + k * (bw + 0.28)
                h = max(0.04, vals[k] / vmax * bh)
                r = Rectangle(width=bw, height=h, fill_color=cols[k], fill_opacity=0.92, stroke_width=0).move_to([x, base_y + h / 2, 0])
                bars_g.add(r)
                vals_g.add(T(fm[k](vals[k]), 20, INK, weight='BOLD').next_to(r, UP, buff=0.08))
                l1 = T(labs[k][0], 18, INK if k < 3 else DIM)
                l2 = T(labs[k][1], 18, DIM)
                if k >= 3:
                    l2 = W(labs[k][1], 18, DIM, width=10, align=ORIGIN)
                labs_g.add(VGroup(l1, l2).arrange(DOWN, buff=0.04).next_to([x, base_y, 0], DOWN, buff=0.12))
            axis = Line([bx0 - width / 2 - 0.1, base_y, 0], [bx0 + width / 2 + 0.1, base_y, 0], stroke_color=FAINT, stroke_width=1.6)
            btitle = T('spread of the true torque (standard deviation)', 20, DIM).move_to([bx0, 1.85, 0])
            self.until(self.cue('At the toes-down positions'))
            self.play(FadeIn(axis), FadeIn(btitle), run_time=0.4)
            self.play(GrowFromEdge(bars_g[0], DOWN), GrowFromEdge(bars_g[1], DOWN), FadeIn(vals_g[0]), FadeIn(vals_g[1]),
                      FadeIn(labs_g[0]), FadeIn(labs_g[1]), run_time=0.8)
            self.until(self.cue("and at EG's worst"))
            self.play(GrowFromEdge(bars_g[2], DOWN), FadeIn(vals_g[2]), FadeIn(labs_g[2]), run_time=0.7)
            self.until(self.cue('At a typical other position'))
            self.play(GrowFromEdge(bars_g[3], DOWN), FadeIn(vals_g[3]), FadeIn(labs_g[3]), run_time=0.7)
            self.until(self.cue("and at each person's strongest"))
            self.play(GrowFromEdge(bars_g[4], DOWN), FadeIn(vals_g[4]), FadeIn(labs_g[4]), run_time=0.7)
            self.until(self.cue('With the toes down'))
            fig, calf, foot = leg_toes_down(1.6, 1.9)
            fig_lab = T('toes down: calf already short', 20, DIM).move_to([3.0, -1.55, 0])
            weak = W('short muscle → weak: a maximum effort here reaches only 15–28 % of the person\'s peak', 22, INK,
                     width=chars_for(3.1, 22)).move_to([5.0, 1.0, 0])
            self.play(FadeIn(fig, shift=DOWN * 0.2), FadeIn(fig_lab), run_time=0.7)
            self.play(FadeIn(weak), run_time=0.6)
            self.until(self.cue('Small pushes also'))
            emg = W('small pushes → small EMG signals, closer to the noise', 22, INK, width=chars_for(5.6, 22)) \
                .move_to([3.85, -2.3, 0])
            self.play(FadeIn(emg), run_time=0.6)
            self.until(self.cue('As we just saw'))
            ban = M('does not break the model inside session 1  ·  <span foreground="#F0E442">makes every later error look much bigger</span>',
                    22).move_to([0, -3.35, 0])
            self.play(FadeIn(ban), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b03: reason 2, offsets in the answer key (session 1) ────────────
        with self.beat('c14_b03'):
            hdr = reason_header(2, 'The answer key has offsets')
            self.play(FadeIn(hdr), run_time=0.5)
            P1 = yes_panel(1, 4.8, 2.6).move_to([-3.1, 0.35, 0])
            pt = T('session 1 · YES at −0.486 rad', 22, C_S1).move_to([P1.pl.p(45, 12)[0], P1.pl.p(45, 12)[1] + 0.3, 0])
            self.play(FadeIn(VGroup(P1.pl, P1.trace)), FadeIn(pt), run_time=0.8)
            self.play(FadeIn(P1.rest), run_time=0.4)
            # legend column
            sx = 1.0
            l1 = VGroup(DashedVMobject(Line(ORIGIN, RIGHT * 0.7, stroke_color=C_PASSIVE, stroke_width=3), num_dashes=6),
                        W('passive table: measured in a separate relaxed trial', 22, INK, width=chars_for(4.6, 22))
                        ).arrange(RIGHT, buff=0.25)
            l2 = VGroup(Line(ORIGIN, RIGHT * 0.7, stroke_color=C_GOOD, stroke_width=4),
                        W('relaxed level between pushes, inside the active trial', 22, INK, width=chars_for(4.6, 22))
                        ).arrange(RIGHT, buff=0.25)
            l3 = VGroup(Rectangle(width=0.7, height=0.3, fill_color=C_BAD, fill_opacity=0.5, stroke_width=0),
                        W('not the same number', 22, INK, width=30)).arrange(RIGHT, buff=0.25)
            legend = VGroup(l1, l2, l3).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([3.7, 1.15, 0])
            self.until(self.cue('Remember, the passive'))
            self.play(Create(P1.tline), FadeIn(P1.tlab), FadeIn(l1), run_time=0.8)
            self.until(self.cue('But the relaxed level'))
            self.play(Create(P1.rline), FadeIn(P1.rlab), FadeIn(l2), run_time=0.8)
            self.until(self.cue('is not the same number'))
            self.play(FadeIn(P1.gap), FadeIn(l3), run_time=0.7)
            self.until(self.cue('At the toes-down positions'))
            ch = F['y05']['audit_item2_check']['legacy']
            bl = bullets([M('table above the relaxed level in\n' + f'<b>{ch["n_positive"]} of {ch["n"]}</b> toes-down trials', 24),
                          M(f'typically <span foreground="#F0E442">≈ {ch["median"]:.1f} Nm</span>,\n'
                            f'up to <span foreground="#F0E442">{ch["max"]:.0f} Nm</span>', 24)], size=24, buff=0.3)
            bl.move_to([3.7, -1.3, 0])
            self.play(FadeIn(bl), run_time=0.7)
        self.play(FadeOut(VGroup(legend, bl)), run_time=0.4)

        # ── b04: the same for session 2 ─────────────────────────────────────
        with self.beat('c14_b04'):
            P2 = yes_panel(2, 4.8, 2.6, ylabel=False).move_to([3.85, 0.35, 0])
            pt2 = T('session 2 · YES at −0.486 rad', 22, C_S2).move_to([P2.pl.p(45, 12)[0], P1.pl.p(45, 12)[1] + 0.3, 0])
            self.until(self.cue('For YES at minus'))
            self.play(FadeIn(VGroup(P2.pl, P2.trace)), FadeIn(pt2), run_time=0.8)
            self.play(FadeIn(P2.rest), run_time=0.3)
            self.until(self.cue('the passive hold went'))
            self.play(Create(P2.tline), FadeIn(P2.tlab), run_time=0.6)
            self.play(Indicate(P1.tlab, scale_factor=1.3, color=C_HI), Indicate(P2.tlab, scale_factor=1.3, color=C_HI), run_time=1.0)
            self.until(self.cue('and the table overshoots'))
            self.play(Create(P2.rline), FadeIn(P2.rlab), run_time=0.5)
            self.play(FadeIn(P2.gap), run_time=0.4)
            disp = F['y05']['display']['yes_m049']
            c1 = M(f'table − relaxed = <span foreground="#F85149"><b>{disp["table_minus_relaxed"][0]:.1f} Nm</b></span> (session 1)', 22) \
                .move_to([P1.pl.p(45, 2)[0], -2.1, 0])
            c2 = M(f'table − relaxed = <span foreground="#F85149"><b>{disp["table_minus_relaxed"][1]:.1f} Nm</b></span> (session 2)', 22) \
                .move_to([P2.pl.p(45, 2)[0], -2.1, 0])
            self.play(FadeIn(c1), FadeIn(c2), run_time=0.6)
            self.until(self.cue("So the answer key's resting level"))
            moved = disp['table_minus_relaxed'][1] - disp['table_minus_relaxed'][0]
            spread = F['y05']['c_yes_m049']['S2']['active_target_windowends']['sd']
            r1 = M(f'the answer key\'s resting level moves by <span foreground="#F0E442"><b>≈ {moved:.1f} Nm</b></span> between sessions', 24) \
                .move_to([0, -2.98, 0])
            r2 = M(f'while the active torque there spreads by only <b>{spread:.1f} Nm</b>', 24, color=DIM).move_to([0, -3.42, 0])
            self.play(FadeIn(r1), run_time=0.6)
            self.play(FadeIn(r2), run_time=0.6)
            self.until(self.cue('And the relaxed level even drifts'))
            seg1 = F['y05']['c_yes_m049']['S1']['relaxed_level']['rest_seg_medians_min_max']
            seg2 = F['y05']['c_yes_m049']['S2']['relaxed_level']['rest_seg_medians_min_max']
            d1 = T(f'relaxed stretches: {seg1[0]:.1f}–{seg1[1]:.1f} Nm', 20, C_GOOD).move_to([P1.pl.p(45, 2)[0], -2.45, 0])
            d2 = T(f'relaxed stretches: {seg2[0]:.1f}–{seg2[1]:.1f} Nm', 20, C_GOOD).move_to([P2.pl.p(45, 2)[0], -2.45, 0])
            self.play(FadeIn(d1), FadeIn(d2), Indicate(P1.rest, color=C_GOOD), Indicate(P2.rest, color=C_GOOD), run_time=1.0)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b05: the answer key does not explain every offset ───────────────
        with self.beat('c14_b05'):
            sh = F['y07']['shift']['cells']
            hdr = reason_header(2, 'Does the key explain the offset?')
            self.play(FadeIn(hdr), run_time=0.5)
            pl = Plot((-1.5, 1.5), (-2, 1), 6.4, 3.7, xticks=(-1, 0, 1), yticks=(-2, -1, 0, 1),
                      xlabel='true average, session 1 − session 2 (Nm)', ylabel='model offset: predicted − true (Nm)',
                      tick_size=18, xfmt=lambda v: f'{v:+g}' if v else '0', yfmt=lambda v: f'{v:+g}' if v else '0')
            pl.move_to([-2.5, -0.35, 0])
            diag = Line(pl.p(-1.5, -1.5), pl.p(1, 1), stroke_color=C_HI, stroke_width=2.5)
            diag_lab = T('fully explained by the move', 20, C_HI)
            _dv = pl.p(1, 1) - pl.p(-1.5, -1.5)
            _ang = np.arctan2(_dv[1], _dv[0])
            diag_lab.rotate(_ang).move_to(pl.p(-0.72, -0.72) + np.array([-np.sin(_ang), np.cos(_ang), 0]) * 0.36)
            z1, z2 = pl.hline(0, FAINT, 1.2), pl.vline(0, FAINT, 1.2)
            dots = {}
            for c in sh:
                k = (c['subject'], round(c['pos'], 3))
                dots[k] = Dot(pl.p(c['s1_minus_s2_mean'], c['lstm_bias']), radius=0.08, color=C_PASSIVE)
            self.until(self.cue("Compare the model's offset"))
            self.play(FadeIn(pl), Create(z1), Create(z2), run_time=0.7)
            self.play(Create(diag), FadeIn(diag_lab), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots.values()], lag_ratio=0.08), run_time=0.8)
            by = {(c['subject'], round(c['pos'], 3)): c for c in sh}
            yes = by[('YES', -0.486)]
            eg = [by[('EG', -0.739)], by[('EG', -0.598)]]
            ies = [by[('IES', -0.591)], by[('IES', -0.484)]]
            rx = 1.15
            self.until(self.cue('At YES'))
            self.play(*[dots[('YES', -0.486)].animate.set_color(C_GOOD).scale(1.5)], run_time=0.5)
            t_yes = VGroup(T('YES −0.486 rad', 22, C_GOOD, weight='BOLD'),
                           T(f'moved {fmt(yes["s1_minus_s2_mean"], 2, True)} Nm  ·  model {fmt(yes["lstm_bias"], 2, True)} Nm', 20, INK),
                           T('the key explains a good part', 20, DIM)).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([4.0, 1.7, 0])
            self.play(FadeIn(t_yes), run_time=0.5)
            self.until(self.cue("But at EG's two"))
            self.play(*[dots[('EG', e)].animate.set_color(C_BAD).scale(1.5) for e in (-0.739, -0.598)], run_time=0.5)
            mv = max(abs(c['s1_minus_s2_mean']) for c in eg)
            t_eg = VGroup(T('EG, two toes-down positions', 22, C_BAD, weight='BOLD'),
                          T(f'moved ≤ {mv:.2f} Nm  ·  model {fmt(max(c["lstm_bias"] for c in eg), 1)} to {fmt(min(c["lstm_bias"] for c in eg), 1)} Nm', 20, INK)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([4.0, 0.35, 0])
            self.play(FadeIn(t_eg), run_time=0.5)
            self.until(self.cue('IES has no passive'))
            self.play(*[dots[('IES', e)].animate.set_color(C_BAD).scale(1.5) for e in (-0.591, -0.484)], run_time=0.5)
            mvi = max(abs(c['s1_minus_s2_mean']) for c in ies)
            t_ies = VGroup(T('IES, two toes-down positions', 22, C_BAD, weight='BOLD'),
                           T(f'no passive subtraction at all', 20, DIM),
                           T(f'moved ≤ {mvi:.2f} Nm  ·  model {fmt(max(c["lstm_bias"] for c in ies), 1)} to {fmt(min(c["lstm_bias"] for c in ies), 1)} Nm', 20, INK)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([4.0, -1.1, 0])
            self.play(FadeIn(t_ies), run_time=0.5)
            self.until(self.cue('There, the offset comes'))
            frm = W('the offset comes from the model itself, reacting to session-2 inputs that changed', 22, C_HI,
                    width=chars_for(5.4, 22)).move_to([3.95, -2.55, 0])
            self.play(FadeIn(frm), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b06: reason 3, the EMG changes size ─────────────────────────────
        with self.beat('c14_b06'):
            hdr = reason_header(3, 'The EMG changes size between sessions')
            self.play(FadeIn(hdr), run_time=0.5)
            jm = next(r for r in F['x03']['size_ratio']['JM'] if abs(r['angle'] + 0.36) < 0.005)
            vmax = 1.3
            B = bars(jm['before'], ['GM', 'GL', 'SOL', 'TA'], MUSCLE_C, width=4.6, height=2.6, vmax=vmax,
                     value_fmt=lambda v: f'{v:.2f}', label_size=20, value_size=22, gap=0.35)
            B.move_to([-3.6, -0.55, 0])
            one = DashedVMobject(Line(B.axis.get_left() + UP * (1 / vmax * 2.6), B.axis.get_right() + UP * (1 / vmax * 2.6),
                                      stroke_color=C_HI, stroke_width=2), num_dashes=40)
            for k in range(4):
                B.vals[k].move_to(B.bars[k].get_top() + DOWN * 0.24).set_color(BG)
            one_lab = T('same size', 18, C_HI).next_to(one, UP, buff=0.05).align_to(one, LEFT)
            bt = T('EMG size per Nm: session 2 ÷ session 1', 20, DIM).move_to([-3.6, 1.95, 0])
            bt2 = T('JM at −0.36 rad', 22, INK, weight='BOLD').next_to(bt, DOWN, buff=0.12)
            self.until(self.cue('the EMG changes size'))
            self.play(FadeIn(B.axis), FadeIn(B.labels), FadeIn(bt), FadeIn(bt2), Create(one), FadeIn(one_lab), run_time=0.7)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in B.bars], lag_ratio=0.15), run_time=1.0)
            self.play(FadeIn(B.vals), run_time=0.3)
            self.until(self.cue('dividing by session'))
            cannot = W("dividing by session 1's maximum cannot correct this", 24, INK, width=chars_for(5.8, 24)).move_to([3.5, 2.0, 0])
            self.play(FadeIn(cannot), run_time=0.6)
            self.until(self.cue('When a muscle reads weaker'))
            weak = W('a muscle reads weaker in session 2 → the model under-predicts the push', 22, DIM, width=chars_for(5.8, 22)).move_to([3.5, 1.05, 0])
            self.play(FadeIn(weak), run_time=0.6)
            self.until(self.cue("JM's predicted pushes"))
            base_y, th = -2.75, 1.35
            groups = VGroup()
            for gx, ratio, lab in ((2.4, 0.7, '−0.36 rad'), (5.0, 0.4, '−0.64 rad')):
                tb = Rectangle(width=0.55, height=th, fill_color=C_TQ, fill_opacity=0.9, stroke_width=0).move_to([gx - 0.4, base_y + th / 2, 0])
                pb = Rectangle(width=0.55, height=th * ratio, fill_color=C_PRED, fill_opacity=0.95, stroke_width=0).move_to([gx + 0.4, base_y + th * ratio / 2, 0])
                rt = T(f'×{ratio:.1f}', 30, C_PRED, weight='BOLD').next_to(pb, UP, buff=0.08)
                pl_ = T(lab, 20, DIM).move_to([gx, base_y - 0.3, 0])
                groups.add(VGroup(tb, pb, rt, pl_))
            leg = VGroup(VGroup(Square(0.18, fill_color=C_TQ, fill_opacity=1, stroke_width=0), T('true push', 18, INK)).arrange(RIGHT, buff=0.1),
                         VGroup(Square(0.18, fill_color=C_PRED, fill_opacity=1, stroke_width=0), T('predicted push', 18, INK)).arrange(RIGHT, buff=0.1)
                         ).arrange(RIGHT, buff=0.4).move_to([3.7, -0.35, 0])
            self.play(FadeIn(leg), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.15) for g in groups], lag_ratio=0.4), run_time=1.4)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b07: reason 4, the loss is dominated by strong positions ────────
        with self.beat('c14_b07'):
            hdr = reason_header(4, 'The loss is dominated by strong positions')
            self.play(FadeIn(hdr), run_time=0.5)
            d5 = load('y05')
            ssd, sq = d5['loss_YES_share_ssd'], d5['loss_YES_share_sqerr']
            cols = ANGLE_C
            pies, titles, bigs = [], [], []
            for (x0, frac, ttl) in ((-3.3, ssd, 'share of the push variation'), (3.3, sq, 'share of the squared error')):
                pie_ = pie(list(frac / frac.sum()), cols, radius=1.35)
                pie_.move_to([x0, 0.5, 0])
                # pull the two toes-down wedges out
                a = np.pi / 2
                for k in range(2):
                    ang = -2 * np.pi * (frac[k] / frac.sum())
                    mid = a + ang / 2
                    pie_[k].shift(np.array([np.cos(mid), np.sin(mid), 0]) * 0.18)
                    a += ang
                pies.append(pie_)
                titles.append(T(ttl, 22, INK).move_to([x0, 2.3, 0]))
                share = float(frac[:2].sum() / frac.sum()) * 100
                bigs.append(VGroup(T(f'{share:.1f} %', 44, C_HI, weight='BOLD'),
                                   T('in the two most toes-down positions', 20, DIM),
                                   T('an equal share would be 25 %', 20, DIM)).arrange(DOWN, buff=0.08).move_to([x0, -1.55, 0]))
            ramp = VGroup(*[Square(0.28, fill_color=c, fill_opacity=1, stroke_width=0) for c in cols]).arrange(RIGHT, buff=0.04)
            rl = T('toes down', 18, DIM).next_to(ramp, LEFT, buff=0.2)
            rr = T('toes up', 18, DIM).next_to(ramp, RIGHT, buff=0.2)
            ramp_g = VGroup(rl, ramp, rr).move_to([0, -2.95, 0])
            note = VGroup(T('YES · 8 positions', 20, DIM), T('loss in Nm²', 20, DIM)).arrange(DOWN, buff=0.08).move_to([0, 0.5, 0])
            # callouts pointing at the two pulled-out toes-down wedges
            calls = []
            for (x0, frac) in ((-3.3, ssd), (3.3, sq)):
                f = frac / frac.sum()
                am = np.pi / 2 - 2 * np.pi * float(f[:2].sum()) / 2
                st = np.array([x0, 0.5, 0]) + np.array([np.cos(am), np.sin(am), 0]) * (1.35 + 0.2)
                en = st + np.array([0.75, -0.12, 0])
                ln = Line(st, en, stroke_color=C_HI, stroke_width=2.5)
                lb = T('toes-down', 18, C_HI).next_to(en, RIGHT, buff=0.1)
                calls.append(VGroup(ln, lb))
            self.play(FadeIn(titles[0]), FadeIn(titles[1]), FadeIn(ramp_g), run_time=0.5)
            self.until(self.cue('the loss is in newton'))
            self.play(*[FadeIn(p, scale=0.8) for p in pies], FadeIn(note), run_time=1.0)
            self.until(self.cue('For YES'))
            self.play(FadeIn(bigs[0]), FadeIn(calls[0]), Indicate(pies[0][0], color=C_HI, scale_factor=1.6),
                      Indicate(pies[0][1], color=C_HI, scale_factor=1.6), run_time=0.8)
            self.until(self.cue('and about three percent'))
            self.play(FadeIn(bigs[1]), FadeIn(calls[1]), Indicate(pies[1][0], color=C_HI, scale_factor=1.6),
                      Indicate(pies[1][1], color=C_HI, scale_factor=1.6), run_time=0.8)
            self.until(self.cue('The network spends'))
            fin = T('the weak positions get whatever is left', 24, INK).move_to([0, -3.5, 0])
            self.play(FadeIn(fin), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b08: checked and ruled out ──────────────────────────────────────
        with self.beat('c14_b08'):
            rows_def = [
                ('Timing', 'at toes-down, the predictions line up in time with the truth (best shift ≤ 0.03 s)', 'Timing'),
                ('The angle sensor', 'replacing the measured angle by its fixed set value left the strong positions unchanged '
                                      '(YES, HM, EG; earlier screening setup)', 'The angle sensor'),
                ('Training', 'every run converges and stops normally', 'And training does not fail'),
            ]
            rows = []
            for ttl, det, _ in rows_def:
                tk = tick_mark(0.32)
                head = T(ttl, 28, INK, weight='BOLD')
                body = W(det, 24, DIM, width=chars_for(10.6, 24))
                txt = VGroup(head, body).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
                rows.append(VGroup(tk, txt).arrange(RIGHT, buff=0.4, aligned_edge=UP))
            col = VGroup(*rows).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
            cd = card([col], width=12.2, title='Checked and ruled out', color=FAINT, size=26)
            cd.move_to([0, -0.2, 0])
            self.play(FadeIn(cd.box), FadeIn(cd.content[0]), run_time=0.6)
            for r, (_, _, ph) in zip(rows, rows_def):
                self.until(self.cue(ph))
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b09: reason 5 ───────────────────────────────────────────────────
        with self.beat('c14_b09'):
            hdr = reason_header(5, 'Data problems for specific people')
            self.play(FadeIn(hdr), run_time=0.4)
            items = [('JM', "the passive table"), ('IES', 'the target is the total torque'),
                     ('YES', 'trial numbers typed in by hand')]
            names = VGroup(*[T(n, 28, C_HI, weight='BOLD') for n, _ in items]).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            descs = VGroup(*[T(d, 28, INK) for _, d in items])
            xf = names.get_right()[0] + 0.6
            for dsc, nm in zip(descs, names):
                dsc.move_to([xf + dsc.width / 2, nm.get_center()[1], 0])
            rows = VGroup(*[VGroup(nm, dsc) for nm, dsc in zip(names, descs)])
            cd = card([rows], width=9.0, title=None, color=FAINT).move_to([0, 0.2, 0])
            nxt = T('next chapter: bugs and loose ends  →', 28, C_HI, weight='BOLD').move_to([0, -2.2, 0])
            self.play(FadeIn(cd.box), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(r) for r in cd.content[0]], lag_ratio=0.5), run_time=1.5)
            self.until(self.cue('Those deserve'))
            self.play(FadeIn(nxt), run_time=0.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
