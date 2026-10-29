"""Chapter 15: Bugs and loose ends."""
from manim import *
from style2 import *

SEV = {'high': C_BAD, 'medium': C_HI, 'low': FAINT}


def pill(txt, w=1.15):
    color = SEV[txt]
    r = RoundedRectangle(corner_radius=0.13, width=w, height=0.36, fill_color=color, fill_opacity=1, stroke_width=0)
    t = T(txt, 18, INK if txt == 'low' else BG, weight='BOLD').move_to(r)
    return VGroup(r, t)


def item_row(body, sev=None, buff=0.35):
    """One checklist row: severity pill (optional) + body (Mobject)."""
    if sev is None:
        return body
    p = pill(sev)
    return VGroup(p, body).arrange(RIGHT, buff=buff, aligned_edge=UP)


def trial_col(cx, t, pos, tq, tq_range, tq_ticks, w=4.6, title='', title_color=INK, ytop=-0.1):
    """Angle (small, top) + torque (bottom) of one passive trial, centred at x = cx."""
    a = Plot((0, 90), (-0.12, 0.04), w, 0.7, yticks=(-0.1, 0), tick_size=18, yfmt=lambda v: f'{v:g}')
    b = Plot((0, 90), tq_range, w, 2.1, xticks=(0, 30, 60, 90), yticks=tq_ticks, xlabel='time (s)', ylabel='torque (Nm)',
             tick_size=18)
    a.move_to([cx, 2.05, 0])
    b.move_to([cx, ytop, 0])
    # align the plot areas horizontally (labels shift the group centres)
    a.shift(RIGHT * (b.frame.get_center()[0] - a.frame.get_center()[0]))
    a_cap = T('angle (rad)', 18, DIM).next_to(a.frame, UP, buff=0.05).align_to(a.frame, LEFT)
    ttl = T(title, 26, title_color, weight='BOLD').move_to([b.frame.get_center()[0], 2.95, 0])
    a_curve = envelope_curve(a, t, pos, C_ANGLE, 2.0)
    b_curve = envelope_curve(b, t, tq, C_TQ, 1.5)
    return dict(a=a, b=b, a_cap=a_cap, ttl=ttl, a_curve=a_curve, b_curve=b_curve)


class Ch15(BeatScene):
    def construct(self):
        F = facts()
        d5 = load('y05')
        x4 = load('x04')
        disp = F['y05']['display']

        # ── b01: JM's passive trial ─────────────────────────────────────────
        with self.beat('c15_b01'):
            tag = self.intro('c15')
            t, pos, tq = d5['jmP7S1_t'], d5['jmP7S1_pos'], d5['jmP7S1_tq']
            tr = disp['jm_trial21_S1']
            title = T('JM · passive trial at −0.078 rad · session 1', 28, INK, weight='BOLD').move_to([0.4, 2.95, 0])
            A = Plot((0, 90), (-0.12, 0.04), 10.4, 0.9, yticks=(-0.1, 0), tick_size=18, yfmt=lambda v: f'{v:g}')
            B = Plot((0, 90), (-10, 58), 10.4, 2.7, xticks=(0, 20, 40, 60, 80), yticks=(0, 20, 40), xlabel='time (s)',
                     ylabel='torque (Nm)', tick_size=18)
            A.move_to([0.4, 1.75, 0])
            B.move_to([0.4, -0.9, 0])
            A.shift(RIGHT * (B.frame.get_center()[0] - A.frame.get_center()[0]))
            a_lab = T('angle (rad): stays put', 20, DIM).next_to(A.frame, UP, buff=0.05).align_to(A.frame, LEFT)
            i20 = int(np.searchsorted(t, 20.0))
            k0 = int(np.searchsorted(t, 0.3))
            a_curve = A.curve(t[k0:], pos[k0:], C_ANGLE, 2.5)
            c1 = B.curve(t[:i20 + 1], tq[:i20 + 1], C_TQ, 2.0)
            c2 = envelope_curve(B, t[i20:], tq[i20:], C_TQ, 1.6)
            band1 = B.band(0, 20, C_GOOD, 0.16)
            band2 = B.band(20, 90, C_BAD, 0.12)
            lab1 = VGroup(T('relaxed hold', 20, C_GOOD, weight='BOLD'), T(f'{fmt(tr["hold_0_20s"], 1)} Nm', 22, INK)) \
                .arrange(DOWN, buff=0.05).move_to(B.p(10, 44))
            lab2 = T('not relaxed', 30, C_BAD, weight='BOLD').move_to(B.p(55, 53))
            sw = T(f'10-s averages: {min(tr["block_means_after_20s"]):.0f}–{max(tr["block_means_after_20s"]):.0f} Nm', 22, INK) \
                .next_to(B.frame, DOWN, buff=1.0)
            self.until(self.cue('Here is a passive trial'))
            self.play(FadeIn(title), FadeIn(A), FadeIn(B), FadeIn(a_lab), run_time=0.8)
            self.play(Create(a_curve), run_time=0.8)
            self.until(self.cue('The first twenty seconds'))
            self.play(FadeIn(band1), Create(c1), run_time=0.9)
            self.play(FadeIn(lab1), run_time=0.4)
            self.until(self.cue('Then, for seventy seconds'))
            self.play(FadeIn(band2), Create(c2), run_time=2.0, rate_func=linear)
            self.until(self.cue('Whatever was happening'))
            self.play(FadeIn(lab2), FadeIn(sw), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b02: why the code let it through ────────────────────────────────
        with self.beat('c15_b02'):
            e_t, e_pos, e_tq = x4['p20_t'], x4['p20_pos'], x4['p20_tq']
            piece = F['x04']['p20_pieces']
            eg_sd = piece[1]['tq_sd']
            cut = piece[0]['seconds']
            L = trial_col(-3.1, e_t, e_pos, e_tq, (-100, 62), (-80, -40, 0, 40), title='EG · passive trial', title_color=INK)
            R = trial_col(3.5, t, pos, tq, (-10, 58), (0, 20, 40), title='JM · the trial we just saw', title_color=INK)
            self.play(*[FadeIn(m) for col in (L, R) for m in (col['a'], col['b'], col['a_cap'], col['ttl'])],
                      *[Create(col[k]) for col in (L, R) for k in ('a_curve', 'b_curve')], run_time=1.6)
            self.until(self.cue('wherever the angle jumps'))
            vl_a = L['a'].vline(cut, C_HI, 2.5)
            vl_b = L['b'].vline(cut, C_HI, 2.5)
            cut_lab = T('cut', 20, C_HI, weight='BOLD').move_to(L['b'].p(cut + 6, 54))
            self.play(Create(vl_a), Create(vl_b), FadeIn(cut_lab), run_time=0.8)
            self.until(self.cue('then rejects any piece'))
            pb = L['b']
            k1 = pb.band(0, cut, C_GOOD, 0.16)
            k2 = pb.band(cut, 90, C_BAD, 0.14)
            kl1 = T('kept', 18, C_GOOD, weight='BOLD').move_to(pb.p(cut / 2, 54))
            kl2 = T('rejected', 20, C_BAD, weight='BOLD').move_to(pb.p(62, 54))
            v_eg = VGroup(T('angle jumps → cut into two pieces', 20, INK),
                          M(f'later piece: SD <b>{eg_sd:.1f}</b> > 20 Nm → <span foreground="#F85149"><b>rejected</b></span>', 20)
                          ).arrange(DOWN, buff=0.06).move_to([L['b'].frame.get_center()[0], -2.3, 0])
            self.play(FadeIn(k1), FadeIn(k2), FadeIn(kl1), FadeIn(kl2), FadeIn(v_eg), run_time=0.8)
            self.until(self.cue('This JM trial has no jump'))
            rb = R['b']
            j1 = rb.band(0, 90, C_HI, 0.10)
            jl = T('one piece', 20, C_HI, weight='BOLD').move_to(rb.p(45, 54))
            v_jm = VGroup(T('no jump → no cut, one piece', 20, INK),
                          M(f'SD <b>{tr["sd"]:.1f}</b> ≤ 20 · average <b>{tr["mean"]:.1f}</b> ≤ 20 → '
                            f'<span foreground="#F0E442"><b>accepted</b></span>', 20)
                          ).arrange(DOWN, buff=0.06).move_to([R['b'].frame.get_center()[0], -2.3, 0])
            self.play(FadeIn(j1), FadeIn(jl), FadeIn(v_jm), run_time=0.8)
            self.until(self.cue('A real relaxed hold'))
            hs = F['y05']['b_static_hold_sd']['framework_holds']
            strip = VGroup(RoundedRectangle(corner_radius=0.12, width=12.6, height=0.55, fill_color=PANEL, fill_opacity=1,
                                            stroke_color=C_GOOD, stroke_width=2),
                           M(f'real relaxed holds: SD <b>{hs["sd_min"]:.3f}–{hs["sd_max"]:.2f} Nm</b>, typically '
                             f'<b>{hs["sd_median"]:.2f} Nm</b> → a limit of 20 is far too loose', 21))
            strip[1].move_to(strip[0])
            strip.move_to([0, -3.0, 0])
            self.play(FadeIn(strip), run_time=0.6)
            self.until(self.cue('In the git version'))
            lim = M('limit on the average: <b>10 Nm</b> in git → <span foreground="#F85149"><b>20 Nm</b></span> in the working copy '
                    '· SD limit 20 in both', 20, color=DIM).move_to([0, -3.55, 0])
            self.play(FadeIn(lim), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b03: the passive table with the spike ───────────────────────────
        with self.beat('c15_b03'):
            ca, cl, cs = d5['jm_curve_angle'], d5['jm_curve_legacy_S1'], d5['jm_curve_static_S1']
            lp, lq = d5['jm_legacy_S1_pos'], d5['jm_legacy_S1_tq']
            sp, sq = d5['jm_static_S1_pos'], d5['jm_static_S1_tq']
            P = Plot((-0.85, 0.3), (-10, 14), 7.4, 3.7, xticks=(-0.8, -0.6, -0.4, -0.2, 0, 0.2), yticks=(-10, -5, 0, 5, 10),
                     xlabel='joint angle (rad): toes down ← → toes up', ylabel='passive torque in the table (Nm)', tick_size=18)
            P.move_to([-2.4, -0.15, 0])
            ttl = T('JM · session-1 passive table', 26, INK, weight='BOLD').move_to([-2.4, 2.75, 0])
            z = P.hline(0, FAINT, 1.2)
            leg_curve = P.curve(ca, cl, C_BAD, 3.5)
            leg_dots = VGroup(*[Dot(P.p(a, b), radius=0.09, color=C_BAD) for a, b in zip(lp, lq)])
            cl_curve = P.curve(ca, cs, C_GOOD, 3.5)
            cl_dots = VGroup(*[Dot(P.p(a, b), radius=0.09, color=C_GOOD) for a, b in zip(sp, sq)])
            spike_i = int(np.argmax(lq))
            peak = P.p(lp[spike_i], lq[spike_i])
            sp_lab = T(f'+{lq[spike_i]:.1f} Nm', 36, C_BAD, weight='BOLD').next_to(peak, RIGHT, buff=0.2).shift(UP * 0.1)
            clean_at = float(np.interp(lp[spike_i], ca, cs))
            cl_lab = T(f'≈ {fmt(clean_at, 0)} Nm', 30, C_GOOD, weight='BOLD').move_to(P.p(-0.42, -6.5))
            cl_arrow = Arrow(cl_lab.get_right() + RIGHT * 0.05, P.p(lp[spike_i], clean_at) + DOWN * 0.08, buff=0.05, color=C_GOOD,
                             stroke_width=3, max_tip_length_to_length_ratio=0.3)
            lg = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.6, stroke_color=C_BAD, stroke_width=4),
                               W('the table the code built', 22, INK, width=chars_for(3.3, 22))).arrange(RIGHT, buff=0.2),
                        VGroup(Line(ORIGIN, RIGHT * 0.6, stroke_color=C_GOOD, stroke_width=4),
                               W('relaxed values only, joined up', 22, INK, width=chars_for(3.3, 22))).arrange(RIGHT, buff=0.2)
                        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([4.5, 1.7, 0])
            note = W('+11.7 = this trial averaged with the non-relaxed second half of another passive trial', 22, INK,
                     width=chars_for(4.0, 22)).move_to([4.6, -0.7, 0])
            self.play(FadeIn(ttl), FadeIn(P), Create(z), run_time=0.7)
            self.play(Create(leg_curve), FadeIn(leg_dots), FadeIn(lg[0]), run_time=1.0)
            self.until(self.cue('Averaged together'))
            self.play(FadeIn(sp_lab), Indicate(leg_dots[spike_i], color=C_BAD, scale_factor=1.8), FadeIn(note), run_time=1.0)
            self.until(self.cue('where the relaxed value'))
            self.play(Create(cl_curve), FadeIn(cl_dots), FadeIn(lg[1]), run_time=1.0)
            self.play(FadeIn(cl_lab), Create(cl_arrow), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b04: the consequence ────────────────────────────────────────────
        with self.beat('c15_b04'):
            sh = disp['jm_p7_legacy_vs_static']
            n1 = VGroup(T(f'{sh["S1"][2]:.1f} Nm', 42, C_BAD, weight='BOLD'), T('session-1 targets shifted', 22, INK)) \
                .arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to([-3.4, 2.75, 0])
            n2 = VGroup(T(f'{sh["S2"][2]:.1f} Nm', 42, C_BAD, weight='BOLD'), T('session-2 targets shifted', 22, INK)) \
                .arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to([3.4, 2.75, 0])
            own = np.array([r['r2_own_target'] for r in F['y07']['own_target']['per_seed']])
            jmr = load('y03')['r2'][4, :, 5]
            P = Plot((-62, 4), (0, 3), 10.4, 1.1, xticks=(-60, -40, -20, 0), xlabel='R² of the ten seeds', tick_size=18)
            P.move_to([0.2, -1.65, 0])
            zero = P.vline(0, DIM, 2, y0=0.0, y1=3.0)
            # the own-target scores live in a small zoomed axis (-10 to 1), otherwise they collapse into one clump at 0
            Z = Plot((-10, 1), (0, 3), 6.0, 0.7, xticks=(-10, -5, 0), tick_size=18)
            Z.move_to([0.2, 0.75, 0])
            zzero = Z.vline(0, DIM, 2, y0=0.0, y1=3.0)
            zoom_lab = T('zoomed axis', 18, DIM).next_to(Z.frame, LEFT, buff=0.3)
            jit = [0.0, 0.2, -0.2, 0.1, -0.1, 0.25, -0.25, 0.05, 0.15, -0.15]

            def strip(plot, vals, colors, mw):
                g = VGroup()
                for k, v in enumerate(vals):
                    g.add(Dot(plot.p(v, 1.5 + 2 * jit[k]), radius=0.1, color=colors(v), stroke_color=BG, stroke_width=1.2))
                med = float(np.median(vals))
                g.add(Line(plot.p(med, 0.15), plot.p(med, 2.85), stroke_color=C_TQ, stroke_width=3))
                return g

            sA = strip(Z, own, lambda v: C_BAD if v < 0 else C_GOOD, 0)
            sB = strip(P, jmr, lambda v: C_BAD, 0)
            medA = float(np.median(own))
            med_lab = T(f'median {medA:.2f}', 18, C_TQ).move_to([Z.p(medA, 3)[0], Z.frame.get_top()[1] + 0.25, 0])
            lA = VGroup(T('against its own (buggy) targets', 24, INK, weight='BOLD'),
                        M(f'median R² <b>{medA:.2f}</b> · {int((own < 0).sum())} of 10 below zero (lowest {fmt(own.min(), 1)})', 22, color=DIM)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            lA.move_to([-6.4 + lA.width / 2, 1.95, 0])
            lB = VGroup(T('against the clean answer key', 24, INK, weight='BOLD'),
                        M(f'every seed {fmt(jmr.max(), 0)} to {fmt(jmr.min(), 0)} (median {fmt(np.median(jmr), 0)})', 22, color=DIM)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            lB.move_to([-6.4 + lB.width / 2, -0.35, 0])
            ban = RoundedRectangle(corner_radius=0.14, width=9.2, height=1.1, fill_color=PANEL, fill_opacity=1,
                                   stroke_color=C_HI, stroke_width=2)
            ban_t = W('Grading against labels built like the training labels can hide a labelling error.', 26, C_HI,
                      width=chars_for(7.6, 26), align=ORIGIN).move_to(ban)
            ban_g = VGroup(ban, ban_t).move_to([0, -3.2, 0])
            self.until(self.cue("JM's training targets"))
            self.play(FadeIn(n1, shift=UP * 0.15), run_time=0.6)
            self.until(self.cue('and the session-2 targets'))
            self.play(FadeIn(n2, shift=UP * 0.15), run_time=0.6)
            self.until(self.cue('Both are wrong'))
            self.play(FadeIn(Z), Create(zzero), FadeIn(zoom_lab), FadeIn(lA), run_time=0.7)
            self.play(LaggedStart(*[FadeIn(m, scale=0.5) for m in sA], lag_ratio=0.08), FadeIn(med_lab), run_time=1.0)
            self.until(self.cue('Against a clean answer key'))
            self.play(FadeIn(lB), FadeIn(P), Create(zero), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(m, scale=0.5) for m in sB], lag_ratio=0.08), run_time=1.0)
            self.until(self.cue('The lesson'))
            self.play(FadeIn(ban_g), run_time=0.7)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b05: four issues already met ────────────────────────────────────
        with self.beat('c15_b05'):
            items = [('No random seed: runs cannot be reproduced', 'high', 'No random seed'),
                     ('IES: the target is the total torque', 'medium', 'The target for IES'),
                     ('Validation is the end of the same trials, with no gap', 'medium', 'Validation is the end'),
                     ('The filters use the future: offline only', 'medium', 'And the filters use')]
            rows = [item_row(T(tx, 28, INK), sev) for tx, sev, _ in items]
            col = VGroup(*rows).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
            cd = card([col], width=11.4, title='Other issues: four we have already met', size=28)
            cd.move_to([0, -0.1, 0])
            sev_cap = T('severity', 18, DIM).next_to(rows[0][0], UP, buff=0.08)
            self.play(FadeIn(cd.box), FadeIn(cd.content[0]), run_time=0.6)
            for r, (_, _, ph) in zip(rows, items):
                self.until(self.cue(ph))
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b06: four smaller ones ──────────────────────────────────────────
        with self.beat('c15_b06'):
            rw = chars_for(9.8, 24)
            r4 = VGroup(W('CH, at about −0.33 rad: toes pulled up in session 1, pushed down in session 2', 24, INK, width=rw),
                        M('a guess from session 1\'s average scores R² <span foreground="#F85149"><b>'
                          f'{fmt(F["y07"]["ch_flip"]["ref_r2"], 1)}</b></span>\n'
                          f'the baseline scores '
                          f'<span foreground="#3FB950"><b>{F["y07"]["ch_flip"]["lstm_r2_median"]:.2f}</b></span>: the network really reads the muscles',
                          22, color=DIM)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            items = [(W("Each muscle's scale comes from one single sample, sometimes from the validation part", 24, INK, width=rw), 'low', "Each muscle's scale"),
                     (W('The downsampling filter wobbles at the very end of each trial', 24, INK, width=rw), 'low', 'The downsampling filter'),
                     (W('Only eight angles: the angle input works like a trial label, so the network can learn an offset per '
                        'angle and quietly absorb table errors', 24, INK, width=rw), 'low', 'With only eight angles'),
                     (r4, 'low', 'And at about minus')]
            rows = [item_row(b, sev) for b, sev, _ in items]
            col = VGroup(*rows).arrange(DOWN, buff=0.34, aligned_edge=LEFT)
            cd = card([col], width=12.2, title='Four smaller ones', size=26)
            cd.move_to([0, -0.2, 0])
            self.play(FadeIn(cd.box), FadeIn(cd.content[0]), run_time=0.6)
            for r, (_, _, ph) in zip(rows, items):
                self.until(self.cue(ph))
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b07: housekeeping ───────────────────────────────────────────────
        with self.beat('c15_b07'):
            ents = [('The code in git is not what was run',
                     "it handles only 2 people, so it cannot have produced last semester's results; the baseline uses the "
                     'uncommitted working copy. Needs confirming.', 'The code in git'),
                    ('Comments that do not match the code',
                     "e.g. passive values 'matched by position tags': the tags are never used", 'Several comments'),
                    ('YES: trial numbers typed in by hand',
                     "the file's labels are mixed up both ways: some pushing trials are labelled passive, and some "
                     'non-pushing trials look like pushes', 'And for YES')]
            rows = []
            for hd, dt, _ in ents:
                rows.append(VGroup(T(hd, 28, C_HI, weight='BOLD'), W(dt, 24, INK, width=chars_for(11.2, 24)))
                            .arrange(DOWN, aligned_edge=LEFT, buff=0.1))
            col = VGroup(*rows).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            cd = card([col], width=12.4, title='Housekeeping', size=28)
            cd.move_to([0, -0.2, 0])
            self.play(FadeIn(cd.box), FadeIn(cd.content[0]), run_time=0.6)
            for r, (_, _, ph) in zip(rows, ents):
                self.until(self.cue(ph))
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.6)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

        # ── b08: reporting ──────────────────────────────────────────────────
        with self.beat('c15_b08'):
            jmv = F['y03']['scatter_summary']['lowest_r2_cell']
            ents = [('Showing R² below 0 as 0 hides failures',
                     M(f'e.g. JM at {fmt(jmv["position_rad"], 3)} rad: R² <b>{fmt(jmv["r2_med"], 1)}</b> would be shown as <b>0</b>', 24, color=DIM),
                     'Showing negative'),
                    ('Two baselines give different numbers',
                     T('the frozen code (10 seeds) and its rebuild in the newer code (3 seeds)', 24, DIM), 'There are two versions'),
                    ('Documents summarise differently',
                     T('floor at −1, floor at 0, or no floor', 24, DIM), 'And different documents')]
            rows = [VGroup(T(hd, 28, INK, weight='BOLD'), dt).arrange(DOWN, aligned_edge=LEFT, buff=0.08) for hd, dt, _ in ents]
            col = VGroup(*rows).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            cd = card([col], width=12.2, title='Reporting', size=28)
            cd.move_to([0, 0.5, 0])
            chips = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.16, width=3.2, height=0.6, fill_color=PANEL, fill_opacity=1,
                                                     stroke_color=C_HI, stroke_width=2), T(q, 26, C_HI, weight='BOLD'))
                             for q in ('which baseline?', 'which seeds?', 'which summary?')])
            for c in chips:
                c[1].move_to(c[0])
            chips.arrange(RIGHT, buff=0.35)
            foot = VGroup(T('When you quote a number:', 24, INK), chips).arrange(DOWN, buff=0.25).move_to([0, -2.85, 0])
            self.play(FadeIn(cd.box), FadeIn(cd.content[0]), run_time=0.6)
            for r, (_, _, ph) in zip(rows, ents):
                self.until(self.cue(ph))
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('When someone quotes'))
            self.play(FadeIn(foot[0]), LaggedStart(*[FadeIn(c, scale=0.8) for c in chips], lag_ratio=0.3), run_time=1.2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
