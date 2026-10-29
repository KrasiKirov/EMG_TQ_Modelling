"""Chapter 11: Scoring a prediction."""
from manim import *
from style2 import *

TD_C, OT_C = C_HI, DIM          # toes-down cells highlighted, other cells grey


def legend_row(items, size=20):
    """items: [(color, text)] -> row of line swatches + text."""
    return VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.5, stroke_color=c, stroke_width=3.5), T(t, size, INK))
                    .arrange(RIGHT, buff=0.15) for c, t in items]).arrange(RIGHT, buff=0.6)


class Ch11(BeatScene):
    def construct(self):
        d = load('y03')
        F = facts()
        y3, y1 = F['y03'], F['y01']
        t = d['t_window_s']
        key = d['yes_strong_key']
        pred0 = d['yes_strong_pred'][0]
        gsum = y1['B_group_summaries']['X00_baseline']
        td_sum, ot_sum = gsum['toes_down_12'], gsum['other_36']
        yes_pos = [float(v) for v in d['positions_snapped_rad'][5]]   # snapped values (0.2451 shows as 0.25)

        # ───────────────────────── b01: running the model over a trial ─────────────────────────
        with self.beat('c11_b01'):
            tag = self.intro('c11')
            head = T('session 2 · YES at +0.138 rad · one trial', 24, INK).move_to([0.6, 3.1, 0]).shift(LEFT * 3.0)
            pl = Plot((0, 89), (-40, 5), 11.2, 2.3, xticks=(0, 20, 40, 60, 80), yticks=(-40, -20, 0),
                      xlabel='time (s)', tick_size=18)
            pl.move_to([0.35, 1.25, 0])
            yl = T('predicted torque (Nm)', 18, DIM).rotate(PI / 2).next_to(pl, LEFT, buff=0.1)
            badge = VGroup(RoundedRectangle(corner_radius=0.15, width=2.3, height=0.5, fill_color=PANEL,
                                            fill_opacity=1, stroke_color=C_HI, stroke_width=2),
                           T('dropout off', 20, C_HI)).move_to([-4.6, -1.75, 0])
            badge[1].move_to(badge[0])
            self.play(FadeIn(head), FadeIn(pl), FadeIn(yl), FadeIn(badge), run_time=0.7)
            tv = ValueTracker(0)
            ds = 6
            idx = np.arange(0, len(t), ds)
            xs_all, ys_all = t[idx], np.clip(pred0[idx], -40, 5)

            def curve_fn():
                n = max(2, int(tv.get_value() / 0.01 / ds) + 1)
                n = min(n, len(idx))
                pts = [pl.p(a, b) for a, b in zip(xs_all[:n], ys_all[:n])]
                return VMobject(stroke_color=C_PRED, stroke_width=2.5).set_points_as_corners(pts)

            def win_fn():
                x = tv.get_value()
                h = pl.p(0, 5)[1] - pl.p(0, -40)[1]
                r = Rectangle(width=0.14, height=h, stroke_color=C_HI, stroke_width=2, fill_color=C_HI,
                              fill_opacity=0.25)
                return r.move_to([pl.p(x, 0)[0], (pl.p(0, 5)[1] + pl.p(0, -40)[1]) / 2, 0])

            def dot_fn():
                x = min(tv.get_value(), 88.9)
                i = int(x / 0.01)
                return Dot(pl.p(x, max(-40, pred0[i])), radius=0.09, color=C_PRED)

            cnt_lab = T('predictions so far', 22, DIM)
            cnt = always_redraw(lambda: T(f'{int(tv.get_value() / 0.01) + 1:,}', 48, C_PRED, weight='BOLD')
                                .move_to([-1.0, -1.75, 0]))
            cnt_lab.move_to([2.2, -1.72, 0])
            wlab = T('window (0.2 s, drawn wider)', 20, C_HI).move_to([0, 3.0, 0])
            wlab.align_to(pl, RIGHT)
            curve = always_redraw(curve_fn)
            win = always_redraw(win_fn)
            dot = always_redraw(dot_fn)
            self.until(self.cue('the kept model'))
            self.add(curve, win, dot, cnt)
            self.play(FadeIn(cnt_lab), FadeIn(wlab), run_time=0.4)
            self.play(tv.animate.set_value(88.9), run_time=6.2, rate_func=linear)
            fin = T('8,891 predictions per trial · one every 10 ms', 30, C_PRED, weight='BOLD')
            fin.next_to(badge, RIGHT, buff=0.5).set_y(-1.75)
            self.play(FadeOut(cnt), FadeOut(cnt_lab), FadeOut(win), FadeOut(wlab), FadeOut(dot), FadeIn(fin),
                      run_time=0.5)
            self.until(self.cue('The predictions are then grouped'))
            curve_s = curve.copy()
            self.remove(curve)
            self.add(curve_s)
            chips = VGroup()
            for p, c in zip(yes_pos, ANGLE_C):
                lab = T(f'{fmt(p, 2)} rad', 20, INK)
                box = RoundedRectangle(corner_radius=0.12, width=1.5, height=0.95, fill_color=PANEL,
                                       fill_opacity=1, stroke_color=c, stroke_width=3)
                sub = T('1 trial', 18, DIM)
                g = VGroup(lab, sub).arrange(DOWN, buff=0.1)
                chips.add(VGroup(box, g.move_to(box)))
            chips.arrange(RIGHT, buff=0.14).move_to([0, -0.15, 0])
            self.play(FadeOut(pl), FadeOut(yl), FadeOut(badge), FadeOut(head), FadeOut(fin),
                      ReplacementTransform(curve_s, chips), run_time=1.0)
            self.until(self.cue('compared with the answer key'))
            foot = T('scored against the answer key (still-hold passive table)', 26, C_HI).move_to([0, -1.9, 0])
            ar = VGroup(*[Arrow(c.get_bottom(), c.get_bottom() + DOWN * 0.7, buff=0.05, stroke_width=3, color=C_TQ,
                                max_tip_length_to_length_ratio=0.5) for c in chips])
            self.play(FadeIn(ar, lag_ratio=0.1), FadeIn(foot), run_time=0.8)

        # ───────────────────────── b02: MAE ─────────────────────────
        with self.beat('c11_b02'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            i0, i1 = 2000, 2500
            tt = t[i0:i1] - t[i0]
            tr, pr = key[i0:i1], pred0[i0:i1]
            up = Plot((0, 5), (-40, 5), 10.4, 2.2, xticks=(0, 1, 2, 3, 4, 5), yticks=(-40, -20, 0),
                      xlabel='time (s)', tick_size=18)
            up.move_to([0.4, 1.7, 0])
            yl = T('torque (Nm)', 18, DIM).rotate(PI / 2).next_to(up, LEFT, buff=0.1)
            c_tr = up.curve(tt, tr, C_TQ, 2.8)
            c_pr = up.curve(tt, pr, C_PRED, 2.8)
            ks = list(range(0, 500, 6))
            errs = VGroup(*[Line(up.p(tt[k], tr[k]), up.p(tt[k], pr[k]), stroke_color=C_BAD, stroke_width=3.5)
                            for k in ks])
            ae = np.abs(tr[ks] - pr[ks])
            mae_w = float(ae.mean())
            ymax_lo = float(np.ceil(ae.max()))
            lo = Plot((0, 5), (0, ymax_lo), 10.4, 1.3, yticks=(0, 2, 4), tick_size=18)
            lo.move_to([0.4, -1.55, 0])
            lo_t = T('size of each miss (Nm)', 20, DIM).next_to(lo, UP, buff=0.12).align_to(lo, LEFT)
            bars_lo = VGroup(*[Line(lo.p(tt[k], 0), lo.p(tt[k], a), stroke_color=C_BAD, stroke_width=3.5)
                               for k, a in zip(ks, ae)])
            mline = lo.hline(mae_w, C_HI, 3.0, dashed=True)
            mlab = T(f'MAE = {fmt(mae_w, 1)} Nm', 28, C_HI, weight='BOLD').next_to(lo, RIGHT, buff=0.0)
            mlab.move_to([lo.p(3.0, 0)[0], lo.p(0, 2.6)[1], 0])
            self.play(FadeIn(up), FadeIn(yl), Create(c_tr), run_time=1.0)
            self.play(Create(c_pr), run_time=0.9)
            self.until(self.cue('the average size of the miss'))
            self.play(FadeIn(errs, lag_ratio=0.02), run_time=1.0)
            self.play(FadeIn(lo), FadeIn(lo_t), run_time=0.4)
            self.play(*[TransformFromCopy(a, b) for a, b in zip(errs, bars_lo)], run_time=1.0)
            self.play(Create(mline), FadeIn(mlab), run_time=0.7)
            self.until(self.cue('An MAE of 0.5'))
            ex = T('An MAE of 0.5 Nm: typically half a newton metre off', 28, INK).move_to([0, -3.2, 0])
            self.play(FadeIn(ex, shift=UP * 0.1), run_time=0.6)

        # ───────────────────────── b03: R² against a lazy guess ─────────────────────────
        with self.beat('c11_b03'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            j0, j1 = 1900, 2500
            tt = t[j0:j1] - t[j0]
            tr, pr = key[j0:j1], pred0[j0:j1]
            mean_all = float(key.mean())
            pl3 = Plot((0, 6), (-40, 14), 10.4, 2.3, xticks=(0, 2, 4, 6), yticks=(-40, -20, 0), xlabel='time (s)',
                       tick_size=18)
            pl3.move_to([0.3, 1.6, 0])
            yl3 = T('torque (Nm)', 18, DIM).rotate(PI / 2).next_to(pl3, LEFT, buff=0.1)
            lg = legend_row([(C_TQ, 'answer key'), (C_PRED, 'prediction'), (C_HI, 'lazy guess')], 20)
            lg.move_to([0.3, 3.4, 0]).align_to(pl3, RIGHT)
            c_tr = pl3.curve(tt, tr, C_TQ, 2.8)
            c_pr = pl3.curve(tt, pr, C_PRED, 2.8)
            lazy = pl3.hline(mean_all, C_HI, 2.5, dashed=True)
            sel = [40, 160, 280, 400, 520]
            lz_seg = VGroup(*[Line(pl3.p(tt[k] - 0.05, tr[k]), pl3.p(tt[k] - 0.05, mean_all), stroke_color=C_HI,
                                   stroke_width=5) for k in sel])
            md_seg = VGroup(*[Line(pl3.p(tt[k] + 0.05, tr[k]), pl3.p(tt[k] + 0.05, pr[k]), stroke_color=C_BAD,
                                   stroke_width=5) for k in sel])
            self.until(self.cue('The score we report'))
            self.play(FadeIn(pl3), FadeIn(yl3), FadeIn(lg), run_time=0.6)
            self.play(Create(c_tr), run_time=0.8)
            self.until(self.cue('It compares the model'))
            self.play(Create(c_pr), run_time=0.8)
            self.until(self.cue('a flat line at'))
            self.play(Create(lazy), run_time=0.8)
            # squares
            k_s = 0.042
            lz_e = [abs(tr[k] - mean_all) for k in sel]
            md_e = [abs(tr[k] - pr[k]) for k in sel]
            base = -1.55

            def squares(es, color, cx):
                g = VGroup(*[Square(side_length=max(0.04, e * k_s), fill_color=color, fill_opacity=0.85,
                                    stroke_width=0) for e in es])
                g.arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
                g.move_to([cx, 0, 0])
                g.shift(UP * (base - g.get_bottom()[1]))
                return g

            sq_m = squares(md_e, C_BAD, 3.3)
            sq_l = squares(lz_e, C_HI, -3.0)
            lab_m = T("model's squared errors", 22, C_BAD).move_to([3.3, base - 0.35, 0])
            lab_l = T("lazy guess's squared errors", 22, C_HI).move_to([-3.0, base - 0.35, 0])
            self.until(self.cue("Add up the model's"))
            self.play(Create(md_seg), run_time=0.6)
            self.play(FadeIn(sq_m, shift=DOWN * 0.3, lag_ratio=0.15), FadeIn(lab_m), run_time=1.0)
            self.until(self.cue("Add up the lazy"))
            self.play(Create(lz_seg), run_time=0.6)
            self.play(FadeIn(sq_l, shift=DOWN * 0.3, lag_ratio=0.15), FadeIn(lab_l), run_time=1.0)
            # formula
            self.until(self.cue('Divide the first'))
            num = T("sum of the model's squared errors", 24, C_BAD)
            den = T("sum of the lazy guess's squared errors", 24, C_HI)
            fr = VGroup(num, Line(LEFT, RIGHT, stroke_color=INK, stroke_width=2).set_width(den.width + 0.3), den)
            fr.arrange(DOWN, buff=0.1)
            lhs = T('R²  =  1  −', 34, INK, weight='BOLD')
            form = VGroup(lhs, fr).arrange(RIGHT, buff=0.3).move_to([0, -3.0, 0])
            self.play(FadeIn(form, shift=UP * 0.15), run_time=0.8)
            # scale
            self.until(self.cue('If the model is perfect'))
            self.play(*[FadeOut(m) for m in (sq_m, sq_l, lab_m, lab_l, md_seg, lz_seg)], run_time=0.5)
            ln = Line([-5.6, -1.65, 0], [5.6, -1.65, 0], stroke_color=FAINT, stroke_width=3)
            pts = {1: 4.3, 0: 0.0, -1: -4.3}
            YS = -1.65
            d1 = Dot([pts[1], YS, 0], radius=0.14, color=C_GOOD)
            d0 = Dot([pts[0], YS, 0], radius=0.14, color=C_HI)
            dn = Dot([pts[-1], YS, 0], radius=0.14, color=C_BAD)
            l1 = VGroup(T('1', 40, C_GOOD, weight='BOLD'), T('perfect', 24, INK)).arrange(DOWN, buff=0.1)
            l0 = VGroup(T('0', 40, C_HI, weight='BOLD'), T('no better than', 24, INK), T('the flat line', 24, INK)
                        ).arrange(DOWN, buff=0.08)
            ln_ = VGroup(T('below 0', 34, C_BAD, weight='BOLD'), T('worse than', 24, INK), T('the flat line', 24, INK)
                         ).arrange(DOWN, buff=0.08)
            l1.next_to(d1, UP, buff=0.18)
            l0.next_to(d0, UP, buff=0.18)
            ln_.next_to(dn, UP, buff=0.18)
            self.play(Create(ln), run_time=0.3)
            self.play(FadeIn(d1), FadeIn(l1), run_time=0.5)
            self.until(self.cue('If it is no better'))
            self.play(FadeIn(d0), FadeIn(l0), run_time=0.5)
            self.until(self.cue('And if it is worse'))
            self.play(FadeIn(dn), FadeIn(ln_), run_time=0.5)
            self.until(self.cue('Note that this lazy guess'))
            wtxt = W('knows the true average of this very trial', 20, C_HI, width=chars_for(3.4, 20), align=ORIGIN)
            warn = VGroup(RoundedRectangle(corner_radius=0.1, width=3.8, height=wtxt.height + 0.3, fill_color=PANEL,
                                           fill_opacity=0.95, stroke_color=C_HI, stroke_width=2), wtxt)
            warn.move_to(pl3.p(4.2, -26))
            wtxt.move_to(warn[0])
            wlead = Line(warn[0].get_top(), pl3.p(4.2, mean_all), stroke_color=C_HI, stroke_width=2)
            warn.add(wlead)
            self.play(FadeIn(warn), Indicate(lazy, color=C_HI, scale_factor=1.0), run_time=0.8)

        # ───────────────────────── b04: same miss, strong vs weak position ─────────────────────────
        with self.beat('c11_b04'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            rng = np.random.default_rng(3)
            noise = np.convolve(rng.standard_normal(len(key) + 40), np.ones(25) / 25, mode='same')[20:20 + len(key)]
            noise = noise - noise.mean()
            noise *= 0.4 / np.mean(np.abs(noise))             # a miss of 0.4 Nm on average
            key_w = d['yes_m486_key']
            r2_strong = 1 - np.mean(noise ** 2) / np.var(key)
            r2_weak = 1 - np.mean(noise ** 2) / np.var(key_w)
            j0, j1 = 2000, 2600
            tt = t[j0:j1] - t[j0]

            def panel(kk, yr, yticks, cx, ttl):
                p_ = Plot((0, 6), yr, 5.0, 2.2, xticks=(0, 2, 4, 6), yticks=yticks, xlabel='time (s)', tick_size=18)
                p_.move_to([cx, 0.85, 0])
                lz = p_.hline(float(kk.mean()), C_HI, 2.5, dashed=True)
                trc = p_.curve(tt, kk[j0:j1], C_TQ, 2.6)
                prc = p_.curve(tt, (kk + noise)[j0:j1], C_PRED, 2.0)
                hh = abs(p_.p(0, 1)[1] - p_.p(0, 0)[1])
                sb = Line(p_.p(6, yr[0]) + RIGHT * 0.2, p_.p(6, yr[0]) + RIGHT * 0.2 + UP * hh, stroke_color=C_BAD,
                          stroke_width=6)
                sb_l = T('1 Nm', 18, C_BAD)
                t_ = T(ttl, 22, INK)
                yl_ = T('torque (Nm)', 18, DIM).rotate(PI / 2).next_to(p_, LEFT, buff=0.1)
                return p_, lz, trc, prc, sb, sb_l, t_, yl_

            L = panel(key, (-40, 5), (-40, -20, 0), -3.45, 'strong position: +0.138 rad')
            Rr = panel(key_w, (-4.2, -0.8), (-4, -3, -2, -1), 3.45, 'toes-down position: −0.486 rad')
            for P, cx in ((L, -3.45), (Rr, 3.45)):
                P[6].move_to([cx, 2.65, 0])
                P[5].next_to(P[4], UP, buff=0.05)
            lz_lab = T('lazy guess', 18, C_HI)
            self.play(*[FadeIn(P[0]) for P in (L, Rr)], *[FadeIn(P[6]) for P in (L, Rr)], *[FadeIn(P[7]) for P in (L, Rr)], run_time=0.6)
            self.play(Create(L[2]), Create(Rr[2]), run_time=0.8)
            self.until(self.cue('The lazy guess is very good'))
            lz_lab.move_to(L[0].p(3.1, float(key.mean()) - 7))
            self.play(Create(L[1]), Create(Rr[1]), FadeIn(lz_lab), run_time=0.7)
            capl = W('swings a lot: the flat line is far off', 20, INK, width=chars_for(5.0, 20), align=ORIGIN)
            capr = W('hardly moves: the flat line is already close', 20, INK, width=chars_for(5.0, 20), align=ORIGIN)
            capl.move_to([-3.45, -1.4, 0])
            capr.move_to([3.45, -1.4, 0])
            self.play(FadeIn(capl), FadeIn(capr), run_time=0.5)
            self.until(self.cue('the same error'))
            self.play(Create(L[3]), Create(Rr[3]), FadeIn(L[4]), FadeIn(Rr[4]), FadeIn(L[5]), FadeIn(Rr[5]),
                      run_time=0.9)
            miss = T('same miss in both panels: 0.4 Nm on average (illustration)', 22, C_BAD).move_to([0, -2.1, 0])
            self.play(FadeIn(miss), run_time=0.4)
            self.until(self.cue('excellent R²'))
            bl = VGroup(T('R²', 30, INK), T(fmt(r2_strong, 3), 40, C_GOOD, weight='BOLD')).arrange(RIGHT, buff=0.2)
            bl.move_to([-3.45, -2.8, 0])
            self.play(FadeIn(bl, shift=UP * 0.1), run_time=0.5)
            self.until(self.cue('a terrible one'))
            br = VGroup(T('R²', 30, INK), T(fmt(r2_weak, 2), 40, C_BAD, weight='BOLD')).arrange(RIGHT, buff=0.2)
            br.move_to([3.45, -2.8, 0])
            self.play(FadeIn(br, shift=UP * 0.1), run_time=0.5)
            self.until(self.cue('R² is a score relative'))
            rel = T('R² is relative to how much there was to predict', 24, C_HI).move_to([0, -3.5, 0])
            self.play(FadeIn(rel), run_time=0.5)

        # ───────────────────────── b05: pooled R² hides the weak position ─────────────────────────
        with self.beat('c11_b05'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            pooled = F['y07']['pooled']['per_subject']
            yes_p = pooled['YES']
            med_all = [pooled[s]['median'] for s in pooled]
            badge = card([T("old code's R²,", 24, DIM), T('all 8 positions pooled', 24, DIM),
                          T(f'{fmt(yes_p["min"], 2)}–{fmt(yes_p["max"], 2)}', 60, C_GOOD, weight='BOLD'),
                          T('YES: in all 10 seeds', 26, INK)], width=5.2, color=C_GOOD, pad=0.3)
            badge.content.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            badge.content.move_to(badge.box)
            badge.move_to([-3.9, 0.7, 0])
            r2m = d['r2_med'][5]
            r2lo = d['r2_min'][5]
            r2hi = d['r2_max'][5]
            pr_ = Plot((0, 8), (-2.2, 1.1), 5.4, 2.9, yticks=(-2, -1, 0, 1), tick_size=18)
            pr_.move_to([3.55, 0.85, 0])
            pr_.shift(RIGHT * 0.0)
            ttl_r = T('YES: R² of each position alone', 22, INK).move_to([3.55, 2.95, 0]).align_to(pr_, LEFT)
            zero = pr_.hline(0, FAINT, 1.5, dashed=False)
            bars_ = VGroup()
            for p in range(8):
                x0, x1 = p + 0.15, p + 0.85
                a, b = pr_.p(x0, 0), pr_.p(x1, r2m[p])
                bars_.add(Rectangle(width=abs(b[0] - a[0]), height=max(0.02, abs(b[1] - a[1])),
                                    fill_color=ANGLE_C[p], fill_opacity=0.9, stroke_width=0)
                          .move_to([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0]))
            whisk = VGroup(*[Line(pr_.p(p + 0.5, max(r2lo[p], -2.2)), pr_.p(p + 0.5, r2hi[p]), stroke_color=INK,
                                  stroke_width=2.5) for p in range(8)])
            xl = VGroup(*[T(fmt(yes_pos[p], 2), 18, DIM).rotate(PI / 2).move_to(
                [pr_.p(p + 0.5, 0)[0], pr_.p(0, -2.2)[1] - 0.42, 0]) for p in range(8)])
            xt = T('position (rad)', 18, DIM).next_to(xl, DOWN, buff=0.1)
            yt = T('R²', 18, DIM).next_to(pr_.frame[1], UP, buff=0.05).shift(RIGHT * 0.3)
            rng_lab = T(f'{fmt(r2lo[0], 1)} to {fmt(r2hi[0], 2)}', 22, C_BAD, weight='BOLD')
            rng_lab.move_to([pr_.p(1.6, -1.9)[0] + 0.85, pr_.p(0, -1.9)[1], 0])
            foot = W(f'pooled, typical run: about {fmt(min(med_all), 2)} to {fmt(max(med_all), 2)} per person', 24, INK,
                     width=chars_for(5.4, 24)).move_to([-3.9, -1.55, 0])
            pooln = W('pooling counts the big gaps between strong and weak positions as "explained" by the model', 22, C_HI,
                      width=chars_for(5.6, 22)).move_to([-3.9, -2.75, 0])
            self.until(self.cue("Last semester's code"))
            self.play(FadeIn(badge, shift=UP * 0.15), run_time=0.7)
            self.until(self.cue('while'))
            self.play(FadeIn(pr_), FadeIn(ttl_r), FadeIn(zero), FadeIn(xl), FadeIn(xt), FadeIn(yt), run_time=0.6)
            self.play(FadeIn(bars_, lag_ratio=0.1), FadeIn(whisk), run_time=1.0)
            self.play(FadeIn(rng_lab), run_time=0.4)
            self.until(self.cue('Pooling counts'))
            self.play(FadeIn(pooln, shift=UP * 0.1), FadeIn(foot), run_time=0.7)
            self.until(self.cue('That is why we score'))
            self.play(Indicate(pooln, color=C_HI, scale_factor=1.04), run_time=0.8)

        # ───────────────────────── b06: how the dashboard summarises ─────────────────────────
        with self.beat('c11_b06'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            r2 = d['r2']
            td = d['toes_down'].astype(bool)
            eg = r2[1, :, 0]
            floored = np.maximum(eg, 0)
            pyc = 0.65

            def mini(cx, xr=(0, 11)):
                p_ = Plot(xr, (-0.8, 1.0), 3.0, 2.3, yticks=(-0.5, 0, 0.5, 1.0), xlabel='seed', tick_size=18,
                          yfmt=lambda v: f'{v:g}')
                p_.move_to([cx, pyc, 0])
                return p_

            P1, P2 = mini(-4.5), mini(0.0)
            cell_ttl = T('example cell: EG at −0.739 rad, 10 seeds (one dot per seed)', 20, DIM).move_to([-2.2, -1.65, 0])
            ttl1 = W('per run: negative R² shown as 0', 22, INK, width=chars_for(3.9, 22), align=ORIGIN)
            ttl2 = W('per cell: median of the 10 seeds', 22, INK, width=chars_for(3.9, 22), align=ORIGIN)
            ttl3 = W('per group: median of the cells', 22, INK, width=chars_for(3.9, 22), align=ORIGIN)
            for tt_, cx in ((ttl1, -4.5), (ttl2, 0.0), (ttl3, 4.5)):
                tt_.move_to([cx, 2.55, 0])
            z1, z2 = P1.hline(0, FAINT, 1.5, dashed=False), P2.hline(0, FAINT, 1.5, dashed=False)
            raw_d = VGroup(*[Dot(P1.p(k + 1, v), radius=0.08, color=C_BAD if v < 0 else C_PRED)
                             for k, v in enumerate(eg)])
            fl_d = VGroup(*[Dot(P2.p(k + 1, v), radius=0.08, color=C_PRED) for k, v in enumerate(floored)])
            ylab = T('R²', 18, DIM)
            self.until(self.cue('Any negative R²'))
            self.play(FadeIn(P1), FadeIn(z1), FadeIn(ttl1), FadeIn(cell_ttl), run_time=0.5)
            self.play(FadeIn(raw_d, lag_ratio=0.1), run_time=0.8)
            self.play(*[dd.animate.move_to(P1.p(k + 1, floored[k])).set_color(C_PRED) for k, dd in enumerate(raw_d)],
                      run_time=0.9)
            self.until(self.cue('for each cell'))
            self.play(FadeIn(P2), FadeIn(z2), FadeIn(ttl2), run_time=0.5)
            self.play(FadeIn(fl_d), run_time=0.4)
            medv = float(np.median(floored))
            mline = Line(P2.p(0.3, medv), P2.p(10.7, medv), stroke_color=C_HI, stroke_width=4)
            mlab = T(f'median {fmt(medv, 2)}', 22, C_HI).next_to(mline, RIGHT, buff=-1.9).shift(UP * 0.0)
            mlab.move_to(P2.p(4.0, 0.86))
            self.play(Create(mline), FadeIn(mlab), run_time=0.7)
            self.until(self.cue('then the median over the cells'))
            P3 = Plot((0, 3), (0, 1.05), 2.8, 2.3, yticks=(0, 0.5, 1.0), tick_size=18, yfmt=lambda v: f'{v:g}')
            P3.move_to([4.6, pyc, 0]).shift(RIGHT * 0.1)
            med0 = d['r2_floor0_med']
            td_vals, ot_vals = med0[td], med0[~td]
            r_ = np.random.default_rng(1)
            dtd = VGroup(*[Dot(P3.p(0.8 + r_.uniform(-0.2, 0.2), v), radius=0.07, color=TD_C) for v in td_vals])
            dot_ = VGroup(*[Dot(P3.p(2.2 + r_.uniform(-0.2, 0.2), v), radius=0.07, color=OT_C) for v in ot_vals])
            m_td, m_ot = float(td_sum['r2_floor0']), float(ot_sum['r2_floor0'])
            assert abs(np.median(td_vals) - m_td) < 0.005 and abs(np.median(ot_vals) - m_ot) < 0.005
            ml_td = Line(P3.p(0.4, m_td), P3.p(1.2, m_td), stroke_color=TD_C, stroke_width=4)
            ml_ot = Line(P3.p(1.8, m_ot), P3.p(2.6, m_ot), stroke_color=INK, stroke_width=4)
            gl = VGroup(VGroup(T('toes-down', 18, TD_C), T('12 cells', 18, TD_C), T(fmt(m_td, 2), 26, TD_C, weight='BOLD'))
                        .arrange(DOWN, buff=0.05),
                        VGroup(T('other', 18, INK), T('36 cells', 18, INK), T(fmt(m_ot, 2), 26, INK, weight='BOLD'))
                        .arrange(DOWN, buff=0.05))
            gl[0].move_to([P3.p(0.8, 0)[0], P3.p(0, 0)[1] - 0.85, 0])
            gl[1].move_to([P3.p(2.2, 0)[0], P3.p(0, 0)[1] - 0.85, 0])
            vl = VGroup()
            self.play(FadeIn(P3), FadeIn(ttl3), run_time=0.5)
            self.play(FadeIn(dtd), FadeIn(dot_), FadeIn(gl), run_time=0.7)
            self.play(Create(ml_td), Create(ml_ot), FadeIn(vl), run_time=0.5)
            self.until(self.cue('Remember'))
            rem = T('toes-down = each person\'s two most toes-down positions: 6 × 2 = 12 cells', 22, C_HI)
            rem.move_to([0, -3.2, 0])
            self.play(FadeIn(rem, shift=UP * 0.1), run_time=0.6)

        # ───────────────────────── b07: baseline R² ─────────────────────────
        with self.beat('c11_b07'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            vals = [td_sum['r2_floor0'], ot_sum['r2_floor0']]
            g = bars(vals, ['toes-down (12 cells)', 'other (36 cells)'], [TD_C, OT_C], width=5.6,
                     height=3.0, vmax=1.0, value_fmt=lambda v: f'{v:.2f}', label_size=24, value_size=44, gap=0.9)
            g.move_to([0, 0.0, 0])
            ttl = T('baseline, median R² (10 seeds)', 28, INK).move_to([0, 3.0, 0])
            self.play(FadeIn(ttl), FadeIn(g.axis), FadeIn(g.labels), run_time=0.5)
            self.until(self.cue('At the thirty-six'))
            self.play(GrowFromEdge(g.bars[1], DOWN), FadeIn(g.vals[1]), run_time=0.9)
            self.until(self.cue('At the twelve'))
            self.play(GrowFromEdge(g.bars[0], DOWN), FadeIn(g.vals[0]), run_time=0.9)

        # ───────────────────────── b08: but the error in Nm ─────────────────────────
        with self.beat('c11_b08'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)

            def pair(vals, vmax, fmt_v, title, cx):
                gg = bars(vals, ['toes-down', 'other'], [TD_C, OT_C], width=3.4, height=2.3, vmax=vmax,
                          value_fmt=fmt_v, label_size=20, value_size=30, gap=0.5)
                gg.move_to([cx, -0.35, 0])
                tl = W(title, 22, INK, width=chars_for(4.0, 22), align=ORIGIN).move_to([cx, 2.3, 0])
                return gg, tl

            g1, t1 = pair([td_sum['mae_nm'], ot_sum['mae_nm']], 1.4, lambda v: f'{v:.2f}', 'error: MAE (Nm)', -4.6)
            g2, t2 = pair([td_sum['peak_torque_nm'], ot_sum['peak_torque_nm']], 24, lambda v: f'{v:.1f}',
                          'typical peak (Nm)', 0.0)
            pct = [td_sum['mae_over_peak'] * 100, ot_sum['mae_over_peak'] * 100]
            g3, t3 = pair(pct, 17, lambda v: f'{v:.1f} %', "each cell's MAE ÷ its own peak", 4.6)
            note2 = T('peak = deepest point of a push', 20, DIM).move_to([0, -2.55, 0])
            ratio = T('toes-down misses by 13 %: nearly three times as much', 22, C_HI).move_to([0, -3.3, 0])
            self.until(self.cue('But look at the error'))
            self.play(FadeIn(t1), FadeIn(g1.axis), FadeIn(g1.labels), run_time=0.5)
            self.play(*[GrowFromEdge(b, DOWN) for b in g1.bars], FadeIn(g1.vals), run_time=0.9)
            self.until(self.cue('A typical peak'))
            self.play(FadeIn(t2), FadeIn(g2.axis), FadeIn(g2.labels), FadeIn(note2), run_time=0.5)
            self.play(*[GrowFromEdge(b, DOWN) for b in g2.bars], FadeIn(g2.vals), run_time=0.9)
            self.until(self.cue('If you take each'))
            self.play(FadeIn(t3), FadeIn(g3.axis), FadeIn(g3.labels), run_time=0.5)
            self.play(*[GrowFromEdge(b, DOWN) for b in g3.bars], FadeIn(g3.vals), run_time=0.9)
            self.play(FadeIn(ratio), run_time=0.5)

        # ───────────────────────── b09: warning about the zero floor ─────────────────────────
        with self.beat('c11_b09'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            jm_r2 = float(d['r2_med'][4, 5])
            assert abs(jm_r2 + 54.30) < 0.01
            c = card([W('Showing R² below 0 as 0 hides how bad a cell is.', 30, INK, width=chars_for(10.8, 30))],
                     width=12.0, color=C_BAD, pad=0.4, title='Warning', title_color=C_BAD).move_to([0, 1.7, 0])
            left = VGroup(T('JM at −0.078 rad', 24, DIM), T(f'R² ≈ {fmt(jm_r2, 0)}', 56, C_BAD, weight='BOLD'),
                          T('the true median over 10 seeds', 22, DIM)).arrange(DOWN, buff=0.15)
            right = VGroup(T('dashboard, default setting', 24, DIM), T('0.00', 56, INK, weight='BOLD'),
                           T('negative scores floored at 0', 22, DIM)).arrange(DOWN, buff=0.15)
            VGroup(left, right).arrange(RIGHT, buff=2.4).move_to([0, -1.3, 0])
            arr = Arrow(left.get_right() + RIGHT * 0.2, right.get_left() + LEFT * 0.2, buff=0, color=C_HI,
                        stroke_width=5)
            self.play(FadeIn(c, shift=UP * 0.15), run_time=0.7)
            self.until(self.cue('One of'))
            self.play(FadeIn(left, shift=UP * 0.1), run_time=0.6)
            self.until(self.cue('and with'))
            self.play(GrowArrow(arr), FadeIn(right, shift=UP * 0.1), run_time=0.7)
