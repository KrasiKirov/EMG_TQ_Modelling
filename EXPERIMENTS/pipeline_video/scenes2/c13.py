"""Chapter 13 - Overfitting, or a different session?"""
from manim import *
from style2 import *
from scipy.interpolate import CubicSpline

C_OFF = C_BAD


def put(plot, cx, cy):
    """Centre the plot FRAME (not the labels) at (cx, cy)."""
    x0, x1 = plot.x_range
    y0, y1 = plot.y_range
    mid = (plot.p(x0, y0) + plot.p(x1, y1)) / 2
    plot.shift(np.array([cx, cy, 0]) - mid)
    return plot


def legend_row(items, size=22):
    g = VGroup()
    for col, txt, kind in items:
        if kind == 'line':
            mark = Line(LEFT * 0.22, RIGHT * 0.22, stroke_color=col, stroke_width=4)
        elif kind == 'ring':
            mark = Circle(radius=0.11, stroke_color=col, stroke_width=3, fill_opacity=0)
        elif kind == 'dot':
            mark = Dot(radius=0.11, color=col)
        else:
            mark = Rectangle(width=0.26, height=0.26, fill_color=col, fill_opacity=1, stroke_width=0)
        g.add(VGroup(mark, T(txt, size, INK)).arrange(RIGHT, buff=0.14))
    return g.arrange(RIGHT, buff=0.55)


def over_bracket(x0, x1, y, h=0.14, color=C_HI):
    return VMobject(stroke_color=color, stroke_width=3).set_points_as_corners(
        [[x0, y - h, 0], [x0, y, 0], [x1, y, 0], [x1, y - h, 0]])


class Ch13(BeatScene):
    def construct(self):
        F = facts()
        Y4 = load('y04')
        D4 = F['y04']['display']
        subs = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']

        # ---------------------------------------------------------------- b01
        with self.beat('c13_b01'):
            tag = self.intro('c13')
            rng = np.random.default_rng(3)
            f_true = lambda x: 3.1 + 1.3 * np.sin(0.7 * x)
            x_tr = np.linspace(0.8, 9.2, 9)
            y_tr = f_true(x_tr) + rng.normal(0, 0.75, 9)
            cs = CubicSpline(x_tr, y_tr)
            xx = np.linspace(x_tr[0], x_tr[-1], 300)
            x_new = x_tr[:-1] + 0.5
            y_new = f_true(x_new) + np.random.default_rng(11).normal(0, 0.8, len(x_new))
            pl = Plot((0, 10), (0, 6.2), 9.0, 3.3, xlabel='input', ylabel='output', tick_size=22)
            put(pl, 0.2, 1.5)
            self.play(FadeIn(pl), run_time=0.7)
            self.until(self.cue('Overfitting means'))
            d_tr = VGroup(*[Dot(pl.p(a, b), radius=0.11, color=C_FIT) for a, b in zip(x_tr, y_tr)])
            self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in d_tr], lag_ratio=0.1), run_time=1.2)
            self.until(self.cue('memorises its training data'))
            curve = pl.curve(xx, cs(xx), C_PRED, 3.2, clip=(0, 6.2))
            self.play(Create(curve), run_time=1.6)
            cap = T('memorising instead of learning', 36, C_HI, weight='BOLD').move_to([0, -3.1, 0])
            self.play(FadeIn(cap, shift=UP * 0.15), run_time=0.6)
            self.until(self.cue('The tell-tale sign'))
            chip1 = T('on its training data: tiny error', 24, C_FIT).move_to([-3.3, -2.35, 0])
            self.play(FadeIn(chip1), run_time=0.5)
            self.until(self.cue('and much worse'))
            d_new = VGroup(*[Circle(radius=0.11, stroke_color=C_VAL, stroke_width=3.5, fill_opacity=0
                                    ).move_to(pl.p(a, b)) for a, b in zip(x_new, y_new)])
            sticks = VGroup(*[DashedVMobject(Line(pl.p(a, b), pl.p(a, float(cs(a))), stroke_color=C_OFF,
                                                  stroke_width=2.5), num_dashes=5) for a, b in zip(x_new, y_new)])
            chip2 = T('on new data: large error', 24, C_VAL).move_to([3.3, -2.35, 0])
            self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in d_new], lag_ratio=0.1), run_time=1.0)
            self.play(Create(sticks), FadeIn(chip2), run_time=0.8)
            leg = legend_row([(C_FIT, 'training data', 'dot'), (C_VAL, 'new data, same source', 'ring'),
                              (C_PRED, "the model's curve", 'line')], 20).move_to([0.2, -1.55, 0])
            self.play(FadeIn(leg), run_time=0.5)

        # ---------------------------------------------------------------- b02
        mae3 = [float(v) for v in Y4['kept_x00f_median_mae_fit_val_test']]
        with self.beat('c13_b02'):
            self.clear_all(keep=[tag])
            title = T('typical miss (MAE) at the kept epoch, dropout off', 24, INK).move_to([0.8, 3.05, 0])
            g = bars(mae3, ['fit windows', 'validation windows', 'session 2 (test)'], [C_FIT, C_VAL, C_TEST],
                     width=8.0, height=3.0, vmax=1.4, value_fmt=lambda v: f'{v:.2f} Nm', label_size=22, value_size=28)
            g.shift(np.array([-1.5, -1.0, 0]) - g.axis.get_center())
            foot = T('rebuilt copy · one seed per person · median over the 6 people', 22, DIM).move_to([0, -3.3, 0])
            self.play(FadeIn(title), FadeIn(g.axis), FadeIn(foot), run_time=0.6)
            self.until(self.cue('measure three errors'))
            self.play(FadeIn(g.labels), run_time=0.6)
            self.until(self.cue('On the fit windows'))
            self.play(GrowFromEdge(g.bars[0], DOWN), FadeIn(g.vals[0]), run_time=0.8)
            self.until(self.cue('On the validation windows'))
            self.play(GrowFromEdge(g.bars[1], DOWN), FadeIn(g.vals[1]), run_time=0.8)
            self.until(self.cue('On session 2'))
            self.play(GrowFromEdge(g.bars[2], DOWN), FadeIn(g.vals[2]), run_time=0.8)
            self.until(self.cue('about twice'))
            twice = VGroup(T(f'≈ {mae3[2] / mae3[1]:.0f}×', 34, C_TEST, weight='BOLD'),
                           T('the validation error', 24, C_TEST, weight='BOLD')).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            twice.next_to(g.bars[2], RIGHT, buff=0.35).set_y(0.9)
            self.play(FadeIn(twice, shift=LEFT * 0.2), run_time=0.6)

        # ---------------------------------------------------------------- b03
        with self.beat('c13_b03'):
            self.play(FadeOut(twice), run_time=0.4)
            top = lambda i: g.bars[i].get_top()
            xl, xm, xr = g.bars[0].get_center()[0], g.bars[1].get_center()[0], g.bars[2].get_center()[0]
            yb = max(top(0)[1], top(1)[1]) + 0.65
            br = over_bracket(xl, xm, yb)
            br_t = W('almost equal: not memorising individual windows', 22, C_HI, width=chars_for(5.2, 22),
                     align=ORIGIN).next_to(br, UP, buff=0.22)
            self.play(Create(br), FadeIn(br_t), run_time=0.8)
            self.until(self.cue('But validation is just'))
            yv = top(1)[1]
            dash = DashedVMobject(Line([xm + 1.0, yv, 0], [xr + 1.6, yv, 0], stroke_color=DIM, stroke_width=2),
                                  num_dashes=14)
            sh = DoubleArrow([xr + 1.45, yv, 0], [xr + 1.45, top(2)[1], 0], buff=0, color=C_OFF, stroke_width=5,
                             tip_length=0.18)
            sh_t = VGroup(T('session shift', 26, C_OFF, weight='BOLD'),
                          W('the drop happens between sessions', 22, INK, width=chars_for(3.5, 22))
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([xr + 3.2, (yv + top(2)[1]) / 2 + 0.2, 0])
            sh_t.shift(RIGHT * (xr + 1.75 - sh_t.get_left()[0]))
            note = T('validation = the end of the same trials: same electrode contact, passive table, angles',
                     20, DIM).move_to([0, -2.85, 0])
            self.play(FadeIn(note), run_time=0.6)
            self.until(self.cue('What we can say'))
            self.play(Create(dash), GrowFromCenter(sh), FadeIn(sh_t, shift=LEFT * 0.2), run_time=0.9)

        # ---------------------------------------------------------------- b04
        with self.beat('c13_b04'):
            self.clear_all(keep=[tag])
            ratios = [float(D4['ratios_test_over_val'][s]) for s in ['CH', 'YES', 'HM', 'IES', 'EG', 'JM']]
            names = ['CH', 'YES', 'HM', 'IES', 'EG', 'JM']
            g4 = bars(ratios, names, [C_TEST] * 6, width=8.4, height=3.2, vmax=7.0,
                      value_fmt=lambda v: f'{v:.1f}×', label_size=26, value_size=26)
            g4.shift(np.array([-1.5, -1.2, 0]) - g4.axis.get_center())
            title = T('test error ÷ validation error, per person', 24, INK).move_to([0.8, 3.05, 0])
            one_y = g4.axis.get_center()[1] + 1.0 / 7.0 * 3.2
            one = DashedVMobject(Line([g4.axis.get_left()[0], one_y, 0], [g4.axis.get_right()[0] + 0.3, one_y, 0],
                                      stroke_color=C_HI, stroke_width=2.5), num_dashes=36)
            one_t = W('1 = same error as on validation', 20, C_HI, width=chars_for(3.2, 20)).move_to(
                [g4.axis.get_right()[0] + 2.05, one_y + 0.05, 0])
            self.play(FadeIn(title), FadeIn(g4.axis), FadeIn(g4.labels), run_time=0.6)
            self.play(Create(one), FadeIn(one_t), run_time=0.7)
            self.until(self.cue('larger than the validation'))
            self.play(*[GrowFromEdge(b, DOWN) for b in g4.bars], FadeIn(g4.vals), run_time=1.5)
            self.until(self.cue('from 1.2 times'))
            self.play(Indicate(g4.bars[0], color=C_HI, scale_factor=1.05), run_time=0.8)
            self.until(self.cue('to 6 times'))
            self.play(Indicate(g4.bars[5], color=C_HI, scale_factor=1.05), run_time=0.8)
            self.until(self.cue('whose number'))
            flag = VGroup(T('inflated by the', 22, C_BAD), T('passive-table bug', 22, C_BAD)).arrange(
                DOWN, buff=0.06).move_to([g4.bars[5].get_center()[0] - 3.2, g4.bars[5].get_top()[1] - 0.35, 0])
            ar = Arrow(flag.get_right() + RIGHT * 0.05, [g4.bars[5].get_left()[0] - 0.05, flag.get_center()[1], 0],
                       buff=0.05, color=C_BAD, stroke_width=4)
            self.play(FadeIn(flag), GrowArrow(ar), run_time=0.7)

        # ---------------------------------------------------------------- b05
        with self.beat('c13_b05'):
            self.clear_all(keep=[tag])
            rows = {(r['subject'], r['rank']): r for r in F['y07']['per_trial']['rows']}
            order = [('EG', "At EG's most"), ('IES', 'For IES'), ('HM', 'For HM'), ('YES', 'YES at minus')]
            sc = 1.3
            zero_y = 0.95
            panels = {}
            for k, (sj, _) in enumerate(order):
                cx = -5.025 + k * 3.35
                r = rows[(sj, 0)]
                vals = [r['r2_fit'], r['r2_val'], r['r2_test']]
                box = RoundedRectangle(corner_radius=0.14, width=3.0, height=4.4, fill_color=PANEL, fill_opacity=1,
                                       stroke_color=FAINT, stroke_width=2).move_to([cx, 0.85, 0])
                ttl = T(f'{sj}  {fmt(r["angle_s2"], 3)} rad', 22, INK, weight='BOLD').move_to([cx, 2.75, 0])
                zl = Line([cx - 1.35, zero_y, 0], [cx + 1.35, zero_y, 0], stroke_color=DIM, stroke_width=1.5)
                grp = VGroup(box, ttl, zl)
                for j, (v, col, nm) in enumerate(zip(vals, [C_FIT, C_VAL, C_TEST], ['fit', 'val', 'test'])):
                    bx = cx + (j - 1) * 0.95
                    h = max(abs(v) * sc, 0.03)
                    rect = Rectangle(width=0.7, height=h, fill_color=col, fill_opacity=0.95, stroke_width=0)
                    rect.move_to([bx, zero_y + (h / 2 if v >= 0 else -h / 2), 0])
                    lab = T(fmt(v, 2), 20, INK)
                    lab.move_to([bx, (rect.get_top()[1] + 0.22) if v >= 0 else (rect.get_bottom()[1] - 0.22), 0])
                    nl = T(nm, 20, col).move_to([bx, -0.75, 0])
                    grp.add(rect, lab, nl)
                panels[sj] = (box, grp)
            cap = T('one run of the rebuilt copy  (seed 0)', 22, DIM).move_to([0, -2.0, 0])
            self.play(FadeIn(cap), run_time=0.4)
            for sj, ph in order:
                self.until(self.cue(ph))
                self.play(FadeIn(panels[sj][1], shift=UP * 0.15), run_time=0.7)
            ex = panels['YES'][0].copy().set_fill(opacity=0).set_stroke(C_HI, 4)
            ex_t = T('already drops within session 1 (validation 0.60)', 24, C_HI).move_to([0, -2.85, 0])
            self.play(Create(ex), FadeIn(ex_t), run_time=0.7)

        # ---------------------------------------------------------------- b06
        with self.beat('c13_b06'):
            self.clear_all(keep=[tag])
            ep = Y4['x00f_EG_epoch']
            fit, val, te = Y4['x00f_EG_mae_fit'], Y4['x00f_EG_mae_val'], Y4['x00f_EG_mae_test']
            best, tmin = int(Y4['x00f_EG_best_epoch']), int(Y4['x00f_EG_epoch_min_mae_test'])
            pl = Plot((0, 36), (0, 2.5), 8.0, 3.5, xticks=(0, 10, 20, 30), yticks=(0, 0.5, 1.0, 1.5, 2.0),
                      xlabel='epoch', ylabel='MAE (Nm)', tick_size=18)
            put(pl, -1.35, 1.05)
            ttl = T('EG  ·  rebuilt copy  ·  seed 0', 24, INK).move_to([-1.35, 3.2, 0])
            leg = legend_row([(C_FIT, 'fit', 'line'), (C_VAL, 'validation', 'line'), (C_TEST, 'test (session 2)', 'line')],
                             20).move_to(pl.p(18, 2.3))
            self.play(FadeIn(pl), FadeIn(ttl), FadeIn(leg), run_time=0.7)
            self.until(self.cue('Here is EG'))
            c_fit = pl.curve(ep, fit, C_FIT, 3.0)
            c_val = pl.curve(ep, val, C_VAL, 3.0)
            c_te = pl.curve(ep, te, C_TEST, 3.0)
            self.play(Create(c_fit), Create(c_val), run_time=1.3)
            self.until(self.cue('while the test'))
            self.play(Create(c_te), run_time=1.0)
            # right column
            pct = {}
            for s in subs:
                e_, t_ = Y4[f'x00f_{s}_epoch'], Y4[f'x00f_{s}_mae_test']
                b_ = int(Y4[f'x00f_{s}_best_epoch'])
                m_ = int(Y4[f'x00f_{s}_epoch_min_mae_test'])
                pct[s] = (m_ < b_, 1 - float(t_.min()) / float(t_[list(e_).index(b_)]))
            self.until(self.cue('In five of the six'))
            ring = Circle(radius=0.14, stroke_color=INK, stroke_width=3.5, fill_opacity=0).move_to(pl.p(tmin, te[tmin - 1]))
            ring_t = T(f'test lowest: epoch {tmin}', 20, INK).move_to(pl.p(tmin + 6.3, 1.85))
            self.play(Create(ring), FadeIn(ring_t), run_time=0.6)
            self.until(self.cue('than the one kept'))
            star = Star(n=5, outer_radius=0.2, color=C_HI, fill_opacity=1, stroke_width=0).move_to(pl.p(best, val[best - 1]))
            vl = pl.vline(best, C_HI, 1.5, y1=1.0)
            star_t = T(f'kept epoch {best}', 20, C_HI).move_to(pl.p(best + 5.0, 0.25))
            self.play(Create(vl), FadeIn(star, scale=2), FadeIn(star_t), run_time=0.7)
            rows_ = VGroup(T('test error at its best', 20, DIM), T('epoch vs the kept epoch', 20, DIM))
            rows_.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            lines = VGroup()
            for s in subs:
                earlier, v = pct[s]
                txt = f'{v * 100:.0f} % lower' if earlier else 'later epoch'
                col = C_HI if s == 'IES' else INK
                lines.add(VGroup(T(s, 24, col, weight='BOLD'), T(txt, 24, col)))
            for r_ in lines:
                r_[1].move_to(r_[0].get_left() + RIGHT * 1.15, aligned_edge=LEFT)
            lines.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            tb = card([rows_, lines], width=3.7, size=22, pad=0.25)
            tb.move_to([5.0, 1.0, 0])
            self.until(self.cue('usually only a few'))
            self.play(FadeIn(tb, shift=LEFT * 0.2), run_time=0.7)
            self.until(self.cue('We cannot choose'))
            warn = card([W('Choosing the epoch with the test would let the test shape the model.', 22, INK,
                           width=chars_for(3.2, 22))], width=3.7, size=22, color=C_BAD, pad=0.25).move_to([5.0, -2.2, 0])
            self.play(FadeIn(warn, shift=LEFT * 0.2), run_time=0.7)

        # ---------------------------------------------------------------- b07
        with self.beat('c13_b07'):
            self.clear_all(keep=[tag])
            dl = F['y07']['dropout_loss']['per_subject']
            head = T('printed training loss ÷ real fit error, at the kept epoch', 24, INK).move_to([-1.6, 3.05, 0])
            base_y, unit = -1.4, 3.0 / 12.0
            groups = VGroup()
            printed_bars, fit_bars, labels = VGroup(), VGroup(), VGroup()
            for gi, sj in enumerate(['HM', 'CH']):
                cx = -4.6 + gi * 3.2
                ratio = dl[sj]['ratio']
                for j, (v, col, txt) in enumerate(((1.0, C_FIT, '1×'), (ratio, C_OFF, f'{ratio:.1f}×'))):
                    h = v * unit
                    rect = Rectangle(width=1.0, height=h, fill_color=col, fill_opacity=0.95, stroke_width=0
                                     ).move_to([cx + (j - 0.5) * 1.2, base_y + h / 2, 0])
                    (printed_bars if j else fit_bars).add(rect)
                    labels.add(T(txt, 26, INK).next_to(rect, UP, buff=0.1))
                labels.add(T(sj, 28, INK, weight='BOLD').move_to([cx, base_y - 0.35, 0]))
            axis = Line([-6.2, base_y, 0], [-0.0, base_y, 0], stroke_color=FAINT, stroke_width=1.6)
            leg = legend_row([(C_OFF, 'loss printed in the log (dropout on)', 'box'),
                              (C_FIT, 'real fit error (dropout off)', 'box')], 20)
            leg.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-3.0, -2.6, 0])
            allr = '  ·  '.join(f'{s} {dl[s]["ratio"]:.1f}' for s in subs)
            small = T(f'all six people:  {allr}', 20, DIM).move_to([-1.6, -3.45, 0])
            self.play(FadeIn(head), FadeIn(axis), run_time=0.5)
            self.until(self.cue('the training loss printed'))
            self.play(*[GrowFromEdge(b, DOWN) for b in fit_bars], *[GrowFromEdge(b, DOWN) for b in printed_bars],
                      FadeIn(leg), run_time=1.2)
            self.until(self.cue('For this tiny network'))
            self.play(FadeIn(labels), FadeIn(small), run_time=0.8)
            self.until(self.cue('So a training loss'))
            take = card([T('training loss far above', 26, INK), T('validation loss?', 26, INK),
                         T('Not a warning sign.', 28, C_GOOD, weight='BOLD'), T('It is dropout.', 28, C_GOOD, weight='BOLD')],
                        width=4.2, size=26).move_to([4.5, 0.4, 0])
            self.play(FadeIn(take, shift=LEFT * 0.2), run_time=0.7)

        # ---------------------------------------------------------------- b08
        with self.beat('c13_b08'):
            self.clear_all(keep=[tag])
            D6 = load('y06')
            rates = [float(r) for r in D6['drop_rates']]
            maes = [float(m) for m in D6['drop_td_mae']]
            labs = [f'dropout {r:g}' + ('\n(rebuilt baseline)' if abs(r - 0.3) < 1e-9 else '') for r in rates]
            cols = [C_PRED if abs(r - 0.3) < 1e-9 else C_PASSIVE for r in rates]
            g8 = bars(maes, [l.replace('\n', ' ') for l in labs], cols, width=7.6, height=2.8, vmax=1.6,
                      value_fmt=lambda v: f'{v:.2f} Nm', label_size=20, value_size=24)
            g8.shift(np.array([-2.5, -1.0, 0]) - g8.axis.get_center())
            title = T('toes-down error (MAE, Nm)  ·  YES + EG  ·  2 seeds', 24, INK).move_to([-1.6, 3.05, 0])
            self.play(FadeIn(title), FadeIn(g8.axis), FadeIn(g8.labels), run_time=0.6)
            self.until(self.cue('raising dropout'))
            self.play(*[GrowFromEdge(b, DOWN) for b in g8.bars], FadeIn(g8.vals), run_time=1.3)
            self.until(self.cue('while the R²'))
            note = T('R² only moved within the seed noise', 22, DIM).move_to([-2.5, -2.9, 0])
            self.play(FadeIn(note), run_time=0.5)
            self.until(self.cue('Within session 1'))
            close = card([T('Within session 1', 26, C_GOOD, weight='BOLD'),
                          W('no general sign of overfitting', 26, INK, width=chars_for(3.7, 26)),
                          T('Across sessions', 26, C_BAD, weight='BOLD'),
                          VGroup(T('no transfer;', 26, INK), T('validation cannot see it', 24, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)],
                         width=4.3, size=26).move_to([4.6, 0.3, 0])
            self.play(FadeIn(close, shift=LEFT * 0.2), run_time=0.8)
