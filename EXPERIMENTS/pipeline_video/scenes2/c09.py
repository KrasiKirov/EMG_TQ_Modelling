from manim import *
from style2 import *

Y4 = load('y04')
F4 = facts()['y04']
SIG_C = [C_GM, C_GL, C_SOL, C_TA, C_ANGLE]
WGT = '#9CC4FF'                      # colour of "weights"

# ── real learning curve: HM, rebuilt copy X00f, seed 0 ───────────────────────
EP = Y4['x00f_HM_epoch']
VAL = Y4['x00f_HM_mse_val']                      # validation MSE (Nm²) per epoch
LR = Y4['x00f_HM_lr']
BEST = int(Y4['x00f_HM_best_epoch'])
CUTS = [int(c) for c in Y4['x00f_HM_lr_cut_triggered']]
N_EP = len(EP)
assert BEST == 53 and N_EP == 78 and CUTS == [20, 30, 44, 63, 73]
RUN_EPOCHS, RUN_BEST = Y4['x00_epochs_run'], Y4['x00_best_epoch']
SUBJ = [str(s) for s in Y4['x00_subjects']]
STEPS = F4['steps_per_epoch']
N_FIT, STEPS_N = STEPS['n_fit_windows'], STEPS['steps']
assert (N_FIT, STEPS_N) == (56896, 7112)
LR0, LRF = F4['hyperparameters']['LEARNING_RATE'], F4['hyperparameters']['LR_FACTOR']


def comma(n):
    return f'{int(n):,}'


def chip(i=0, w=0.5, h=0.34, color=C_FIT, op=1.0):
    """A little 'window' chip: rounded rectangle with five coloured stripes."""
    box = RoundedRectangle(corner_radius=0.05, width=w, height=h, stroke_color=color, stroke_width=1.5,
                           fill_color=PANEL, fill_opacity=1)
    st = VGroup(*[Rectangle(width=w - 0.14, height=h * 0.1, stroke_width=0, fill_color=SIG_C[r],
                            fill_opacity=0.3 + 0.7 * ((i * 3 + r * 2) % 5) / 4) for r in range(5)]
                ).arrange(DOWN, buff=h * 0.04).move_to(box)
    g = VGroup(box, st)
    g.box = box
    return g


def shuffle_icon(s=1.0, color=INK):
    a = Arrow([-0.35, 0.2, 0], [0.35, -0.2, 0], buff=0, color=color, stroke_width=3, tip_length=0.14)
    b = Arrow([-0.35, -0.2, 0], [0.35, 0.2, 0], buff=0, color=color, stroke_width=3, tip_length=0.14)
    return VGroup(a, b).scale(s)


def arrow(a, b, color=DIM, w=3):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=w, tip_length=0.16,
                 max_stroke_width_to_length_ratio=30, max_tip_length_to_length_ratio=0.6)


def rolling_median(v, k=9):
    out = np.array([np.median(v[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(v))])
    return out


class Ch09(BeatScene):
    def construct(self):
        # ============================================================ b01 : the loss
        with self.beat('c09_b01') as dur:
            tag = self.intro('c09')
            U = 0.7                                             # screen units per Nm
            head = T('training = finding good values for the 1,001 weights', 28).move_to([0, 2.65, 0])
            self.play(FadeIn(head), run_time=0.7)
            self.until(self.cue('Good is measured by the loss'))
            head2 = M('good  =  <span foreground="#9BE564">low loss</span>', 32).move_to(head)
            self.play(ReplacementTransform(head, head2), run_time=0.7)
            self.until(self.cue('For each window'))
            # --- one window: prediction vs answer
            ax = Arrow([-5.6, -1.2, 0], [-5.6, 2.0, 0], buff=0, color=FAINT, stroke_width=3, tip_length=0.15)
            ax_lab = T('torque (Nm)', 18, DIM).rotate(PI / 2).move_to([-6.0, 0.4, 0])
            x_d, y_ans, y_pred = -3.2, -0.5, -0.5 + 2 * U
            d_ans = Dot([x_d, y_ans, 0], radius=0.11, color=C_TQ)
            d_pred = Dot([x_d, y_pred, 0], radius=0.11, color=C_PRED)
            l_ans = T('answer', 20, C_TQ).next_to(d_ans, LEFT, buff=0.2)
            l_pred = T('prediction', 20, C_PRED).next_to(d_pred, LEFT, buff=0.2)
            gap = Line([x_d, y_ans, 0], [x_d, y_pred, 0], stroke_color=C_BAD, stroke_width=5)
            gap_lab = T('error = prediction − answer', 20, C_BAD).move_to([-3.2, y_pred + 0.5, 0]).align_to(l_pred, LEFT)
            self.play(FadeIn(ax), FadeIn(ax_lab), FadeIn(d_ans), FadeIn(l_ans), FadeIn(d_pred), FadeIn(l_pred),
                      run_time=0.8)
            self.play(Create(gap), FadeIn(gap_lab), run_time=0.8)
            sq = Square(2 * U, stroke_color=C_BAD, stroke_width=2, fill_color=C_BAD, fill_opacity=0.3)
            sq.move_to([x_d + U, (y_ans + y_pred) / 2, 0])
            sq_lab = T('error²', 24, INK).move_to(sq)
            self.play(GrowFromEdge(sq, LEFT), run_time=1.0)
            self.play(FadeIn(sq_lab), run_time=0.4)
            # --- three squares
            cost = {}
            xs = {1: 0.95, 2: 2.95, 3: 5.3}
            for m in (1, 2, 3):
                self.until(self.cue(f'A miss of {["one", "two", "three"][m - 1]}'))
                s = Square(m * U, stroke_color=C_BAD, stroke_width=2, fill_color=C_BAD, fill_opacity=0.35)
                s.move_to([xs[m], -0.5 + m * U / 2, 0])
                lab = VGroup(T(f'miss {m} Nm', 20, DIM), T(f'cost {m * m} Nm²', 22, INK, weight='BOLD')
                             ).arrange(DOWN, buff=0.06).move_to([xs[m], -1.05, 0])
                self.play(GrowFromEdge(s, DOWN), FadeIn(lab), run_time=0.8)
                cost[m] = VGroup(s, lab)
            # --- average
            self.until(self.cue('Then average over many windows'))
            rng = np.random.default_rng(9)
            sizes = rng.choice([0.18, 0.24, 0.3, 0.38, 0.5], 9)
            many = VGroup(*[Square(sz, stroke_color=C_BAD, stroke_width=1.5, fill_color=C_BAD, fill_opacity=0.35)
                            for sz in sizes]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN).move_to([-4.3, -2.65, 0])
            many_lab = T('many windows', 20, DIM).next_to(many, DOWN, buff=0.15)
            self.play(LaggedStart(*[FadeIn(q, shift=UP * 0.1) for q in many], lag_ratio=0.1), FadeIn(many_lab),
                      run_time=1.2)
            self.until(self.cue('This is the mean squared error'))
            ar = arrow([-2.2, -2.65, 0], [-1.5, -2.65, 0], INK)
            mse = M('<b>loss</b> = mean squared error (Nm²)', 30, C_HI).move_to([2.3, -2.65, 0])
            self.play(GrowArrow(ar), FadeIn(mse, shift=RIGHT * 0.15), run_time=1.0)
            self.until(self.cue('Squaring means'))
            self.play(Indicate(cost[3], color=C_HI, scale_factor=1.12), Indicate(cost[1], color=C_HI, scale_factor=1.2),
                      run_time=1.2)
        self.clear_all(keep=[tag])

        # ============================================================ b02 : one training step
        with self.beat('c09_b02'):
            rngp = np.random.default_rng(2)
            # fit-set pool
            pool = VGroup(*[RoundedRectangle(corner_radius=0.03, width=0.26, height=0.17, stroke_width=0,
                                             fill_color=C_FIT, fill_opacity=0.4) for _ in range(48)])
            pool.arrange_in_grid(8, 6, buff=(0.1, 0.11)).move_to([-5.0, 1.15, 0])
            pool_box = SurroundingRectangle(pool, buff=0.2, stroke_color=C_FIT, stroke_width=2, corner_radius=0.12)
            pool_lab = T('fit set', 22, C_FIT).next_to(pool_box, UP, buff=0.12)
            self.play(FadeIn(pool), FadeIn(pool_box), FadeIn(pool_lab), run_time=0.8)
            self.until(self.cue('Pick eight windows at random'))
            pick = sorted(rngp.choice(48, 8, replace=False).tolist())
            self.play(*[pool[i].animate.set_fill(C_HI, opacity=1) for i in pick], run_time=0.5)
            batch = VGroup(*[chip(i, 0.5, 0.2) for i in range(8)]).arrange(DOWN, buff=0.07).move_to([-2.5, 1.15, 0])
            batch_lab = T('batch = 8 windows', 20, INK).next_to(batch, DOWN, buff=0.2)
            self.play(LaggedStart(*[TransformFromCopy(pool[i], batch[k]) for k, i in enumerate(pick)], lag_ratio=0.1),
                      FadeIn(batch_lab), run_time=1.4)
            self.until(self.cue('Predict all eight'))
            net = VGroup(RoundedRectangle(corner_radius=0.14, width=1.5, height=1.2, stroke_color=INK, stroke_width=2,
                                          fill_color=PANEL, fill_opacity=1), T('network', 22, INK)).move_to([-0.2, 1.15, 0])
            ar1 = arrow([-1.9, 1.15, 0], net.get_left())
            pys = [batch[k].get_y() for k in range(8)]
            d_pred = VGroup(*[Dot([1.5, y, 0], radius=0.07, color=C_PRED) for y in pys])
            d_ans = VGroup(*[Dot([1.9, y, 0], radius=0.07, color=C_TQ) for y in pys])
            pa_lab = VGroup(T('predictions', 18, C_PRED), T('answers', 18, C_TQ)).arrange(RIGHT, buff=0.25
                                                                                           ).move_to([1.7, -0.45, 0])
            ar2 = arrow(net.get_right(), [1.2, 1.15, 0])
            self.play(FadeIn(net), GrowArrow(ar1), run_time=0.7)
            self.play(GrowArrow(ar2), LaggedStart(*[FadeIn(d, shift=RIGHT * 0.2) for d in d_pred], lag_ratio=0.08),
                      FadeIn(d_ans), FadeIn(pa_lab), run_time=1.2)
            self.until(self.cue('measure the loss'))
            loss = VGroup(RoundedRectangle(corner_radius=0.14, width=1.7, height=1.2, stroke_color=C_BAD, stroke_width=2,
                                           fill_color=PANEL, fill_opacity=1), T('loss', 30, C_BAD, weight='BOLD')
                          ).move_to([4.3, 1.15, 0])
            ar3 = arrow([2.3, 1.15, 0], loss.get_left())
            self.play(GrowArrow(ar3), FadeIn(loss), run_time=0.8)
            # weights
            self.until(self.cue('nudge every weight'))
            hs = rngp.uniform(0.25, 0.8, 18)
            wts = VGroup(*[Rectangle(width=0.15, height=h, stroke_width=0, fill_color=WGT, fill_opacity=0.85)
                           for h in hs]).arrange(RIGHT, buff=0.11, aligned_edge=DOWN).move_to([3.0, -2.7, 0])
            w_lab = T('the 1,001 weights (some shown)', 20, WGT).next_to(wts, DOWN, buff=0.15)
            ar4 = arrow([4.3, 0.5, 0], [4.3, -1.6, 0], C_BAD)
            self.play(FadeIn(wts), FadeIn(w_lab), GrowArrow(ar4), run_time=0.8)
            sig = rngp.choice([-1, 1], 18)
            nud = VGroup(*[Arrow(w.get_top() + UP * 0.08, w.get_top() + UP * 0.08 + (0.32 * s) * UP, buff=0,
                                 color=C_HI, stroke_width=3, tip_length=0.11) for w, s in zip(wts, sig)])
            hs2 = np.clip(hs + sig * 0.12, 0.15, 0.95)
            wts2 = VGroup(*[Rectangle(width=0.15, height=h, stroke_width=0, fill_color=WGT, fill_opacity=0.85)
                            for h in hs2])
            for new, old in zip(wts2, wts):
                new.move_to(old.get_bottom() + UP * new.height / 2)
            self.play(FadeIn(nud), run_time=0.5)
            self.play(Transform(wts, wts2), nud.animate.shift(UP * 0.0), run_time=0.8)
            self.until(self.cue('The rule that decides the nudge'))
            opt = M('<b>optimiser</b>: Nadam (a variant of Adam)', 24).move_to([-6.4, -1.7, 0]).align_to([-6.4, 0, 0], LEFT)
            self.play(FadeIn(opt, shift=RIGHT * 0.15), run_time=0.7)
            self.until(self.cue('the size of the nudge'))
            lr = M(f'<b>learning rate</b> (step size): <span foreground="#F0E442">{LR0}</span>', 24
                   ).move_to([-6.4, -2.5, 0]).align_to([-6.4, 0, 0], LEFT)
            self.play(FadeIn(lr, shift=RIGHT * 0.15), run_time=0.7)
        self.clear_all(keep=[tag])

        # ============================================================ b03 : one epoch
        with self.beat('c09_b03') as dur:
            BW, BX, BY = 11.0, -5.5, 0.9
            outline = RoundedRectangle(corner_radius=0.1, width=BW, height=0.55, stroke_color=FAINT, stroke_width=2
                                       ).move_to([BX + BW / 2, BY, 0])
            top_lab = VGroup(T(f'{comma(N_FIT)} fit windows', 32, C_FIT, weight='BOLD'),
                             T('taken 8 at a time', 26, INK)).arrange(RIGHT, buff=0.5, aligned_edge=DOWN
                                                                       ).move_to([0, 2.2, 0])
            brace = VGroup(Line([BX, BY + 0.5, 0], [BX + BW, BY + 0.5, 0]), Line([BX, BY + 0.4, 0], [BX, BY + 0.6, 0]),
                           Line([BX + BW, BY + 0.4, 0], [BX + BW, BY + 0.6, 0])).set_stroke(C_FIT, 2.5)
            self.until(self.cue('There are 56,896'))
            self.play(FadeIn(outline), FadeIn(top_lab, shift=DOWN * 0.15), Create(brace), run_time=1.0)
            alpha = ValueTracker(0.0)
            fill = always_redraw(lambda: Rectangle(width=max(1e-3, BW * alpha.get_value()), height=0.55,
                                                   stroke_width=0, fill_color=C_FIT, fill_opacity=0.9
                                                   ).move_to([BX + max(1e-3, BW * alpha.get_value()) / 2, BY, 0]))
            counter = always_redraw(lambda: VGroup(
                T('step', 26, DIM), T(f'{comma(round(alpha.get_value() * STEPS_N))}', 36, INK, weight='BOLD'),
                T(f'of {comma(STEPS_N)}', 26, DIM)).arrange(RIGHT, buff=0.2, aligned_edge=DOWN).move_to([0, 0.05, 0]))
            self.until(self.cue('Do that 7,112 times'))
            self.add(fill, counter)
            t_run = max(2.0, (self.cue('That is one epoch') - self.cue('Do that 7,112 times')) * dur - 0.3)
            self.play(alpha.animate.set_value(1.0), run_time=t_run, rate_func=linear)
            counter.clear_updaters()
            self.until(self.cue('That is one epoch'))
            ep_card = M(f'{comma(STEPS_N)} steps  =  <b>1 epoch</b>  =  every fit window seen once', 32, C_HI
                        ).move_to([0, -0.95, 0])
            self.play(FadeIn(ep_card, shift=UP * 0.15), run_time=0.8)
            self.until(self.cue('The same 7,112'))
            note = VGroup(T(f'{comma(STEPS_N)} windows per trial × 8 trials = {comma(N_FIT)} windows', 22, DIM),
                          T(f'{comma(N_FIT)} windows ÷ batches of 8 = {comma(STEPS_N)} steps', 22, DIM)
                          ).arrange(DOWN, buff=0.12).move_to([0, -1.95, 0])
            self.play(FadeIn(note), run_time=0.8)
            self.until(self.cue('At the end of each epoch'))
            stamp_t = T('QUIZ', 36, C_VAL, weight='BOLD')
            stamp = VGroup(RoundedRectangle(corner_radius=0.1, width=stamp_t.width + 0.5, height=0.8, stroke_color=C_VAL,
                                            stroke_width=4), stamp_t).rotate(0.1).move_to([-4.6, -3.1, 0])
            q_lab = T('validation error checked · no learning', 26, INK).next_to(stamp, RIGHT, buff=0.5)
            self.play(FadeIn(stamp, scale=1.6), run_time=0.5)
            self.play(FadeIn(q_lab, shift=RIGHT * 0.15), run_time=0.7)
        self.clear_all(keep=[tag])

        # ============================================================ b04 : a real learning curve
        with self.beat('c09_b04') as dur:
            P_W, P_H = 10.6, 3.0
            plot = Plot((0, 80), (0, 1.0), P_W, P_H, xticks=[0, 20, 40, 60, 80], yticks=[0, 0.25, 0.5, 0.75, 1.0],
                        xlabel='epoch', ylabel='validation error (MSE, Nm²)', tick_size=18, xfmt=lambda v: f'{v:g}',
                        yfmt=lambda v: f'{v:g}')
            plot.shift(np.array([-4.9, -0.9, 0]) - plot.p(0, 0))
            curve = plot.curve(EP, VAL, C_VAL, 3.0)
            ic = shuffle_icon(1.3)
            ic_cap = T('every epoch starts again: the windows are put in a new random order', 28)
            mid = VGroup(ic, ic_cap).arrange(DOWN, buff=0.4).move_to([0, 0.3, 0])
            self.play(FadeIn(mid, shift=UP * 0.1), run_time=0.8)
            self.until(self.cue('Epoch after epoch'))
            ic_s = shuffle_icon(0.5)
            ic_c = T('new random order every epoch', 20, DIM)
            corner = VGroup(ic_s, ic_c).arrange(RIGHT, buff=0.2).move_to([6.7, 3.05, 0], aligned_edge=RIGHT)
            self.play(FadeOut(mid), FadeIn(plot), FadeIn(corner), run_time=0.8)
            t_curve = max(2.0, (self.cue('Here is a real run') - self.cue('Epoch after epoch')) * dur - 1.2)
            self.play(Create(curve), run_time=t_curve, rate_func=linear)
            self.until(self.cue('Here is a real run'))
            title = T('rebuilt copy of the baseline · HM · one seed', 20, DIM).move_to([-4.9, 3.05, 0], aligned_edge=LEFT)
            self.play(FadeIn(title), run_time=0.6)
            self.until(self.cue('The validation error drops steeply'))
            band1 = plot.band(0.5, 4.5, DIM, 0.12)
            p1, p3 = plot.p(1, VAL[0]), plot.p(3, VAL[2])
            dots = VGroup(Dot(p1, radius=0.09, color=INK), Dot(p3, radius=0.09, color=INK))
            callout = T(f'epoch 1: {VAL[0]:.2f}  →  epoch 3: {VAL[2]:.2f} Nm²', 24, INK).move_to([-3.4, 2.55, 0],
                                                                                            aligned_edge=LEFT)
            self.play(FadeIn(band1), FadeIn(dots), FadeIn(callout), run_time=0.8)
            self.until(self.cue('After that, it jumps'))
            band2 = plot.band(4.5, 78.5, DIM, 0.10)
            self.play(FadeOut(band1), FadeOut(callout), FadeOut(dots), FadeIn(band2), run_time=0.7)
            self.until(self.cue('around a level'))
            trend_y = rolling_median(VAL, 11)
            trend = plot.curve(EP[5:], trend_y[5:], INK, 4.5)
            trend_lab = T('level that falls only slowly', 22, INK).move_to(plot.p(48, 0.78))
            self.play(Create(trend), FadeIn(trend_lab), run_time=1.5)
        self.play(FadeOut(band2), FadeOut(trend), FadeOut(trend_lab), run_time=0.5)

        # ============================================================ b05 : learning-rate cuts
        with self.beat('c09_b05') as dur:
            self.play(FadeOut(title), FadeOut(corner), VGroup(plot, curve).animate.shift(UP * 0.7), run_time=0.6)
            rule = M('no improvement for 10 epochs  →  learning rate <b>× 0.7</b>', 24, INK).move_to([0, 3.05, 0])
            lrp = Plot((0, 80), (0, 0.0035), P_W, 1.2, xticks=[0, 20, 40, 60, 80], yticks=[0, 0.001, 0.002, 0.003],
                       ylabel='learning rate', tick_size=18, xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
            lrp.shift(np.array([-4.9, -3.0, 0]) - lrp.p(0, 0))
            lrp.ylabel.shift(LEFT * 0.4)
            trained = [LR0] + [float(v) for v in LR[:-1]]       # rate each epoch was actually trained with
            xs, ys = [], []
            for e in range(N_EP):
                xs += [e + 0.5, e + 1.5]
                ys += [trained[e], trained[e]]
            stair = lrp.curve(xs, ys, C_HI, 3.5)
            self.until(self.cue('The first'))
            self.play(FadeIn(rule), FadeIn(lrp), run_time=0.8)
            self.play(Create(stair), run_time=1.0, rate_func=linear)
            self.until(self.cue('You can see where'))
            cut_marks, cut_labs, cut_lr = VGroup(), VGroup(), VGroup()
            for c in CUTS:
                cut_marks.add(plot.vline(c, INK, 1.8, y0=0, y1=1.0), lrp.vline(c, INK, 1.8))
                cut_labs.add(T('× 0.7', 18, INK).move_to(plot.p(c, 0.93)).shift(RIGHT * 0.38))
            self.play(LaggedStart(*[AnimationGroup(Create(cut_marks[2 * i]), Create(cut_marks[2 * i + 1]),
                                                   FadeIn(cut_labs[i])) for i in range(len(CUTS))], lag_ratio=0.25),
                      run_time=2.5)
        self.play(FadeOut(rule), FadeOut(cut_marks), FadeOut(cut_labs), FadeOut(lrp), FadeOut(stair), run_time=0.5)

        # ============================================================ b06 : early stopping
        with self.beat('c09_b06') as dur:
            grp = VGroup(plot, curve)
            self.play(grp.animate.scale(1.1).move_to([0.1, 0.35, 0]), run_time=0.8)
            self.until(self.cue('The second rule is early stopping'))
            rule2 = T('no improvement for 25 epochs  →  stop (here at epoch 78)', 24, INK).move_to([0, 3.05, 0])
            self.play(FadeIn(rule2), run_time=0.5)
            self.until(self.cue('the weights from the best epoch'))
            pb = plot.p(BEST, VAL[BEST - 1])
            star = Star(5, outer_radius=0.24, inner_radius=0.1, fill_color=C_HI, fill_opacity=1, stroke_color=INK, stroke_width=2).move_to(pb)
            star_lab = T(f'best validation epoch: {BEST}', 22, INK).move_to(plot.p(53, 0.5))
            self.play(FadeIn(star, scale=2.0), FadeIn(star_lab), run_time=0.9)
            # patience bracket
            p0, p1 = plot.p(BEST, 0.86), plot.p(N_EP, 0.86)
            br = VGroup(Line(p0, p1), Line(p0 + DOWN * 0.1, p0 + UP * 0.1), Line(p1 + DOWN * 0.1, p1 + UP * 0.1)
                        ).set_stroke(INK, 2.5)
            pat_lab = T('25 epochs of patience', 22, INK).next_to(br, UP, buff=0.1)
            sign = VGroup(RegularPolygon(8, radius=0.36, fill_color=C_BAD, fill_opacity=1, stroke_color=INK,
                                         stroke_width=2).rotate(PI / 8), T('STOP', 18, INK, weight='BOLD'))
            sign.move_to(plot.p(N_EP, 0.6))
            sign_ln = DashedVMobject(Line(sign.get_bottom() + DOWN * 0.02, plot.p(N_EP, VAL[-1]) + UP * 0.08,
                                          stroke_color=C_BAD, stroke_width=1.5), num_dashes=6)
            self.play(Create(br), FadeIn(pat_lab), run_time=0.8)
            self.play(FadeIn(sign, scale=1.8), Create(sign_ln), run_time=0.8)
            back = Arrow(sign.get_left() + LEFT * 0.05, plot.p(BEST + 1.2, VAL[BEST - 1] + 0.05), buff=0.05, color=C_GOOD,
                         stroke_width=4, tip_length=0.2, max_tip_length_to_length_ratio=0.5)
            back_lab = T('restore the best weights', 22, C_GOOD).move_to(plot.p(62, 0.7))
            self.play(GrowArrow(back), FadeIn(back_lab), run_time=1.0)
            self.until(self.cue('So the model we keep'))
            keep = M('the model we keep = the <b>best</b> validation epoch, not the last one', 26, C_HI).move_to([0, -3.35, 0])
            self.play(FadeIn(keep, shift=UP * 0.1), run_time=0.7)
        self.clear_all(keep=[tag])

        # ============================================================ b07 : all 60 runs
        with self.beat('c09_b07') as dur:
            head = M('the <b>60</b> baseline runs: 6 people × 10 seeds', 28).move_to([0, 2.75, 0])
            S_W, S_H = 9.8, 4.0
            sp = Plot((0, 200), (0, 66), S_W, S_H, xticks=[0, 50, 100, 150, 200], xlabel='epoch', tick_size=18,
                      xfmt=lambda v: f'{v:g}')
            sp.shift(np.array([-4.4, -2.65, 0]) - sp.p(0, 0))
            names = VGroup(*[T(s, 20, DIM).move_to([-4.85, sp.p(0, 11 * (5 - g) + 4.5)[1], 0]) for g, s in enumerate(SUBJ)])
            lim = sp.vline(200, DIM, 2.0, y0=0, y1=66)
            lim_lab = T('upper limit: 200 epochs', 20, DIM).move_to([sp.p(200, 66)[0] - 1.35, sp.p(0, 66)[1] + 0.3, 0])
            self.play(FadeIn(head), FadeIn(sp), FadeIn(names), run_time=0.8)
            self.until(self.cue('The limit is two hundred epochs'))
            self.play(Create(lim), FadeIn(lim_lab), run_time=0.8)
            self.until(self.cue('Across all sixty runs'))
            lines, dots = VGroup(), VGroup()
            for g in range(6):
                for s in range(10):
                    y = 11 * (5 - g) + s + 0.5
                    lines.add(Line(sp.p(0, y), sp.p(int(RUN_EPOCHS[g, s]), y), stroke_color=INK, stroke_width=2.4,
                                   stroke_opacity=0.55))
                    dots.add(Dot(sp.p(int(RUN_BEST[g, s]), y), radius=0.04, color=C_HI))
            self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.02), run_time=2.6)
            self.play(FadeIn(dots), run_time=0.8)
            self.until(self.cue('stopped between 32 and 115'))
            lo, hi = int(RUN_EPOCHS.min()), int(RUN_EPOCHS.max())
            blo, bhi = int(RUN_BEST.min()), int(RUN_BEST.max())
            assert (lo, hi, blo, bhi) == (32, 115, 7, 90)
            leg = VGroup(VGroup(Line(LEFT * 0.25, RIGHT * 0.25, stroke_color=INK, stroke_width=3),
                                T(f'stopped after {lo}–{hi} epochs', 22, INK)).arrange(RIGHT, buff=0.15),
                         VGroup(Dot(radius=0.07, color=C_HI), T(f'best epoch {blo}–{bhi}', 22, C_HI)
                                ).arrange(RIGHT, buff=0.15)).arrange(RIGHT, buff=0.8).move_to([0, 2.15, 0])
            self.play(FadeIn(leg, shift=UP * 0.1), run_time=0.8)
        self.clear_all(keep=[tag])

        # ============================================================ b08 : recap
        with self.beat('c09_b08') as dur:
            card_ = card([M('The model we keep = the <b>best validation epoch</b>', 28),
                          M('Validation = the <b>end of session 1\'s trials</b>', 28)], width=9.0).move_to([0, 2.0, 0])
            self.play(FadeIn(card_, shift=DOWN * 0.15), run_time=0.8)
            self.until(self.cue('And remember what validation is'))
            strip = VGroup()
            for i in range(8):
                f = Rectangle(width=0.9, height=0.5, stroke_width=0, fill_color=C_FIT, fill_opacity=0.9)
                v = Rectangle(width=0.25, height=0.5, stroke_width=0, fill_color=C_VAL, fill_opacity=1)
                f.move_to([0, 0, 0])
                v.next_to(f, RIGHT, buff=0)
                strip.add(VGroup(f, v))
            strip.arrange(RIGHT, buff=0.13).move_to([0, -0.15, 0])
            s_lab = T('session 1: 8 trials', 22, DIM).next_to(strip, UP, buff=0.2).align_to(strip, LEFT)
            leg2 = VGroup(VGroup(Square(0.22, stroke_width=0, fill_color=C_FIT, fill_opacity=0.9), T('fit', 20, INK)
                                 ).arrange(RIGHT, buff=0.12),
                          VGroup(Square(0.22, stroke_width=0, fill_color=C_VAL, fill_opacity=1),
                                 T('validation: the end of each trial', 20, INK)).arrange(RIGHT, buff=0.12)
                          ).arrange(RIGHT, buff=0.6).next_to(strip, DOWN, buff=0.3).align_to(strip, RIGHT)
            self.play(FadeIn(s_lab), LaggedStart(*[FadeIn(s) for s in strip], lag_ratio=0.08), FadeIn(leg2), run_time=1.5)
            self.until(self.cue('So the slowing down'))
            chips_ = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.12, width=3.5, height=0.8, stroke_color=C_VAL,
                                                      stroke_width=2, fill_color=PANEL, fill_opacity=1),
                                     T(t, 24, INK)) for t in ('when to slow down', 'when to stop', 'which version to keep')]
                              ).arrange(RIGHT, buff=0.3).move_to([0, -2.4, 0])
            for c in chips_:
                c[1].move_to(c[0])
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in chips_], lag_ratio=0.6), run_time=2.2)
            self.until(self.cue('are all decided by'))
            by = T('all decided by how well the model fits session 1', 24, C_HI).move_to([0, -3.4, 0])
            self.play(FadeIn(by), run_time=0.7)
