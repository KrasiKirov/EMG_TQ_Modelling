import numpy as np
from manim import (DOWN, DR, LEFT, RIGHT, UP, Arrow, Create, CurvedArrow, DashedLine, FadeIn, FadeOut, LaggedStart, Line, Rectangle,
                   RoundedRectangle, Star, Text, ValueTracker, VGroup, always_redraw, linear)

from common4 import place, push_bounds
from style4 import (ANGLE_C, BG, C_BAD, C_FIT, C_HI, C_PRED, C_TQ, C_VAL, DIM, FAINT, INK, MUSCLE_C, PANEL, BeatScene, Plot,
                    T, W, card, facts, load)


def deck_card(w=0.9, h=0.6, color=C_FIT):
    return RoundedRectangle(corner_radius=0.06, width=w, height=h, fill_color=PANEL, fill_opacity=1,
                            stroke_color=color, stroke_width=2)


class Ch05(BeatScene):
    def construct(self):
        F = facts()['v01']
        nw, ep = F['n_windows'], F['n0_epochs']
        z, lc = load('v01'), load('y04')

        with self.beat('c05_b01'):
            tag = self.intro('c05')
            rng = np.random.default_rng(3)
            deck = VGroup(*[deck_card().shift(UP * 0.05 * k + RIGHT * 0.03 * k) for k in range(12)]).move_to([-5.3, 0.6, 0])
            dl = W(f'{nw["train"]:,} fit windows', 22, C_FIT, width=16, align=LEFT).next_to(deck, DOWN, buff=0.3)
            self.play(FadeIn(deck), FadeIn(dl), run_time=0.7)
            for _ in range(2):                                      # shuffle
                moves = [c.animate.shift(RIGHT * rng.uniform(-0.6, 0.6) + UP * rng.uniform(-0.4, 0.4)) for c in deck]
                self.play(*moves, run_time=0.45)
                self.play(*[c.animate.move_to(deck[0].get_center() + UP * 0.05 * k + RIGHT * 0.03 * k)
                            for k, c in enumerate(deck)], run_time=0.45)
            self.until(self.cue('A group of eight'))
            batch = VGroup(*[deck_card(0.62, 0.45) for _ in range(8)]).arrange(RIGHT, buff=0.1).move_to([-1.2, 2.2, 0])
            bl = T('a batch = 8 windows', 22, C_FIT).next_to(batch, UP, buff=0.12)
            self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 2) for b in batch], lag_ratio=0.12), FadeIn(bl), run_time=1.2)
            self.until(self.cue('The model predicts'))
            model = card(['LSTM'], width=2.2, title='model', size=24).move_to([-1.2, 0.2, 0])
            a1 = Arrow(batch.get_bottom(), model.get_top(), buff=0.1, color=DIM, stroke_width=3)
            out = card([f'8 predictions vs 8 answers → error'], width=4.4, size=22).move_to([3.7, 0.2, 0])
            a2 = Arrow(model.get_right(), out.get_left(), buff=0.1, color=DIM, stroke_width=3)
            adj = CurvedArrow(out.get_bottom() + DOWN * 0.05, model.get_bottom() + DOWN * 0.05, angle=-1.2, color=C_HI)
            al = T('adjust its numbers a little', 22, C_HI).next_to(adj, DOWN, buff=0.1)
            self.play(FadeIn(model), FadeIn(a1), run_time=0.6)
            self.play(FadeIn(a2), FadeIn(out), run_time=0.6)
            self.play(FadeIn(adj), FadeIn(al), run_time=0.6)
            self.until(self.cue('After 7,112 batches'))
            k = ValueTracker(0)
            frame = Rectangle(width=10, height=0.3, stroke_color=FAINT, stroke_width=2).move_to([0, -3.0, 0])
            fill = always_redraw(lambda: Rectangle(width=max(0.01, 10 * k.get_value() / ep['steps_per_epoch']), height=0.3,
                                                   fill_color=C_FIT, fill_opacity=0.9, stroke_width=0)
                                 .align_to(frame, LEFT).set_y(-3.0))
            cnt = always_redraw(lambda: T(f'batch {int(k.get_value()):,} of {ep["steps_per_epoch"]:,}', 22, INK)
                                .next_to(frame, UP, buff=0.12))
            self.add(frame, fill, cnt)
            self.play(k.animate.set_value(ep['steps_per_epoch']), run_time=min(2.5, self.left() - 1.2), rate_func=linear)
            fill.clear_updaters()
            cnt.clear_updaters()
            self.play(FadeIn(T('= 1 epoch: every fit window used once', 24, C_HI).next_to(frame, DOWN, buff=0.12)), run_time=0.5)
            self.wait(self.left())

        with self.beat('c05_b02'):
            self.clear_all(keep=[tag])
            bz, bf = load('w04'), facts()['w04']
            ys = [2.45 - 0.7 * k for k in range(8)]
            colors = MUSCLE_C + ['#C9D1D9']
            cw, ch = 0.085, 0.09
            hdr = VGroup(*[T(s_, 17, DIM).move_to([x, 3.05, 0]) for s_, x in
                           (('window', -5.75), ('20 × 5 inputs', -3.7), ('same network, own memory', -1.0),
                            ('predicted', 1.35), ('measured', 2.75), ('error', 4.0))])
            labels, grids, boxes, mems, preds, meas, errs = (VGroup() for _ in range(7))
            for k, y in enumerate(ys):
                w = bz['win'][k].astype(float)
                labels.add(T(f"{bz['angle'][k]:+.3f} rad · {bz['t'][k]:.1f} s".replace('-', '−'), 18, ANGLE_C[int(bz['angle_rank'][k])])
                           .move_to([-5.75, y, 0]))
                g = VGroup()
                for r in range(5):
                    col = w[:, r]
                    rng_ = (col.max() - col.min()) or 1
                    for c in range(20):
                        v = 0.15 + 0.85 * (col[c] - col.min()) / rng_ if r < 4 else 0.5
                        g.add(Rectangle(width=cw, height=ch, stroke_width=0, fill_color=colors[r], fill_opacity=float(v))
                              .move_to([-3.7 + (c - 9.5) * (cw + 0.006), y + (2 - r) * (ch + 0.006), 0]))
                grids.add(g)
                box = RoundedRectangle(corner_radius=0.08, width=2.6, height=0.56, fill_color=PANEL, fill_opacity=1,
                                       stroke_color=C_PRED, stroke_width=1.6).move_to([-1.0, y, 0])
                boxes.add(VGroup(box, T('LSTM', 16, C_PRED).move_to([-1.85, y, 0])))
                mems.add(VGroup(*[Rectangle(width=0.13, height=0.26, stroke_color=C_PRED, stroke_width=1, fill_color=C_PRED,
                                            fill_opacity=0) for _ in range(8)]).arrange(RIGHT, buff=0.04).move_to([-0.6, y, 0]))
                pw = bf['windows'][k]
                preds.add(T(f"{pw['pred_Nm']:.1f}".replace('-', '−'), 22, C_PRED).move_to([1.35, y, 0]))
                meas.add(T(f"{pw['torque_Nm']:.1f}".replace('-', '−'), 22, C_TQ).move_to([2.75, y, 0]))
                errs.add(T(f"{abs(pw['pred_Nm'] - pw['torque_Nm']):.1f}", 20, C_HI).move_to([4.0, y, 0]))
            self.play(FadeIn(hdr[0]), FadeIn(hdr[1]), run_time=0.4)
            self.until(self.cue('The shuffle deals'))
            self.play(LaggedStart(*[FadeIn(VGroup(l, g), shift=RIGHT * 0.4) for l, g in zip(labels, grids)], lag_ratio=0.12),
                      run_time=1.6)
            self.until(self.cue('The model runs the same'))
            self.play(FadeIn(hdr[2]), LaggedStart(*[FadeIn(VGroup(b, m)) for b, m in zip(boxes, mems)], lag_ratio=0.08),
                      run_time=1.0)
            same = T('same 1,001 numbers in every copy', 18, C_PRED).move_to([-1.0, -3.15, 0])
            self.play(FadeIn(same), run_time=0.4)
            self.until(self.cue('Each copy starts'))
            hl = VGroup(*[Rectangle(width=cw + 0.02, height=5 * (ch + 0.006) + 0.04, stroke_color=C_HI, stroke_width=2)
                          .move_to(g[40].get_center()).set_x(g[0].get_x()) for g in grids])
            self.add(hl)
            dt = max(0.08, min(0.15, (self.left() - 2) / 20))
            for c in range(20):
                self.play(*[h.animate.set_x(g[c].get_x()) for h, g in zip(hl, grids)],
                          *[sq.animate.set_fill(opacity=0.12 + 0.75 * (c + 1) / 20 * (0.5 + 0.5 * np.sin(1.7 * j + k)))
                            for k, m in enumerate(mems) for j, sq in enumerate(m)], run_time=dt)
            self.play(FadeOut(hl), run_time=0.3)
            self.until(self.cue('The eight windows never'))
            seps = VGroup(*[DashedLine([-2.4, y - 0.35, 0], [0.4, y - 0.35, 0], stroke_color=FAINT, stroke_width=1.4)
                            for y in ys[:-1]])
            nev = T('the 8 windows never share anything', 20, C_HI).move_to([-1.0, -3.15, 0])
            self.play(FadeIn(seps), FadeOut(same), FadeIn(nev), run_time=0.6)
            self.until(self.cue('Out come eight'))
            self.play(FadeIn(hdr[3]), FadeIn(hdr[4]), LaggedStart(*[FadeIn(VGroup(p_, m_)) for p_, m_ in zip(preds, meas)],
                                                                   lag_ratio=0.08), run_time=1.0)
            self.play(FadeIn(hdr[5]), FadeIn(errs), run_time=0.5)
            self.until(self.cue('The errors are averaged'))
            one = card(['one number:', 'the average squared error'], width=2.6, size=20, color=C_HI).move_to([5.75, 0.0, 0])
            self.play(*[e.animate.move_to(one.get_center()).set_opacity(0) for e in errs], FadeIn(one), run_time=0.9)
            self.until(self.cue('Then all one thousand'))
            nudge = W('nudge all 1,001 numbers a little', 20, C_HI, width=16).next_to(one, DOWN, buff=0.35)
            self.play(FadeIn(nudge), run_time=0.6)
            self.play(*[b[0].animate.set_stroke(C_HI, width=2.6) for b in boxes], run_time=0.4)
            self.play(*[b[0].animate.set_stroke(C_PRED, width=1.6) for b in boxes], run_time=0.4)
            foot = T('predictions here: the trained model, for illustration', 16, DIM).to_corner(DR, buff=0.25)
            self.play(FadeIn(foot), run_time=0.3)
            self.wait(self.left())

        with self.beat('c05_b03'):
            self.clear_all(keep=[tag])
            e = lc['x00f_HM_epoch'].astype(float)
            v = lc['x00f_HM_mse_val'].astype(float)
            best = int(lc['x00f_HM_best_epoch'])
            p = place(Plot((0, 80), (0, 1.0), 11.0, 4.0, xticks=[1, 20, 40, 53, 60, 78], yticks=[0, 0.5, 1.0],
                           xlabel='epoch', ylabel='validation error', yfmt=lambda x: f'{x:g}'), -5.2, -2.0)
            curve = p.curve(e, v, C_VAL, 3)
            cap = T('a real run · rebuilt copy of last semester’s model · person HM', 20, DIM).next_to(p, UP, buff=0.1)
            self.play(FadeIn(p), FadeIn(cap), run_time=0.6)
            self.play(FadeIn(curve), run_time=1.2)
            self.until(self.cue('When the error stops'))
            cuts = VGroup()
            for c in (20, 30, 44, 63, 73):
                cuts.add(p.vline(c, DIM, 1.5, y0=0, y1=0.95), T('×0.7', 16, DIM).move_to(p.p(c, 0.98)))
            self.play(FadeIn(cuts), run_time=0.8)
            self.until(self.cue('After twenty-five epochs'))
            star = Star(n=5, outer_radius=0.16, color=C_HI, fill_opacity=1).move_to(p.p(best, v[best - 1]))
            br = Line(p.p(best, 0.72), p.p(78, 0.72), stroke_color=C_HI, stroke_width=3)
            brl = T('25 epochs without improvement', 18, C_HI).next_to(br, UP, buff=0.08)
            stop = T('stop', 20, C_BAD, weight='BOLD').next_to(p.p(78, 0.55), UP, buff=0.05)
            keep = T(f'kept: epoch {best}', 20, C_HI).next_to(star, DOWN, buff=0.15)
            self.play(FadeIn(star, scale=2), FadeIn(keep), run_time=0.6)
            self.play(FadeIn(br), FadeIn(brl), FadeIn(stop), run_time=0.7)
            self.until(self.cue('For the final model'))
            foot = T(f'final model, {ep["runs"]} runs: stopped after {ep["epochs_min"]}–{ep["epochs_max"]} epochs · '
                     f'kept epoch {ep["best_min"]}–{ep["best_max"]} · limit {ep["epoch_limit"]}', 22, INK)\
                .next_to(p, DOWN, buff=0.15)
            self.play(FadeIn(foot), run_time=0.6)
            self.wait(self.left())
