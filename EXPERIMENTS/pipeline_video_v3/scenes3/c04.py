import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, CurvedArrow, FadeIn, FadeOut, LaggedStart, Line, Rectangle,
                   RoundedRectangle, Star, Text, ValueTracker, VGroup, always_redraw, linear)

from common3 import place, push_bounds
from style3 import (BG, C_BAD, C_FIT, C_HI, C_PRED, C_TQ, C_VAL, DIM, FAINT, INK, MUSCLE_C, PANEL, BeatScene, Plot,
                    T, W, card, facts, load)


def deck_card(w=0.9, h=0.6, color=C_FIT):
    return RoundedRectangle(corner_radius=0.06, width=w, height=h, fill_color=PANEL, fill_opacity=1,
                            stroke_color=color, stroke_width=2)


class Ch04(BeatScene):
    def construct(self):
        F = facts()['v01']
        nw, ep = F['n_windows'], F['n0_epochs']
        z, lc = load('v01'), load('y04')

        with self.beat('c04_b01'):
            tag = self.intro('c04')
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

        with self.beat('c04_b02'):
            self.clear_all(keep=[tag])
            ps, _ = push_bounds(z, 23.5)
            i1 = int(np.searchsorted(z['eg_t'], ps + 0.6))
            win = np.column_stack([z['eg_env'][i1 - 19:i1 + 1], np.full(20, 0.5)])        # 20 × 5 (angle constant)
            cell, grid = 0.34, VGroup()
            colors = MUSCLE_C + ['#C9D1D9']
            for r in range(5):
                for c in range(20):
                    v = float(np.clip(win[c, r] / max(win[:, r].max(), 1e-6), 0.08, 1)) if r < 4 else 0.5
                    grid.add(Rectangle(width=cell, height=cell, stroke_color=BG, stroke_width=1.5, fill_color=colors[r],
                                       fill_opacity=v).move_to([(c - 9.5) * cell - 2.2, (2 - r) * cell + 1.2, 0]))
            rl = VGroup(*[T(n, 18, colors[r]).next_to(grid[r * 20], LEFT, buff=0.15)
                          for r, n in enumerate(['GM', 'GL', 'soleus', 'TA', 'angle'])])
            tl = T('oldest  →  newest  (0.2 s, in time order)', 20, DIM).next_to(grid, DOWN, buff=0.15)
            mem = VGroup(*[Rectangle(width=0.36, height=0.36, stroke_color=C_PRED, stroke_width=2, fill_color=C_PRED,
                                     fill_opacity=0) for _ in range(8)]).arrange(RIGHT, buff=0.08).move_to([4.8, 1.2, 0])
            ml = T('model memory (8 numbers)', 20, C_PRED).next_to(mem, UP, buff=0.15)
            self.play(FadeIn(grid), FadeIn(rl), FadeIn(tl), FadeIn(mem), FadeIn(ml), run_time=0.8)
            self.until(self.cue('and the model reads'))
            hi = Rectangle(width=cell, height=5 * cell, stroke_color=C_HI, stroke_width=3).move_to(
                [grid[0].get_x(), grid[40].get_y(), 0])
            self.add(hi)
            rng = np.random.default_rng(1)
            for c in range(20):
                self.play(hi.animate.set_x(grid[c].get_x()),
                          *[m.animate.set_fill(opacity=float(rng.uniform(0.15, 0.9))) for m in mem], run_time=0.17)
            self.until(self.cue('Its memory starts empty'))
            self.play(FadeOut(hi), *[m.animate.set_fill(opacity=0) for m in mem], run_time=0.6)
            nxt = W('next window: memory starts empty again', 22, C_PRED, width=24).next_to(mem, DOWN, buff=0.3)
            self.play(FadeIn(nxt), run_time=0.5)
            self.until(self.cue('So the order'))
            rules = VGroup(T('shuffled: the order of the windows', 24, C_FIT),
                           T('never shuffled: the order inside a window', 24, C_HI)).arrange(DOWN, buff=0.15,
                                                                                            aligned_edge=LEFT)
            rules.move_to([0, -2.6, 0])
            self.play(FadeIn(rules), run_time=0.6)
            self.wait(self.left())

        with self.beat('c04_b03'):
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
            self.until(self.cue('When the validation error'))
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
