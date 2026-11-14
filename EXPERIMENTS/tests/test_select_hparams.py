import numpy as np

from EXPERIMENTS.analysis.select_hparams import choose, epoch_gain, finalists


def table(rows):
    """rows: {config: (n_params, {person: val_loss})}"""
    return rows


def test_picks_lowest_relative_validation_loss():
    t = {'base': (100, {'A': 1.0, 'B': 2.0, 'C': 4.0}),
         'good': (100, {'A': 0.5, 'B': 1.0, 'C': 2.0}),
         'bad': (100, {'A': 2.0, 'B': 4.0, 'C': 8.0})}
    best, chosen, summary = choose(t, baseline='base')
    assert best == 'good' and chosen == 'good'
    assert np.isclose(summary['good']['mean_log_ratio'], np.log(0.5))


def test_one_standard_error_rule_prefers_the_smaller_network():
    # 'big' is best on average but 'small' is within one standard error of it -> 'small' is chosen
    t = {'base': (100, {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}),
         'big': (1000, {'A': 0.80, 'B': 0.90, 'C': 0.70, 'D': 1.00}),
         'small': (200, {'A': 0.85, 'B': 0.92, 'C': 0.80, 'D': 0.95})}
    best, chosen, _ = choose(t, baseline='base')
    assert best == 'big' and chosen == 'small'


def test_scale_differences_between_people_do_not_dominate():
    # person C has losses 100x larger; relative (per-person) comparison keeps every person equal
    t = {'base': (100, {'A': 1.0, 'B': 1.0, 'C': 100.0}),
         'x': (100, {'A': 0.5, 'B': 0.5, 'C': 110.0}),
         'y': (100, {'A': 1.1, 'B': 1.1, 'C': 80.0})}
    best, _, _ = choose(t, baseline='base')
    assert best == 'x'


def test_finalists_are_the_top_k_plus_the_baseline():
    summary = {c: {'mean_log_ratio': m} for c, m in [('base', 0.0), ('a', -0.4), ('b', -0.1), ('c', -0.3), ('d', 0.2)]}
    assert finalists(summary, 'base', k=2) == ['a', 'c', 'base']
    assert finalists(summary, 'base', k=5) == ['a', 'c', 'b', 'base', 'd']   # baseline already in the top k: not repeated
    assert finalists(summary, 'base', k=2, also=['b']) == ['a', 'c', 'b', 'base']   # e.g. the grid's 1-SE pick
    assert finalists(summary, 'base', k=2, also=['c']) == ['a', 'c', 'base']


def test_epoch_gain_pairs_runs_by_person_and_seed():
    # person A: 400 epochs halves the loss on both seeds; person B: no change -> mean log ratio = log(0.5) / 2
    long = {('A', 0): 0.5, ('A', 1): 1.0, ('B', 0): 3.0, ('B', 1): 4.0, ('B', 2): 9.0}
    short = {('A', 0): 1.0, ('A', 1): 2.0, ('B', 0): 3.0, ('B', 1): 4.0}
    g = epoch_gain(long, short)
    assert np.isclose(g['mean_log_ratio'], np.log(0.5) / 2)
    assert g['n_pairs'] == 4 and g['better_people'] == 1 and g['n_people'] == 2
