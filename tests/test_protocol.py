import hashlib

from EXPERIMENTS.protocol import protocol as PT


def test_protocol_loads_and_hashes():
    p = PT.load()
    assert p['id'] == 'study_v2'
    assert p['hash'] == hashlib.sha256(PT.PATH.read_bytes()).hexdigest()
    assert isinstance(p['deviations'], list)


def test_cells_and_gates():
    p = PT.load()
    cells = p['cells']
    assert sum(not c['gated'] for c in cells.values()) == 15
    assert sum(c['gated'] for c in cells.values()) == 12
    assert p['gates']['order'] == ['w200_full', 'w500_full', 'w1000_full']
    assert all(s >= 10 for seeds in p['final_seeds'].values() for s in seeds)
    assert p['fixed']['shuffle'] is False


def test_primary_direction_only():
    p = PT.load()
    assert p['primary_direction'] == 'S1toS2' and 'supporting_direction' not in p


def test_rounds_coarse_fine_full_budget():
    r = PT.load()['rounds']
    assert [r[k]['name'] for k in ('R1', 'R2', 'R3')] == ['coarse', 'fine', 'seed check']
    assert all(r[k]['protocol'] == 'study400' for k in r) and r['R3']['keep'] == 2


def test_review_rules_18_nov():
    p = PT.load()
    assert 'no 1-SE' in p['a0_rule'] and p['fine']['layers'] == 'same'
    assert p['gates']['reference'].startswith('one comparison') and 'both of its arms' in p['score_from_rule']
