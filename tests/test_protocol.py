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
