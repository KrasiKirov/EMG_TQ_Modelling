from EXPERIMENTS.analysis import verify_study as V


def facets(**kw):
    f = {'length_s': 0.2, 'step_ms': 10, 'overlap_pct': 95, 'target': 'active', 'state': 'reset', 'chunk_s': None,
         'shuffle': False, 'epochs': 150}
    f.update(kw)
    return f


def test_a_correctly_labelled_setting_passes():
    assert V.check_facets('W00_reference', facets(), 'Reference') == []
    assert V.check_facets('WG_0500ms_o75', facets(length_s=0.5, step_ms=50, overlap_pct=76), '500 ms · 76 %') == []
    assert V.check_facets('WO4_00_tot', facets(overlap_pct=0, target='total'), 'Overlap 0 % (total)') == []


def test_a_run_that_does_not_match_its_id_is_caught():
    # an "overlap 50 %" id whose run kept every window
    assert any('overlap_pct' in p for p in V.check_facets('WO3_50', facets(), 'Overlap 50 %'))
    # a total-torque id trained on active torque
    assert any('target' in p for p in V.check_facets('WL2_1s_tot', facets(length_s=1.0, step_ms=50, overlap_pct=99), 'Length 1 s (total)'))


def test_a_title_that_does_not_match_the_run_is_caught():
    probs = V.check_facets('WG_1s_o90', facets(length_s=1.0, step_ms=50, overlap_pct=90), '2 s · 90 %')
    assert any('title' in p and 'length_s' in p for p in probs)


def test_unknown_ids_are_reported_not_ignored():
    assert V.check_facets('WX9_new', facets(), 'New') == ['WX9_new: id not recognised by the verifier']
