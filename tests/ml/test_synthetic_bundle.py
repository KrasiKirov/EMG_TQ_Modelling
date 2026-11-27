def test_synthetic_bundle_shapes(synthetic_bundle):
    b = synthetic_bundle
    assert len(b['active']) == 16
    assert all(r['env_raw'].shape == (9000, 4) for r in b['active'])
    assert {r['session'] for r in b['active']} == {'S1', 'S2'}


def test_patch_subject(patch_subject):
    from ML.dataset import prepare as RR
    assert RR.load_subject('SYN')['subject'] == 'SYN'
