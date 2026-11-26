import numpy as np

from EXPERIMENTS.pipeline import scoring as SC


def test_person_scale_ies_metadata_target(synthetic_bundle):
    r = synthetic_bundle['active'][0]
    target = r['tq'] - 10.0 * r['angle']            # a passive model applied, no holds needed
    s = SC.person_scale(target)
    assert s > 0 and np.isclose(s, np.percentile(np.abs(target), 99))
