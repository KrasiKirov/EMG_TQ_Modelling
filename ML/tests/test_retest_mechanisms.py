"""Validation isolation, alignment and population-error decomposition checks."""
import contextlib
import io
import unittest
import numpy as np
import pandas as pd
from ML.preprocessing.dataset_builder import build_dataset, build_windows, lookup_passive_torque
from ML.training.retest_mechanisms import metrics, bounded_weights


def trial(index,session):
    rng=np.random.default_rng(index);n=20000;t=np.arange(n)*.001
    d=pd.DataFrame({'time':t,'position':np.full(n,-.2),'torque':np.sin(t)})
    for c in ['gm','gl','sol','ta']:d[c]=rng.normal(size=n)
    d.attrs.update(trial_index=index,comment=session+', p1',domainIncr=.001)
    return d


class RetestMechanismTests(unittest.TestCase):
    def test_population_decomposition_and_signed_bias(self):
        y=np.array([1.,2.,3.,5.]);p=y+np.array([1.,2.,3.,4.])
        m=metrics(y,p)
        self.assertAlmostEqual(m['bias'],2.5)
        self.assertAlmostEqual(m['mse'],m['bias']**2+m['residual_sd']**2)
        self.assertAlmostEqual(m['target_sd'],y.std(ddof=0))
        with self.assertRaises(ValueError):metrics(y,p[:,None])

    def test_window_endpoint_and_passive_interpolation(self):
        d=trial(0,'test').iloc[:100].copy()
        for c in ['gm','gl','sol','ta']:d[c+'_env_norm']=np.arange(100)
        X,y,_=build_windows(d,['gm','gl','sol','ta'],[(-.3,1.),(-.1,3.)],window=50)
        self.assertEqual(X.shape,(51,50,5))
        np.testing.assert_allclose(y,d.torque.to_numpy()[49:]-2.,atol=3e-7)
        self.assertEqual(X[0,-1,0],49)
        self.assertAlmostEqual(lookup_passive_torque([(-.3,1.),(-.1,3.)],-.4),0.)

    def test_validation_perturbation_cannot_change_training(self):
        original=[trial(0,'test'),trial(1,'retest')]
        altered=[d.copy(deep=True) for d in original]
        # Validation starts at 17 s (80% of 20 s + half of the 2 s gap).
        altered[0].loc[17000:,'gm']*=100
        altered[0].loc[17000:,'torque']+=5
        audits=[];datasets=[]
        for frames in [original,altered]:
            audit={}
            with contextlib.redirect_stdout(io.StringIO()):
                data=build_dataset(frames,validation_gap_s=2.,audit=audit,
                                   passive_entries_override=[(-.2,0.)])
            audits.append(audit);datasets.append(data)
        for j in [0,1,4,5]:np.testing.assert_array_equal(datasets[0][j],datasets[1][j])
        self.assertEqual(datasets[0][6],datasets[1][6])
        self.assertFalse(np.array_equal(datasets[0][2],datasets[1][2]))
        a=audits[0]['traces'];self.assertLess(a['train']['time'].max(),a['validation']['time'].min()-.49-2.)

    def test_clean_validation_is_the_default(self):
        frames=[trial(0,'test'),trial(1,'retest')]
        explicit={}
        with contextlib.redirect_stdout(io.StringIO()):
            default=build_dataset([d.copy(deep=True) for d in frames],
                                  passive_entries_override=[(-.2,0.)])
            clean=build_dataset([d.copy(deep=True) for d in frames],
                                validation_gap_s=2.,
                                passive_entries_override=[(-.2,0.)])
        for got, expected in zip(default[:6], clean[:6]):
            np.testing.assert_array_equal(got, expected)
        with contextlib.redirect_stdout(io.StringIO()):
            legacy=build_dataset([d.copy(deep=True) for d in frames],
                                 validation_gap_s=None,
                                 passive_entries_override=[(-.2,0.)])
        self.assertNotEqual(default[0].shape, legacy[0].shape)

    def test_session_matched_passive_map_is_the_default(self):
        frames = [trial(0, 'test'), trial(1, 'retest')]
        maps = {'test': [(-.2, 1.0)], 'retest': [(-.2, 3.0)]}
        with contextlib.redirect_stdout(io.StringIO()):
            matched = build_dataset([d.copy(deep=True) for d in frames],
                                    validation_gap_s=None,
                                    passive_entries_override_by_session=maps)
            legacy = build_dataset([d.copy(deep=True) for d in frames],
                                   validation_gap_s=None,
                                   passive_session='retest',
                                   passive_entries_override=[(-.2, 3.0)])
        # The session-1 target is measured torque minus 1 Nm; the historical
        # shared retest map subtracts 3 Nm. Retest targets must be identical.
        np.testing.assert_allclose(matched[1], legacy[1] + 2.0, atol=1e-6)
        np.testing.assert_array_equal(matched[5], legacy[5])
        self.assertEqual(matched[7], [(-.2, 3.0)])

    def test_weight_bounds_and_mean_without_retest_labels(self):
        X=np.zeros((200,50,5));X[100:,:,-1]=1
        y=np.r_[np.linspace(0,.1,100),np.linspace(0,100,100)]
        w,_=bounded_weights(X,y,[0,1])
        self.assertGreaterEqual(w.min(),.5);self.assertLessEqual(w.max(),2.)
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
        self.assertGreater(w[0],w[-1])

if __name__=='__main__':unittest.main()
