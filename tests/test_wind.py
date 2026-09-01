import numpy as np
from aircraft6dof.wind import OneMinusCosineGust,DrydenTurbulence

def test_one_minus_cosine_start_end_zero():
    g=OneMinusCosineGust(10.,np.array([1.,0.,0.]),2.,1.,1.,1.)
    np.testing.assert_allclose(g.value(2.),[0.,0.,0.])
    np.testing.assert_allclose(g.value(5.),[0.,0.,0.])

def test_one_minus_cosine_peak():
    g=OneMinusCosineGust(10.,np.array([0.,1.,0.]),2.,1.,1.,1.)
    np.testing.assert_allclose(g.value(3.5),[0.,10.,0.])

def test_dryden_process_is_reproducible():
    a=DrydenTurbulence(np.array([2.,3.,4.]),np.array([30.,40.,50.]),seed=7)
    b=DrydenTurbulence(np.array([2.,3.,4.]),np.array([30.,40.,50.]),seed=7)
    for _ in range(20):
        np.testing.assert_allclose(a.step(60.,.02),b.step(60.,.02))
