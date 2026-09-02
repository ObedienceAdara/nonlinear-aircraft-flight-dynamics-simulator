import numpy as np
from aircraft6dof.wind import OneMinusCosineGust, DrydenTurbulence


def test_one_minus_cosine_vector_api_start_end_zero():
    g = OneMinusCosineGust(
        amplitude_m_s=np.array([0.0, 10.0, 5.0]),
        start_s=2.0,
        duration_s=4.0,
    )
    np.testing.assert_allclose(g.value(2.0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(g.value(6.0), [0.0, 0.0, 0.0])


def test_one_minus_cosine_vector_api_peak():
    g = OneMinusCosineGust(
        amplitude_m_s=np.array([0.0, 10.0, 5.0]),
        start_s=2.0,
        duration_s=4.0,
    )
    np.testing.assert_allclose(g.value(4.0), [0.0, 10.0, 5.0])


def test_one_minus_cosine_legacy_api_remains_supported():
    g = OneMinusCosineGust(10.0, np.array([0.0, 1.0, 0.0]), 2.0, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(g.value(2.0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(g.value(3.0), [0.0, 10.0, 0.0])
    np.testing.assert_allclose(g.value(5.0), [0.0, 0.0, 0.0])


def test_dryden_process_is_reproducible():
    a = DrydenTurbulence(np.array([2.0, 3.0, 4.0]), np.array([30.0, 40.0, 50.0]), seed=7)
    b = DrydenTurbulence(np.array([2.0, 3.0, 4.0]), np.array([30.0, 40.0, 50.0]), seed=7)
    for _ in range(20):
        np.testing.assert_allclose(a.step(60.0, 0.02), b.step(60.0, 0.02))
