import numpy as np

from aircraft6dof.atmosphere import standard_atmosphere


def test_sea_level_standard_atmosphere():
    a = standard_atmosphere(0.0)
    np.testing.assert_allclose(a.temperature_K, 288.15, atol=1e-9)
    np.testing.assert_allclose(a.pressure_Pa, 101325.0, rtol=1e-9)
    np.testing.assert_allclose(a.density_kg_m3, 1.225, rtol=1e-3)


def test_density_decreases_with_altitude():
    sea = standard_atmosphere(0.0)
    high = standard_atmosphere(5000.0)
    assert high.density_kg_m3 < sea.density_kg_m3


def test_invalid_altitude_rejected():
    import pytest
    with pytest.raises(ValueError):
        standard_atmosphere(-1.0)
