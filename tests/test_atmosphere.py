import numpy as np
from aircraft6dof.atmosphere import standard_atmosphere

def test_standard_atmosphere_sea_level():
    a=standard_atmosphere(0.)
    np.testing.assert_allclose(a.temperature_K,288.15,atol=1e-9)
    np.testing.assert_allclose(a.pressure_Pa,101325.,rtol=1e-9)
    np.testing.assert_allclose(a.density_kg_m3,1.225,rtol=1e-3)

def test_density_decreases():
    assert standard_atmosphere(5000.).density_kg_m3 < standard_atmosphere(0.).density_kg_m3
