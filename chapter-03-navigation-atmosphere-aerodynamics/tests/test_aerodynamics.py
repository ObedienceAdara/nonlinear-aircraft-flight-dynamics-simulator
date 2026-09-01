import numpy as np

from aircraft6dof.aerodynamics import AerodynamicCoefficients, aerodynamic_loads


def test_zero_coefficients_give_zero_loads():
    c = AerodynamicCoefficients(0, 0, 0, 0, 0, 0)
    loads = aerodynamic_loads(500.0, 10.0, 8.0, 1.0, 0.1, 0.05, c)
    np.testing.assert_allclose(loads.force_body_N, np.zeros(3))
    np.testing.assert_allclose(loads.moment_body_Nm, np.zeros(3))


def test_dynamic_pressure_scaling():
    c = AerodynamicCoefficients(0.02, 0.5, 0, 0.01, -0.02, 0)
    a = aerodynamic_loads(100.0, 10.0, 8.0, 1.0, 0.0, 0.0, c)
    b = aerodynamic_loads(200.0, 10.0, 8.0, 1.0, 0.0, 0.0, c)

    np.testing.assert_allclose(b.force_body_N, 2.0 * a.force_body_N)
    np.testing.assert_allclose(b.moment_body_Nm, 2.0 * a.moment_body_Nm)
