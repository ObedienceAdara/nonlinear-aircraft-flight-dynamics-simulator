import numpy as np

from aircraft6dof.airdata import compute_air_data, relative_velocity_body
from aircraft6dof.atmosphere import standard_atmosphere


def test_airdata_from_pure_forward_velocity():
    atmosphere = standard_atmosphere(0.0)
    result = compute_air_data(np.array([100.0, 0.0, 0.0]), atmosphere)

    np.testing.assert_allclose(result.true_airspeed_m_s, 100.0)
    np.testing.assert_allclose(result.angle_of_attack_rad, 0.0)
    np.testing.assert_allclose(result.sideslip_rad, 0.0)
    np.testing.assert_allclose(result.dynamic_pressure_pa, 0.5 * atmosphere.density_kg_m3 * 100.0**2)


def test_relative_velocity_zero_wind_at_identity_attitude():
    result = relative_velocity_body(
        np.array([50.0, 2.0, -1.0]),
        np.zeros(3),
        0.0, 0.0, 0.0,
    )
    np.testing.assert_allclose(result, [50.0, 2.0, -1.0])


def test_relative_velocity_subtracts_wind_in_common_ned_frame():
    # Identity attitude means body and NED axes coincide under this convention.
    result = relative_velocity_body(
        np.array([100.0, 0.0, 0.0]),
        np.array([20.0, 5.0, 0.0]),
        0.0, 0.0, 0.0,
    )
    np.testing.assert_allclose(result, [80.0, -5.0, 0.0])
