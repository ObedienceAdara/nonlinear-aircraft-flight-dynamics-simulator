import numpy as np
import pytest

from aircraft6dof import AircraftState, ControlInput, Environment
from aircraft6dof.aero import AeroCoefficients, coefficient_buildup
from aircraft6dof.airdata import (
    MIN_AIRSPEED_M_S,
    MIN_RATE_NORMALIZATION_AIRSPEED_M_S,
    air_data,
)
from aircraft6dof.mathutils import dcm_body_to_ned_from_quat, quat_from_euler321
from aircraft6dof.simulation import Simulator


def test_level_flight_in_still_air_has_zero_beta():
    a = air_data([50.0, 0.0, 5.0], quat_from_euler321(0, 0, 0), Environment())
    assert a.airspeed_m_s == pytest.approx(np.hypot(50.0, 5.0))
    assert a.alpha_rad == pytest.approx(np.arctan2(5.0, 50.0))
    assert a.beta_rad == pytest.approx(0.0, abs=1e-15)


def test_wind_and_gust_are_subtracted_in_ned_then_rotated_to_body():
    q = quat_from_euler321(0.2, -0.1, 0.7)
    vb = np.array([45.0, 1.0, 3.0])
    env = Environment(wind_ned_m_s=np.array([5.0, 2.0, 0.0]), gust_ned_m_s=np.array([0.0, 4.0, 2.0]))
    C = dcm_body_to_ned_from_quat(q)
    expected = C.T @ (C @ vb - np.array([5.0, 6.0, 2.0]))
    a = air_data(vb, q, env)
    np.testing.assert_allclose(a.velocity_body_m_s, expected, atol=1e-12)
    assert a.airspeed_m_s == pytest.approx(np.linalg.norm(expected))


def test_wind_given_as_lists_is_accepted():
    env = Environment(wind_ned_m_s=[1.0, 0.0, 0.0], gust_ned_m_s=[0.0, 1.0, 0.0])
    a = air_data([30.0, 0.0, 0.0], quat_from_euler321(0, 0, 0), env)
    np.testing.assert_allclose(a.velocity_body_m_s, [29.0, -1.0, 0.0], atol=1e-12)


def test_alpha_and_beta_are_zero_at_or_below_the_airspeed_floor():
    a = air_data([0.0, 0.0, 0.0], quat_from_euler321(0, 0, 0), Environment())
    assert (a.airspeed_m_s, a.alpha_rad, a.beta_rad) == (0.0, 0.0, 0.0)
    a = air_data([MIN_AIRSPEED_M_S, 0.0, 0.0], quat_from_euler321(0, 0, 0), Environment())
    assert (a.alpha_rad, a.beta_rad) == (0.0, 0.0)


def test_simulator_guard_uses_the_shared_air_data():
    """The guard must see the same alpha/beta/V as the dynamics (one implementation)."""
    assert not hasattr(Simulator, "_air_data")


def test_rate_normalization_floor_only_matters_below_it():
    c = AeroCoefficients()
    args = dict(span=10.0, chord=1.5, controls=(0.0, 0.0, 0.0), c=c)
    floor = MIN_RATE_NORMALIZATION_AIRSPEED_M_S
    low = coefficient_buildup(0.0, 0.0, 0.3, 0.2, 0.1, floor / 10.0, **args)
    at = coefficient_buildup(0.0, 0.0, 0.3, 0.2, 0.1, floor, **args)
    assert low == at
    above = coefficient_buildup(0.0, 0.0, 0.3, 0.2, 0.1, floor * 2.0, **args)
    assert above != at
