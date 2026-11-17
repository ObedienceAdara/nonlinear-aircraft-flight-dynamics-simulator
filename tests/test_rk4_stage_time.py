import numpy as np
import pytest

from aircraft6dof.integrators import rk4_step
from aircraft6dof.state import AircraftState


def zero_state():
    return AircraftState(
        position_ned_m=np.zeros(3),
        velocity_body_m_s=np.zeros(3),
        omega_body_rad_s=np.zeros(3),
        quaternion_bn=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_rk4_time_dependent_rhs_is_evaluated_at_all_stage_times():
    calls = []

    def derivative(state, time_s):
        calls.append(time_s)
        return AircraftState(
            position_ned_m=np.array([time_s, 0.0, 0.0]),
            velocity_body_m_s=np.zeros(3),
            omega_body_rad_s=np.zeros(3),
            quaternion_bn=np.zeros(4),
        )

    result = rk4_step(
        zero_state(),
        0.1,
        derivative,
        time_s=0.1,
        stage_time_s=0.0,
    )

    np.testing.assert_allclose(calls, [0.0, 0.05, 0.05, 0.1], atol=1e-15)
    # Integrate x_dot = t over [0, 0.1] with RK4; this polynomial is exact.
    assert result.position_ned_m[0] == pytest.approx(0.005, abs=1e-14)


def test_rk4_keeps_legacy_state_only_rhs_signature():

    def derivative(state):
        return AircraftState(
            position_ned_m=np.array([1.0, 0.0, 0.0]),
            velocity_body_m_s=np.zeros(3),
            omega_body_rad_s=np.zeros(3),
            quaternion_bn=np.zeros(4),
        )

    result = rk4_step(zero_state(), 0.1, derivative)
    assert result.position_ned_m[0] == pytest.approx(0.1, abs=1e-14)
