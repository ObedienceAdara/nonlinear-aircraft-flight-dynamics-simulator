import numpy as np
import pytest

from aircraft6dof.simulation import SimulationDivergenceError, SimulationGuardConfig, Simulator
from aircraft6dof.state import AircraftState, ControlInput, Environment


class NaNInjectingAircraft:
    def __init__(self, nan_on_derivative_call=5):
        self.calls = 0
        self.nan_on_derivative_call = nan_on_derivative_call

    def derivative(self, state, controls, environment):
        self.calls += 1
        velocity = np.zeros(3)
        if self.calls >= self.nan_on_derivative_call:
            velocity[1] = np.nan
        return AircraftState(
            np.zeros(3),
            velocity,
            np.zeros(3),
            np.zeros(4),
        )


def test_nan_state_is_caught_on_second_simulation_step():
    aircraft = NaNInjectingAircraft(nan_on_derivative_call=5)
    initial = AircraftState(
        np.zeros(3),
        np.array([55.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    with pytest.raises(SimulationDivergenceError) as exc_info:
        Simulator(aircraft).run(
            initial,
            ControlInput(),
            Environment(gravity_ned_m_s2=np.zeros(3)),
            duration_s=0.06,
            dt_s=0.02,
            guard=SimulationGuardConfig(history_size=2),
        )

    exc = exc_info.value
    assert exc.step_index == 2
    assert np.isclose(exc.time_s, 0.04)
    assert exc.field == "rk4_output_state"
    assert "velocity_body_m_s" in exc.reason
    assert "t=0.040000 s" in str(exc)
    assert len(exc.last_valid_states) == 2
    assert np.isclose(exc.last_valid_states[-1][0], 0.02)
