import numpy as np
import pytest

from aircraft6dof.actuators import ActuatorChannel, ActuatorSet
from aircraft6dof.simulation import SimulationGuardConfig, Simulator
from aircraft6dof.state import AircraftState, ControlInput, Environment


class RecordingAircraft:
    def __init__(self):
        self.calls = []

    def derivative(self, state, controls, environment):
        self.calls.append(
            (
                float(controls.aileron),
                float(controls.throttle),
                float(environment.gust_ned_m_s[0]),
            )
        )
        return AircraftState(
            position_ned_m=np.array(
                [controls.aileron + environment.gust_ned_m_s[0], 0.0, 0.0]
            ),
            velocity_body_m_s=np.zeros(3),
            omega_body_rad_s=np.zeros(3),
            quaternion_bn=np.zeros(4),
        )


def initial_state():
    return AircraftState(
        position_ned_m=np.zeros(3),
        velocity_body_m_s=np.array([55.0, 0.0, 0.0]),
        omega_body_rad_s=np.zeros(3),
        quaternion_bn=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_simulator_samples_direct_controls_and_environment_at_rk4_stages():
    aircraft = RecordingAircraft()

    def controls(time_s):
        return ControlInput(aileron=time_s)

    def environment(time_s):
        return Environment(
            gust_ned_m_s=np.array([time_s, 0.0, 0.0]),
            gravity_ned_m_s2=np.zeros(3),
        )

    history = Simulator(aircraft).run(
        initial_state(),
        controls,
        environment,
        duration_s=0.1,
        dt_s=0.1,
        guard=SimulationGuardConfig(enabled=False),
    )

    expected_stage_times = np.array([0.0, 0.05, 0.05, 0.1])
    np.testing.assert_allclose(
        [call[0] for call in aircraft.calls],
        expected_stage_times,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        [call[2] for call in aircraft.calls],
        expected_stage_times,
        atol=1e-15,
    )
    # x_dot = aileron(t) + gust(t) = 2t; RK4 integrates this exactly here.
    assert history.state[-1, 0] == pytest.approx(0.01, abs=1e-14)


def test_actuated_surfaces_are_held_but_throttle_is_sampled_at_rk4_stages():
    aircraft = RecordingAircraft()
    actuators = ActuatorSet(
        aileron=ActuatorChannel(0.2, 10.0, 1.0),
        elevator=ActuatorChannel(0.2, 10.0, 1.0),
        rudder=ActuatorChannel(0.2, 10.0, 1.0),
    )

    def controls(time_s):
        return ControlInput(aileron=0.1, elevator=-0.05, rudder=0.02, throttle=time_s)

    history = Simulator(aircraft).run(
        initial_state(),
        controls,
        Environment(gravity_ned_m_s2=np.zeros(3)),
        duration_s=0.1,
        dt_s=0.1,
        actuators=actuators,
        guard=SimulationGuardConfig(enabled=False),
    )

    expected_stage_times = np.array([0.0, 0.05, 0.05, 0.1])
    np.testing.assert_allclose(
        [call[1] for call in aircraft.calls],
        expected_stage_times,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        [call[0] for call in aircraft.calls],
        np.full(4, aircraft.calls[0][0]),
        atol=1e-15,
    )
    # x_dot = throttle(t) = t; RK4 integrates this exactly over the interval.
    assert history.state[-1, 0] == pytest.approx(0.005, abs=1e-14)


def test_grid_aligned_step_command_does_not_leak_into_previous_interval():
    aircraft = RecordingAircraft()

    def controls(time_s):
        # The command turns on exactly at the boundary between the two steps.
        return ControlInput(aileron=1.0 if time_s >= 0.1 else 0.0)

    history = Simulator(aircraft).run(
        initial_state(),
        controls,
        Environment(gravity_ned_m_s2=np.zeros(3)),
        duration_s=0.2,
        dt_s=0.1,
        guard=SimulationGuardConfig(enabled=False),
    )

    assert history.state[1, 0] == pytest.approx(0.0, abs=1e-15)
    assert history.state[2, 0] == pytest.approx(0.1, abs=1e-14)
