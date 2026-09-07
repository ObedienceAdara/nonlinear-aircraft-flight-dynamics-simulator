import numpy as np

from aircraft6dof.actuators import ActuatorChannel, ActuatorSet
from aircraft6dof.simulation import Simulator
from aircraft6dof.state import AircraftState, ControlInput, Environment


class RecordingAircraft:
    def __init__(self):
        self.calls = []

    def derivative(self, state, controls, environment):
        self.calls.append(controls)
        return AircraftState(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(4),
        )


def make_actuators():
    return ActuatorSet(
        aileron=ActuatorChannel(np.array(0.15), np.deg2rad(80.0), np.deg2rad(25.0)),
        elevator=ActuatorChannel(np.array(0.20), np.deg2rad(50.0), np.deg2rad(25.0)),
        rudder=ActuatorChannel(np.array(0.20), np.deg2rad(40.0), np.deg2rad(30.0)),
    )


def test_simulator_passes_actual_actuator_positions_to_dynamics():
    aircraft = RecordingAircraft()
    actuators = make_actuators()
    command = ControlInput(
        aileron=np.deg2rad(3.0),
        elevator=np.deg2rad(-1.5),
        rudder=np.deg2rad(1.0),
        throttle=0.72,
    )
    initial = AircraftState(
        np.zeros(3),
        np.array([55.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    history = Simulator(aircraft).run(
        initial,
        command,
        Environment(gravity_ned_m_s2=np.zeros(3)),
        duration_s=0.02,
        dt_s=0.02,
        actuators=actuators,
    )

    first = aircraft.calls[0]
    assert np.isclose(first.throttle, command.throttle)
    assert abs(first.aileron) < abs(command.aileron)
    assert abs(first.elevator) < abs(command.elevator)
    assert abs(first.rudder) < abs(command.rudder)
    np.testing.assert_allclose(
        history.actuator_deflection_rad[0],
        [first.aileron, first.elevator, first.rudder],
    )
    np.testing.assert_allclose(
        history.control_command_rad[0],
        [command.aileron, command.elevator, command.rudder, command.throttle],
    )


def test_actuator_response_is_multi_step_not_one_timestep():
    actuator = ActuatorChannel(
        time_constant_s=0.20,
        rate_limit_rad_s=np.deg2rad(50.0),
        position_limit_rad=np.deg2rad(25.0),
    )
    command = np.deg2rad(-1.5)
    values = [actuator.step(command, 0.02) for _ in range(20)]

    assert abs(values[0]) < abs(command)
    assert abs(values[4]) > abs(values[0])
    assert abs(values[-1]) > abs(values[4])
    assert abs(values[-1]) < abs(command)
