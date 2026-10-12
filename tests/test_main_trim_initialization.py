import numpy as np

from main import (
    REFERENCE_ALTITUDE_M,
    build_actuators,
    build_aircraft,
    controls,
    environment,
    initial_state,
    trim_initial_condition,
)


def test_default_initial_state_is_a_trim_solution():
    aircraft = build_aircraft()
    env = environment(0.0)
    result = trim_initial_condition(aircraft, env)

    state = initial_state(aircraft, env)
    np.testing.assert_allclose(state.vector(), result.state.vector(), atol=1e-10)
    assert result.condition.altitude_m == REFERENCE_ALTITUDE_M
    assert result.condition.airspeed_m_s == 55.0
    assert result.residual_norm < 1e-9


def test_demo_commands_and_actuators_start_at_trim_baseline():
    result = trim_initial_condition(build_aircraft(), environment(0.0))
    at_start = controls(0.0, result.controls)

    np.testing.assert_allclose(
        [at_start.aileron, at_start.elevator, at_start.rudder, at_start.throttle],
        [
            result.controls.aileron,
            result.controls.elevator,
            result.controls.rudder,
            result.controls.throttle,
        ],
        atol=1e-12,
    )
    actuators = build_actuators(result.controls)
    np.testing.assert_allclose(
        [actuators.aileron.position_rad, actuators.elevator.position_rad, actuators.rudder.position_rad],
        [result.controls.aileron, result.controls.elevator, result.controls.rudder],
        atol=1e-12,
    )

    shifted = controls(10.0, result.controls)
    assert shifted.elevator < result.controls.elevator
    throttle_step = controls(20.0, result.controls)
    assert throttle_step.throttle >= result.controls.throttle
