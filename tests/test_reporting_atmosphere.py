from types import SimpleNamespace

import numpy as np

from aircraft6dof.atmosphere import standard_atmosphere
from aircraft6dof.reporting import _series
from main import REFERENCE_ALTITUDE_M, build_aircraft, environment


def test_reported_atmosphere_matches_fixed_environment_not_instantaneous_altitude():
    env = environment(0.0)
    state = np.zeros((2, 13))
    state[:, 3] = 55.0
    state[:, 9] = 1.0
    state[0, 2] = -REFERENCE_ALTITUDE_M
    state[1, 2] = -(REFERENCE_ALTITUDE_M + 500.0)
    history = SimpleNamespace(
        time_s=np.array([0.0, 1.0]),
        state=state,
        euler_rad=np.zeros((2, 3)),
        control_command_rad=np.array([[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5]]),
        actuator_deflection_rad=np.zeros((2, 3)),
        actuator_deflection_for_state_rad=np.zeros((2, 3)),
    )

    rows = _series(history, None, lambda _t: env, build_aircraft())
    reference = standard_atmosphere(REFERENCE_ALTITUDE_M)

    assert rows[0]["altitude_m"] == REFERENCE_ALTITUDE_M
    assert rows[1]["altitude_m"] == REFERENCE_ALTITUDE_M + 500.0
    for key in ("temperature_K", "pressure_Pa", "density_kg_m3", "speed_of_sound_m_s"):
        assert rows[0][key] == rows[1][key]
    np.testing.assert_allclose(rows[0]["temperature_K"], reference.temperature_K, rtol=1e-9)
    np.testing.assert_allclose(rows[0]["pressure_Pa"], reference.pressure_Pa, rtol=1e-5)
    np.testing.assert_allclose(rows[0]["density_kg_m3"], env.density_kg_m3)
    np.testing.assert_allclose(rows[0]["speed_of_sound_m_s"], env.speed_of_sound_m_s)
