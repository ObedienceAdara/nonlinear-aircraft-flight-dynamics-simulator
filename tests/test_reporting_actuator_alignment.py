from types import SimpleNamespace

import numpy as np

from aircraft6dof import reporting
from aircraft6dof.actuators import ActuatorChannel, ActuatorSet
from aircraft6dof.simulation import SimulationGuardConfig, Simulator
from aircraft6dof.state import AircraftState, ControlInput, Environment


class RecordingAircraft:
    def __init__(self):
        self.calls = []
        self.parameters = SimpleNamespace(
            geometry=SimpleNamespace(
                mass_kg=1200.0,
                wing_area_m2=16.2,
                wing_span_m=10.9,
                mean_chord_m=1.49,
            ),
            aero=SimpleNamespace(),
            propulsion=SimpleNamespace(
                force_and_moment=lambda throttle, speed: (
                    np.zeros(3),
                    np.zeros(3),
                )
            ),
        )

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
        aileron=ActuatorChannel(0.15, np.deg2rad(80.0), np.deg2rad(25.0)),
        elevator=ActuatorChannel(0.20, np.deg2rad(50.0), np.deg2rad(25.0)),
        rudder=ActuatorChannel(0.20, np.deg2rad(40.0), np.deg2rad(30.0)),
    )


def test_reported_moment_uses_actuator_that_produced_state(monkeypatch):
    aircraft = RecordingAircraft()
    initial = AircraftState(
        np.zeros(3),
        np.array([55.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    def controls(t):
        return ControlInput(
            aileron=np.deg2rad(3.0) if t >= 0.02 else 0.0,
            elevator=0.0,
            rudder=0.0,
            throttle=0.72,
        )

    environment = Environment(gravity_ned_m_s2=np.zeros(3))
    history = Simulator(aircraft).run(
        initial,
        controls,
        environment,
        duration_s=0.04,
        dt_s=0.02,
        actuators=make_actuators(),
        guard=SimulationGuardConfig(enabled=False),
    )

    # RK4 evaluates the same interval input four times. X[1] was produced by
    # the first RK4 step, so its state-aligned actuator is call 0 (and calls
    # 1–3, the remaining RK4 substages, carry the same control). At t=0.02
    # the next interval receives the step command and advances the actuator.
    first_step_control = aircraft.calls[0]
    second_step_control = aircraft.calls[4]
    np.testing.assert_allclose(
        history.actuator_deflection_for_state_rad[1],
        [first_step_control.aileron, first_step_control.elevator, first_step_control.rudder],
    )
    assert history.actuator_deflection_rad[1, 0] > history.actuator_deflection_for_state_rad[1, 0]
    np.testing.assert_allclose(
        history.actuator_deflection_for_state_rad[2],
        [second_step_control.aileron, second_step_control.elevator, second_step_control.rudder],
    )

    def fake_aerodynamic_loads(rho, vrel_body, alpha, beta, p, q, r, geometry, controls_vec, coeffs):
        # Diagnostic load: make the reported roll moment uniquely identify the
        # actuator deflection used by the reporting layer.
        da = float(controls_vec[0])
        return np.zeros(3), np.array([1000.0 * da, 0.0, 0.0]), {
            "CL": 0.0,
            "CD": 0.0,
            "CY": 0.0,
            "Cl": da,
            "Cm": 0.0,
            "Cn": 0.0,
        }

    monkeypatch.setattr(reporting, "aerodynamic_loads", fake_aerodynamic_loads)
    rows = reporting._series(history, controls, environment, aircraft)

    expected_step_1_mx = 1000.0 * first_step_control.aileron
    expected_step_2_mx = 1000.0 * second_step_control.aileron
    assert np.isclose(rows[1]["Mx_aero_Nm"], expected_step_1_mx)
    assert np.isclose(rows[2]["Mx_aero_Nm"], expected_step_2_mx)
    assert np.isclose(rows[1]["aileron_for_state_rad"], first_step_control.aileron)
    assert np.isclose(rows[2]["aileron_for_state_rad"], second_step_control.aileron)
