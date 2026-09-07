import importlib.util
from pathlib import Path

import numpy as np

import aircraft6dof.equations as equations_module
import aircraft6dof.integrators as integrators_module
import aircraft6dof.mathutils as mathutils
from aircraft6dof.integrators import rk4_step
from aircraft6dof.mathutils import dcm_body_to_ned_from_quat, euler321_from_quat, quat_multiply
from aircraft6dof.reporting import _series
from aircraft6dof.simulation import Simulator
from aircraft6dof.state import AircraftState


_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
_MAIN_SPEC = importlib.util.spec_from_file_location("fdm_demo_main", _MAIN_PATH)
demo = importlib.util.module_from_spec(_MAIN_SPEC)
assert _MAIN_SPEC.loader is not None
_MAIN_SPEC.loader.exec_module(demo)


def test_normalize_quaternion_only_enforces_unit_norm():
    q = np.array([-0.25, 0.35, -0.15, 0.75])
    normalized = mathutils.normalize_quaternion(q)

    np.testing.assert_allclose(np.linalg.norm(normalized), 1.0, atol=1e-12)
    assert normalized[0] < 0.0
    np.testing.assert_allclose(normalized, q / np.linalg.norm(q), atol=1e-12)


def test_quaternion_history_is_continuous_through_q0_zero():
    dt = 0.01
    omega = np.array([0.0, 0.0, np.pi])
    state = AircraftState(
        np.zeros(3),
        np.zeros(3),
        omega.copy(),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    history = [state.quaternion_bn.copy()]

    def derivative(s):
        qdot = 0.5 * quat_multiply(s.quaternion_bn, np.array([0.0, *omega]))
        return AircraftState(np.zeros(3), np.zeros(3), np.zeros(3), qdot)

    for _ in range(100):
        state = rk4_step(state, dt, derivative)
        history.append(state.quaternion_bn.copy())

    quats = np.asarray(history)
    dots = np.sum(quats[:-1] * quats[1:], axis=1)

    assert np.all(dots > 0.99)
    assert np.any(quats[:, 0] > 0.01)
    assert np.any(quats[:, 0] < -0.01)
    assert abs(quats[50, 0]) < 1e-6
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)


def test_physical_outputs_are_invariant_to_quaternion_sign_convention():
    original_mathutils = mathutils.normalize_quaternion
    original_integrators = integrators_module.normalize_quaternion
    original_equations = equations_module.normalize_quaternion

    def canonicalize(q):
        q = original_mathutils(q)
        return -q if q[0] < 0.0 else q

    try:
        mathutils.normalize_quaternion = canonicalize
        integrators_module.normalize_quaternion = canonicalize
        equations_module.normalize_quaternion = canonicalize

        baseline = Simulator(demo.build_aircraft()).run(
            demo.initial_state(),
            demo.controls,
            demo.environment,
            duration_s=40.0,
            dt_s=0.02,
            actuators=demo.build_actuators(),
        )
    finally:
        mathutils.normalize_quaternion = original_mathutils
        integrators_module.normalize_quaternion = original_integrators
        equations_module.normalize_quaternion = original_equations

    current = Simulator(demo.build_aircraft()).run(
        demo.initial_state(),
        demo.controls,
        demo.environment,
        duration_s=40.0,
        dt_s=0.02,
        actuators=demo.build_actuators(),
    )

    np.testing.assert_allclose(current.time_s, baseline.time_s, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(current.state[:, :9], baseline.state[:, :9], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(current.euler_rad, baseline.euler_rad, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(current.control_command_rad, baseline.control_command_rad, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(current.actuator_deflection_rad, baseline.actuator_deflection_rad, rtol=0.0, atol=1e-12)

    for q_new, q_old in zip(current.state[:, 9:13], baseline.state[:, 9:13]):
        C_new = dcm_body_to_ned_from_quat(q_new)
        C_old = dcm_body_to_ned_from_quat(q_old)
        np.testing.assert_allclose(C_new, C_old, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            euler321_from_quat(q_new),
            euler321_from_quat(q_old),
            rtol=0.0,
            atol=1e-12,
        )

    new_rows = _series(current, demo.controls, demo.environment, demo.build_aircraft())
    old_rows = _series(baseline, demo.controls, demo.environment, demo.build_aircraft())
    physical_fields = (
        "speed_m_s",
        "alpha_rad",
        "beta_rad",
        "dynamic_pressure_Pa",
        "mach",
        "CL",
        "CD",
        "CY",
        "Cl",
        "Cm",
        "Cn",
        "Fx_aero_N",
        "Fy_aero_N",
        "Fz_aero_N",
        "Mx_aero_Nm",
        "My_aero_Nm",
        "Mz_aero_Nm",
    )
    for field in physical_fields:
        new_values = np.asarray([row[field] for row in new_rows])
        old_values = np.asarray([row[field] for row in old_rows])
        np.testing.assert_allclose(new_values, old_values, rtol=0.0, atol=1e-12)

    quaternion_alignment = np.sum(current.state[:, 9:13] * baseline.state[:, 9:13], axis=1)
    assert np.all(np.isclose(np.abs(quaternion_alignment), 1.0, atol=1e-12))
