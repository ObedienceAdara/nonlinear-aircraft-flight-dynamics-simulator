import numpy as np
import pytest
from aircraft6dof.aero import AeroCoefficients, aerodynamic_loads
from aircraft6dof.equations import AircraftParameters,state_derivative
from aircraft6dof.mathutils import quat_from_euler321,dcm_body_to_ned_from_quat
from aircraft6dof.state import AircraftState,ControlInput,Environment,VehicleGeometry
from aircraft6dof.propulsion import Propulsion

def params():
    return AircraftParameters(
        VehicleGeometry(1200.,np.diag([1800.,2100.,3300.]),16.2,11.,1.5),
        AeroCoefficients(),
        Propulsion(max_thrust_N=0.),
    )

def test_zero_rate_quaternion_orientation():
    q=quat_from_euler321(.1,-.2,.3)
    C=dcm_body_to_ned_from_quat(q)
    np.testing.assert_allclose(C.T@C,np.eye(3),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(C),1.,atol=1e-12)

def test_gravity_only_at_level_attitude():
    s=AircraftState(np.zeros(3),np.zeros(3),np.zeros(3),quat_from_euler321(0,0,0))
    e=Environment(gravity_ned_m_s2=np.array([0.,0.,9.80665]))
    d=state_derivative(s,ControlInput(),e,params())
    np.testing.assert_allclose(d.velocity_body_m_s,[0.,0.,9.80665],atol=1e-10)

def test_zero_relative_speed_has_zero_aero_loads():
    s=AircraftState(np.zeros(3),np.zeros(3),np.zeros(3),quat_from_euler321(0,0,0))
    e=Environment(wind_ned_m_s=np.zeros(3),gust_ned_m_s=np.zeros(3),gravity_ned_m_s2=np.zeros(3))
    d=state_derivative(s,ControlInput(),e,params())
    assert np.isfinite(d.vector()).all()



def test_aerodynamic_reference_moment_is_translated_to_cg():
    """The configured CG-to-reference lever arm contributes r x F to moment."""
    base = params().geometry
    offset = np.array([0.15, -0.04, 0.02])
    with_offset = VehicleGeometry(
        mass_kg=base.mass_kg,
        inertia_kg_m2=base.inertia_kg_m2,
        wing_area_m2=base.wing_area_m2,
        wing_span_m=base.wing_span_m,
        mean_chord_m=base.mean_chord_m,
        cg_to_ref_m=offset,
    )
    rho = 1.1
    vrel = np.array([52.0, 2.0, 3.5])
    alpha = np.arctan2(vrel[2], vrel[0])
    beta = np.arcsin(vrel[1] / np.linalg.norm(vrel))
    controls = np.array([np.deg2rad(2.0), np.deg2rad(-1.0), np.deg2rad(0.5)])
    coefficients = AeroCoefficients()

    force_ref, moment_ref, _ = aerodynamic_loads(
        rho, vrel, alpha, beta, 0.1, -0.03, 0.02, base, controls, coefficients
    )
    force_cg, moment_cg, _ = aerodynamic_loads(
        rho, vrel, alpha, beta, 0.1, -0.03, 0.02, with_offset, controls, coefficients
    )
    np.testing.assert_allclose(force_cg, force_ref, atol=1e-12)
    np.testing.assert_allclose(
        moment_cg,
        moment_ref + np.cross(offset, force_ref),
        rtol=1e-12,
        atol=1e-10,
    )


def test_optional_aero_terms_default_to_zero_and_act_when_set():
    from aircraft6dof.aero import coefficient_buildup

    args = (0.05, 0.02, 0.1, 0.05, -0.04, 50.0, 10.0, 1.5, (0.01, -0.02, 0.03))
    base = coefficient_buildup(*args, AeroCoefficients())
    same = coefficient_buildup(*args, AeroCoefficients(CD_alpha=0.0, CY_p=0.0, CY_r=0.0))
    assert base == same
    extra = coefficient_buildup(*args, AeroCoefficients(CD_alpha=0.1, CY_p=0.2, CY_r=0.3))
    ph, rh = 0.1 * 10.0 / 100.0, -0.04 * 10.0 / 100.0
    assert extra["CD"] - base["CD"] == pytest.approx(0.1 * 0.05)
    assert extra["CY"] - base["CY"] == pytest.approx(0.2 * ph + 0.3 * rh)


def test_rigid_body_derivative_independently_applies_force_moment_and_gravity():
    from aircraft6dof.equations import rigid_body_derivative
    from aircraft6dof.mathutils import normalize_quaternion, quat_multiply

    geometry = params().geometry
    state = AircraftState(
        position_ned_m=np.array([12.0, -4.0, -1000.0]),
        velocity_body_m_s=np.array([48.0, 2.0, 5.0]),
        omega_body_rad_s=np.array([0.12, -0.04, 0.08]),
        quaternion_bn=quat_from_euler321(0.1, 0.06, -0.2),
    )
    force = np.array([150.0, -40.0, -900.0])
    moment = np.array([25.0, -60.0, 12.0])
    gravity_ned = np.array([0.0, 0.0, 9.80665])
    derivative = rigid_body_derivative(state, force, moment, gravity_ned, geometry)

    q = normalize_quaternion(state.quaternion_bn)
    C_bn = dcm_body_to_ned_from_quat(q)
    expected_vdot = (force + C_bn.T @ gravity_ned * geometry.mass_kg) / geometry.mass_kg - np.cross(
        state.omega_body_rad_s, state.velocity_body_m_s
    )
    inertia = geometry.inertia_kg_m2
    expected_omega_dot = np.linalg.solve(
        inertia, moment - np.cross(state.omega_body_rad_s, inertia @ state.omega_body_rad_s)
    )
    expected_qdot = 0.5 * quat_multiply(q, np.array([0.0, *state.omega_body_rad_s]))

    np.testing.assert_allclose(derivative.position_ned_m, C_bn @ state.velocity_body_m_s, atol=1e-12)
    np.testing.assert_allclose(derivative.velocity_body_m_s, expected_vdot, atol=1e-12)
    np.testing.assert_allclose(derivative.omega_body_rad_s, expected_omega_dot, atol=1e-12)
    np.testing.assert_allclose(derivative.quaternion_bn, expected_qdot, atol=1e-12)


def test_absolute_sideslip_drag_coefficient_is_optional_and_applied():
    from aircraft6dof.aero import coefficient_buildup

    args = (0.04, 0.025, 0.1, 0.02, -0.03, 50.0, 10.0, 1.5,
            (0.01, -0.02, 0.03))
    baseline = coefficient_buildup(*args, AeroCoefficients())
    extra_coefficient = 0.17
    with_beta_drag = coefficient_buildup(
        *args, AeroCoefficients(CD_abs_beta=extra_coefficient)
    )
    assert baseline["CD"] == pytest.approx(
        AeroCoefficients().CD0
        + AeroCoefficients().CD_alpha * args[0]
        + AeroCoefficients().CD_alpha2 * args[0] ** 2
        + AeroCoefficients().CD_de2 * args[1] ** 2
    )
    assert with_beta_drag["CD"] - baseline["CD"] == pytest.approx(
        extra_coefficient * abs(args[1])
    )
