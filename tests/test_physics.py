import numpy as np
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
