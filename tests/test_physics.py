import numpy as np
from aircraft6dof.aero import AeroCoefficients
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
    e=Environment(wind_ned_m_s2=np.zeros(3),gust_ned_m_s=np.zeros(3),gravity_ned_m_s2=np.zeros(3))
    d=state_derivative(s,ControlInput(),e,params())
    assert np.isfinite(d.vector()).all()
