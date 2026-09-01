import numpy as np
from aircraft6dof.integrators import rk4_step
from aircraft6dof.state import AircraftState

def test_rk4_constant():
    x=AircraftState(np.array([1.,2.,3.]),np.zeros(3),np.zeros(3),np.array([1.,0.,0.,0.]))
    d=AircraftState(np.ones(3),np.zeros(3),np.zeros(3),np.zeros(4))
    y=rk4_step(x,2.,lambda _:d)
    np.testing.assert_allclose(y.position_ned_m,[3.,4.,5.])

def test_rk4_quaternion_normalized():
    x=AircraftState(np.zeros(3),np.array([50.,0.,0.]),np.array([0.,0.,.1]),np.array([1.,0.,0.,0.]))
    z=AircraftState(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(4))
    y=rk4_step(x,.01,lambda s: AircraftState(np.zeros(3),np.zeros(3),s.omega_body_rad_s,z.quaternion_bn))
    np.testing.assert_allclose(np.linalg.norm(y.quaternion_bn),1.,atol=1e-12)
