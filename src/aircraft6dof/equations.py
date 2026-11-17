from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .state import AircraftState,ControlInput,Environment,VehicleGeometry
from .aero import AeroCoefficients,aerodynamic_loads
from .airdata import air_data
from .propulsion import Propulsion
from .mathutils import normalize_quaternion,quat_multiply,dcm_body_to_ned_from_quat

@dataclass(frozen=True)
class AircraftParameters:
    geometry:VehicleGeometry; aero:AeroCoefficients; propulsion:Propulsion

def rigid_body_derivative(s:AircraftState,force_body:np.ndarray,moment_body:np.ndarray,gravity_ned:np.ndarray,geometry:VehicleGeometry)->AircraftState:
    """Rigid-body equations of motion for a given applied (non-gravity) load.

    force_body / moment_body are the aerodynamic + propulsive force [N] and moment
    [N m] about the CG, in body axes. Gravity is added here from gravity_ned.
    Returns the state derivative: NED velocity, body acceleration (dv/dt with the
    omega x v term), body angular acceleration and the quaternion rate.
    """
    q=normalize_quaternion(s.quaternion_bn); vb=s.velocity_body_m_s; om=s.omega_body_rad_s; I=geometry.inertia_kg_m2; m=geometry.mass_kg
    C=dcm_body_to_ned_from_quat(q)
    total_f=force_body+C.T@gravity_ned*m
    vdot=total_f/m-np.cross(om,vb); odot=np.linalg.solve(I,moment_body-np.cross(om,I@om))
    qdot=.5*quat_multiply(q,np.array([0.,*om]))
    return AircraftState(C@vb,vdot,odot,qdot)

def state_derivative(s:AircraftState,u:ControlInput,e:Environment,p:AircraftParameters)->AircraftState:
    om=s.omega_body_rad_s
    a=air_data(s.velocity_body_m_s,s.quaternion_bn,e)
    fb,mb,_=aerodynamic_loads(e.density_kg_m3,a.velocity_body_m_s,a.alpha_rad,a.beta_rad,*om,p.geometry,np.array([u.aileron,u.elevator,u.rudder]),p.aero)
    fp,mp=p.propulsion.force_and_moment(u.throttle,a.airspeed_m_s)
    return rigid_body_derivative(s,fb+fp,mb+mp,e.gravity_ned_m_s2,p.geometry)