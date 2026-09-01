from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .state import AircraftState,ControlInput,Environment,VehicleGeometry
from .aero import AeroCoefficients,aerodynamic_loads
from .propulsion import Propulsion
from .mathutils import normalize_quaternion,quat_multiply,dcm_body_to_ned_from_quat

@dataclass(frozen=True)
class AircraftParameters:
    geometry:VehicleGeometry; aero:AeroCoefficients; propulsion:Propulsion

def state_derivative(s:AircraftState,u:ControlInput,e:Environment,p:AircraftParameters)->AircraftState:
    q=normalize_quaternion(s.quaternion_bn); vb=s.velocity_body_m_s; om=s.omega_body_rad_s; I=p.geometry.inertia_kg_m2; m=p.geometry.mass_kg
    C=dcm_body_to_ned_from_quat(q); v_n=C@vb; vrel_n=v_n-(e.wind_ned_m_s+e.gust_ned_m_s); vrel_b=C.T@vrel_n
    ur,vr,wr=vrel_b; V=max(np.linalg.norm(vrel_b),1e-9); alpha=np.arctan2(wr,ur); beta=np.arcsin(np.clip(vr/V,-1,1))
    fb,mb,_=aerodynamic_loads(e.density_kg_m3,vrel_b,alpha,beta,*om,p.geometry,np.array([u.aileron,u.elevator,u.rudder]),p.aero)
    fp,mp=p.propulsion.force_and_moment(u.throttle,V)
    fg=C.T@e.gravity_ned_m_s2*m
    total_f=fb+fp+fg; total_m=mb+mp
    vdot=total_f/m-np.cross(om,vb); odot=np.linalg.solve(I,total_m-np.cross(om,I@om))
    qdot=.5*quat_multiply(q,np.array([0.,*om]))
    return AircraftState(v_n,vdot,odot,qdot)
