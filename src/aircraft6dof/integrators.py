import numpy as np
from .mathutils import normalize_quaternion
from .state import AircraftState
def add(x,dx,a):
    return AircraftState(x.position_ned_m+a*dx.position_ned_m,x.velocity_body_m_s+a*dx.velocity_body_m_s,x.omega_body_rad_s+a*dx.omega_body_rad_s,x.quaternion_bn+a*dx.quaternion_bn)
def rk4_step(x,dt,f):
    k1=f(x); k2=f(add(x,k1,dt/2)); k3=f(add(x,k2,dt/2)); k4=f(add(x,k3,dt))
    return AircraftState(x.position_ned_m+dt*(k1.position_ned_m+2*k2.position_ned_m+2*k3.position_ned_m+k4.position_ned_m)/6,
      x.velocity_body_m_s+dt*(k1.velocity_body_m_s+2*k2.velocity_body_m_s+2*k3.velocity_body_m_s+k4.velocity_body_m_s)/6,
      x.omega_body_rad_s+dt*(k1.omega_body_rad_s+2*k2.omega_body_rad_s+2*k3.omega_body_rad_s+k4.omega_body_rad_s)/6,
      normalize_quaternion(x.quaternion_bn+dt*(k1.quaternion_bn+2*k2.quaternion_bn+2*k3.quaternion_bn+k4.quaternion_bn)/6))
