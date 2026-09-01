from __future__ import annotations
from dataclasses import dataclass
import math
from .constants import G0,R_AIR,GAMMA_AIR

@dataclass(frozen=True)
class Atmosphere:
    altitude_m: float; temperature_K: float; pressure_Pa: float; density_kg_m3: float; speed_of_sound_m_s: float

def standard_atmosphere(h):
    h=float(h)
    if not math.isfinite(h) or h<0 or h>20000: raise ValueError("altitude must be 0..20000 m")
    T0,P0,L,h11=288.15,101325.,-0.0065,11000.
    if h<=h11:
        T=T0+L*h; P=P0*(T/T0)**(-G0/(L*R_AIR))
    else:
        T=T0+L*h11; P11=P0*(T/T0)**(-G0/(L*R_AIR)); P=P11*math.exp(-G0*(h-h11)/(R_AIR*T))
    rho=P/(R_AIR*T); a=math.sqrt(GAMMA_AIR*R_AIR*T)
    return Atmosphere(h,T,P,rho,a)
