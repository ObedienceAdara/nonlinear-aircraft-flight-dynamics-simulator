"""WGS-84 geodetic utilities for extending the local-NED model."""

from __future__ import annotations
import numpy as np
from .constants import WGS84_A, WGS84_B


def radii_of_curvature(latitude_rad:float)->tuple[float,float]:
    s=np.sin(latitude_rad); e2=1-(WGS84_B/WGS84_A)**2
    N=WGS84_A/np.sqrt(1-e2*s*s)
    M=WGS84_A*(1-e2)/(1-e2*s*s)**1.5
    return M,N


def ned_displacement_to_lla_delta(north_m:float,east_m:float,down_m:float,latitude_rad:float,altitude_m:float)->np.ndarray:
    """First-order WGS-84 mapping from local NED displacement to d(lat,lon,h)."""
    M,N=radii_of_curvature(latitude_rad)
    dlat=north_m/(M+altitude_m)
    dlon=east_m/((N+altitude_m)*np.cos(latitude_rad))
    dh=-down_m
    return np.array([dlat,dlon,dh])


def lla_to_ned_delta(dlat_rad:float,dlon_rad:float,dh_m:float,latitude_rad:float,altitude_m:float)->np.ndarray:
    M,N=radii_of_curvature(latitude_rad)
    north=dlat_rad*(M+altitude_m)
    east=dlon_rad*(N+altitude_m)*np.cos(latitude_rad)
    down=-dh_m
    return np.array([north,east,down])
