from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AircraftState:
    position_ned_m: np.ndarray
    velocity_body_m_s: np.ndarray
    omega_body_rad_s: np.ndarray
    quaternion_bn: np.ndarray
    def vector(self):
        return np.concatenate([self.position_ned_m,self.velocity_body_m_s,self.omega_body_rad_s,self.quaternion_bn])

@dataclass(frozen=True)
class ControlInput:
    aileron: float=0.0
    elevator: float=0.0
    rudder: float=0.0
    throttle: float=0.0

@dataclass(frozen=True)
class VehicleGeometry:
    mass_kg: float
    inertia_kg_m2: np.ndarray
    wing_area_m2: float
    wing_span_m: float
    mean_chord_m: float
    cg_to_ref_m: np.ndarray=field(default_factory=lambda: np.zeros(3))

@dataclass(frozen=True)
class Environment:
    wind_ned_m_s: np.ndarray=field(default_factory=lambda: np.zeros(3))
    gust_ned_m_s: np.ndarray=field(default_factory=lambda: np.zeros(3))
    density_kg_m3: float=1.225
    speed_of_sound_m_s: float=340.294
    gravity_ned_m_s2: np.ndarray=field(default_factory=lambda: np.array([0.,0.,9.80665]))
