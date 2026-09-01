from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class OneMinusCosineGust:
    magnitude_m_s: float; direction_ned: np.ndarray; start_s: float; rise_s: float; hold_s: float; fall_s: float
    def value(self,t):
        d=np.asarray(self.direction_ned,dtype=float); n=np.linalg.norm(d)
        if n<=1e-12:return np.zeros(3)
        tau=t-self.start_s; total=self.rise_s+self.hold_s+self.fall_s
        if tau<0 or tau>total:return np.zeros(3)
        if tau<=self.rise_s: a=.5*self.magnitude_m_s*(1-np.cos(np.pi*tau/self.rise_s))
        elif tau<=self.rise_s+self.hold_s:a=self.magnitude_m_s
        else:
            s=tau-self.rise_s-self.hold_s; a=.5*self.magnitude_m_s*(1+np.cos(np.pi*s/self.fall_s))
        return a*d/n

@dataclass
class DrydenTurbulence:
    sigma_m_s: np.ndarray; scale_length_m: np.ndarray; seed:int=1
    def __post_init__(self):
        self.sigma_m_s=np.asarray(self.sigma_m_s,dtype=float); self.scale_length_m=np.asarray(self.scale_length_m,dtype=float)
        if self.sigma_m_s.shape!=(3,) or self.scale_length_m.shape!=(3,):raise ValueError("sigma and scale_length must be (3,)")
        self.state_m_s=np.zeros(3); self.rng=np.random.default_rng(self.seed)
    def step(self,airspeed_m_s,dt):
        V=max(float(airspeed_m_s),1.0); tau=np.maximum(self.scale_length_m/V,1e-3); a=np.exp(-dt/tau)
        self.state_m_s=a*self.state_m_s+self.sigma_m_s*np.sqrt(1-a*a)*self.rng.standard_normal(3)
        return self.state_m_s.copy()
