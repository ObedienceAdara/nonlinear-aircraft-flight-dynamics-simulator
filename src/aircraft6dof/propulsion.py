from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
@dataclass(frozen=True)
class Propulsion:
    max_thrust_N:float=3000.; thrust_velocity_factor:float=.12; thrust_arm_m:np.ndarray=field(default_factory=lambda:np.zeros(3))
    def force_and_moment(self,throttle,airspeed_m_s):
        t=np.clip(float(throttle),0,1); thrust=self.max_thrust_N*t*max(0.,1-self.thrust_velocity_factor*airspeed_m_s/100.)
        f=np.array([thrust,0.,0.]); return f,np.cross(self.thrust_arm_m,f)
