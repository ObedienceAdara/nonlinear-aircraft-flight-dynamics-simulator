from dataclasses import dataclass
import numpy as np
@dataclass
class ActuatorChannel:
    time_constant_s:float; rate_limit_rad_s:float; position_limit_rad:float; position_rad:float=0.
    def __post_init__(self):
        if not self.time_constant_s>0: raise ValueError("time_constant_s must be positive")
        if not self.rate_limit_rad_s>0: raise ValueError("rate_limit_rad_s must be positive")
        if not self.position_limit_rad>0: raise ValueError("position_limit_rad must be positive")
    def step(self,command_rad,dt):
        if not dt>0: raise ValueError("dt must be positive")
        target=np.clip(command_rad,-self.position_limit_rad,self.position_limit_rad)
        rate=np.clip((target-self.position_rad)/self.time_constant_s,-self.rate_limit_rad_s,self.rate_limit_rad_s)
        self.position_rad=float(np.clip(self.position_rad+rate*dt,-self.position_limit_rad,self.position_limit_rad)); return self.position_rad
@dataclass
class ActuatorSet:
    aileron:ActuatorChannel; elevator:ActuatorChannel; rudder:ActuatorChannel
    def step(self,commands_rad,dt):
        c=np.asarray(commands_rad,float); return np.array([self.aileron.step(c[0],dt),self.elevator.step(c[1],dt),self.rudder.step(c[2],dt)])