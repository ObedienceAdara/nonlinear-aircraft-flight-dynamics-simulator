from dataclasses import dataclass
import numpy as np
from .integrators import rk4_step
from .mathutils import euler321_from_quat
@dataclass
class SimulationHistory:
    time_s:np.ndarray; state:np.ndarray; euler_rad:np.ndarray
class Simulator:
    def __init__(self,aircraft): self.aircraft=aircraft
    def run(self,initial,controls,environment,duration_s,dt_s):
        n=int(round(duration_s/dt_s)); t=np.linspace(0,n*dt_s,n+1); X=np.empty((n+1,13)); E=np.empty((n+1,3)); s=initial
        def f(x): return self.aircraft.derivative(x,controls,environment)
        X[0]=s.vector(); E[0]=euler321_from_quat(s.quaternion_bn)
        for i in range(n):
            s=rk4_step(s,dt_s,f); X[i+1]=s.vector(); E[i+1]=euler321_from_quat(s.quaternion_bn)
        return SimulationHistory(t,X,E)
