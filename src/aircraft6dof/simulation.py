"""Time integration and history capture for the nonlinear 6-DOF model."""

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .integrators import rk4_step
from .mathutils import euler321_from_quat

@dataclass
class SimulationHistory:
    time_s: np.ndarray
    state: np.ndarray
    euler_rad: np.ndarray

class Simulator:
    def __init__(self, aircraft):
        self.aircraft = aircraft

    def run(self, initial, controls, environment, duration_s, dt_s):
        """Integrate a simulation with constant or time-varying inputs.

        ``controls`` and ``environment`` may be objects or callables receiving
        simulation time in seconds and returning the corresponding object.
        """
        n = int(round(duration_s / dt_s))
        t = np.linspace(0.0, n * dt_s, n + 1)
        X = np.empty((n + 1, 13))
        E = np.empty((n + 1, 3))
        s = initial

        def at(value, time):
            return value(time) if callable(value) else value

        X[0] = s.vector()
        E[0] = euler321_from_quat(s.quaternion_bn)
        for i in range(n):
            ti = t[i]
            u = at(controls, ti)
            env = at(environment, ti)
            s = rk4_step(s, dt_s, lambda x: self.aircraft.derivative(x, u, env))
            X[i + 1] = s.vector()
            E[i + 1] = euler321_from_quat(s.quaternion_bn)
        return SimulationHistory(t, X, E)
