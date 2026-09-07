"""Time integration and history capture for the nonlinear 6-DOF model."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .actuators import ActuatorSet
from .integrators import rk4_step
from .mathutils import euler321_from_quat
from .state import ControlInput


@dataclass
class SimulationHistory:
    time_s: np.ndarray
    state: np.ndarray
    euler_rad: np.ndarray
    control_command_rad: np.ndarray
    actuator_deflection_rad: np.ndarray


class Simulator:
    def __init__(self, aircraft):
        self.aircraft = aircraft

    def run(self, initial, controls, environment, duration_s, dt_s, actuators: ActuatorSet | None = None):
        """Integrate a simulation with optional persistent actuator dynamics.

        ``controls`` and ``environment`` may be objects or callables receiving
        simulation time in seconds and returning the corresponding object.

        When ``actuators`` is supplied, each surface command is advanced once
        per integration step with ``ActuatorSet.step`` and the returned actual
        surface positions are passed to the aircraft dynamics. The actuator
        object intentionally lives outside ``AircraftState`` because actuator
        dynamics are a separate, discrete subsystem rather than part of the
        rigid-body aircraft state. The command and actual-deflection histories
        are stored explicitly for deterministic reporting/replay.
        """
        n = int(round(duration_s / dt_s))
        t = np.linspace(0.0, n * dt_s, n + 1)
        X = np.empty((n + 1, 13))
        E = np.empty((n + 1, 3))
        C = np.empty((n + 1, 4))
        A = np.empty((n + 1, 3))
        s = initial

        def at(value, time):
            return value(time) if callable(value) else value

        X[0] = s.vector()
        E[0] = euler321_from_quat(s.quaternion_bn)
        initial_command = at(controls, float(t[0]))
        C[0] = np.array([
            initial_command.aileron,
            initial_command.elevator,
            initial_command.rudder,
            initial_command.throttle,
        ])
        if actuators is None:
            A[0] = C[0, :3]
        else:
            A[0] = np.array([
                actuators.aileron.position_rad,
                actuators.elevator.position_rad,
                actuators.rudder.position_rad,
            ])

        for i in range(n):
            ti = float(t[i])
            command = at(controls, ti)
            command_vec = np.array([
                command.aileron,
                command.elevator,
                command.rudder,
                command.throttle,
            ])
            env = at(environment, ti)

            if actuators is None:
                actual_surfaces = command_vec[:3].copy()
            else:
                actual_surfaces = actuators.step(command_vec[:3], dt_s)

            C[i] = command_vec
            A[i] = actual_surfaces
            effective = ControlInput(
                aileron=float(actual_surfaces[0]),
                elevator=float(actual_surfaces[1]),
                rudder=float(actual_surfaces[2]),
                throttle=float(command_vec[3]),
            )

            s = rk4_step(s, dt_s, lambda x: self.aircraft.derivative(x, effective, env))
            X[i + 1] = s.vector()
            E[i + 1] = euler321_from_quat(s.quaternion_bn)

        # The final actuator sample is the actuator state reached after the
        # final integration step. This preserves a complete node-aligned
        # history for plotting/replay while the interval input used by the
        # dynamics is explicitly stored at index i above.
        final_command = at(controls, float(t[-1]))
        C[-1] = np.array([
            final_command.aileron,
            final_command.elevator,
            final_command.rudder,
            final_command.throttle,
        ])
        if actuators is None:
            A[-1] = C[-1, :3]
        else:
            A[-1] = np.array([
                actuators.aileron.position_rad,
                actuators.elevator.position_rad,
                actuators.rudder.position_rad,
            ])

        return SimulationHistory(t, X, E, C, A)
