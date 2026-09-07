"""Time integration, actuator dynamics, and fail-fast health checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .actuators import ActuatorSet
from .integrators import RK4StateError, rk4_step
from .mathutils import dcm_body_to_ned_from_quat, euler321_from_quat
from .state import AircraftState, ControlInput


@dataclass(frozen=True)
class SimulationGuardConfig:
    """Configurable fail-fast simulation sanity checks.

    Defaults are deliberately generous generic-simulation limits rather than
    aircraft certification limits. Set ``enabled=False`` or individual bounds
    to ``None`` for intentionally extreme test cases.
    """

    enabled: bool = True
    max_airspeed_m_s: float | None = 400.0
    max_angular_rate_rad_s: float | None = 12.0
    max_abs_alpha_rad: float | None = float(np.deg2rad(85.0))
    max_abs_beta_rad: float | None = float(np.deg2rad(85.0))
    history_size: int = 10

    def __post_init__(self):
        if self.history_size < 1:
            raise ValueError("history_size must be >= 1")
        bounds = {
            "max_airspeed_m_s": self.max_airspeed_m_s,
            "max_angular_rate_rad_s": self.max_angular_rate_rad_s,
            "max_abs_alpha_rad": self.max_abs_alpha_rad,
            "max_abs_beta_rad": self.max_abs_beta_rad,
        }
        for name, value in bounds.items():
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite, or None")


class SimulationDivergenceError(RuntimeError):
    """Fail-fast simulation diagnostic containing the last valid states."""

    def __init__(self, *, time_s: float, step_index: int, field: str, reason: str, last_valid_states):
        self.time_s = float(time_s)
        self.step_index = int(step_index)
        self.field = field
        self.reason = reason
        self.last_valid_states = list(last_valid_states)
        super().__init__(self.report())

    def report(self) -> str:
        lines = [
            "Simulation stopped by divergence/sanity guard.",
            f"Failed at t={self.time_s:.6f} s, step={self.step_index}.",
            f"Offending field: {self.field}.",
            f"Reason: {self.reason}",
            f"Last {len(self.last_valid_states)} valid state(s):",
        ]
        for t, vector in self.last_valid_states:
            lines.append(f"  t={t:.6f} s: {np.asarray(vector).tolist()}")
        return "\n".join(lines)


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

    @staticmethod
    def _state_fields(state: AircraftState):
        return (
            ("position_ned_m", state.position_ned_m),
            ("velocity_body_m_s", state.velocity_body_m_s),
            ("omega_body_rad_s", state.omega_body_rad_s),
            ("quaternion_bn", state.quaternion_bn),
        )

    @classmethod
    def _check_state_finite(cls, state: AircraftState, time_s: float, step_index: int):
        for name, value in cls._state_fields(state):
            if not np.all(np.isfinite(value)):
                bad = np.flatnonzero(~np.isfinite(value)).tolist()
                raise SimulationDivergenceError(
                    time_s=time_s,
                    step_index=step_index,
                    field=name,
                    reason=f"state contains non-finite value(s) at indices {bad}: {np.asarray(value).tolist()}",
                    last_valid_states=[],
                )

    @staticmethod
    def _air_data(state: AircraftState, environment):
        C = dcm_body_to_ned_from_quat(state.quaternion_bn)
        v_n = C @ state.velocity_body_m_s
        vrel_n = v_n - (environment.wind_ned_m_s + environment.gust_ned_m_s)
        vrel_b = C.T @ vrel_n
        V = float(np.linalg.norm(vrel_b))
        if V > 1e-9:
            alpha = float(np.arctan2(vrel_b[2], vrel_b[0]))
            beta = float(np.arcsin(np.clip(vrel_b[1] / V, -1.0, 1.0)))
        else:
            alpha = 0.0
            beta = 0.0
        return V, alpha, beta

    @classmethod
    def _check_sanity(cls, state: AircraftState, environment, guard: SimulationGuardConfig, time_s: float, step_index: int):
        if not guard.enabled:
            return
        V, alpha, beta = cls._air_data(state, environment)
        angular_rate = float(np.linalg.norm(state.omega_body_rad_s))
        checks = (
            ("airspeed_magnitude", V, guard.max_airspeed_m_s, "airspeed magnitude"),
            ("angular_rate_magnitude", angular_rate, guard.max_angular_rate_rad_s, "angular-rate magnitude"),
            ("angle_of_attack", abs(alpha), guard.max_abs_alpha_rad, "|angle of attack|"),
            ("sideslip_angle", abs(beta), guard.max_abs_beta_rad, "|sideslip|"),
        )
        for field, value, limit, label in checks:
            if limit is not None and value > limit:
                raise SimulationDivergenceError(
                    time_s=time_s,
                    step_index=step_index,
                    field=field,
                    reason=f"{label}={value:.9g} exceeds configured limit={limit:.9g}",
                    last_valid_states=[],
                )

    def run(self, initial, controls, environment, duration_s, dt_s, actuators: ActuatorSet | None = None,
            guard: SimulationGuardConfig | None = None):
        """Integrate with persistent actuators and fail-fast state/sanity guards."""
        guard = SimulationGuardConfig() if guard is None else guard
        n = int(round(duration_s / dt_s))
        t = np.linspace(0.0, n * dt_s, n + 1)
        X = np.empty((n + 1, 13))
        E = np.empty((n + 1, 3))
        C = np.empty((n + 1, 4))
        A = np.empty((n + 1, 3))
        s = initial
        recent_valid = deque(maxlen=guard.history_size)

        def at(value, time):
            return value(time) if callable(value) else value

        initial_time = float(t[0])
        self._check_state_finite(s, initial_time, 0)
        env0 = at(environment, initial_time)
        self._check_sanity(s, env0, guard, initial_time, 0)
        recent_valid.append((initial_time, s.vector().copy()))

        X[0] = s.vector()
        E[0] = euler321_from_quat(s.quaternion_bn)
        initial_command = at(controls, initial_time)
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

            try:
                s = rk4_step(
                    s,
                    dt_s,
                    lambda x: self.aircraft.derivative(x, effective, env),
                    time_s=float(t[i + 1]),
                    step_index=i + 1,
                )
                self._check_sanity(s, env, guard, float(t[i + 1]), i + 1)
            except RK4StateError as exc:
                raise SimulationDivergenceError(
                    time_s=float(t[i + 1]),
                    step_index=i + 1,
                    field="rk4_output_state",
                    reason=str(exc),
                    last_valid_states=recent_valid,
                ) from None
            except SimulationDivergenceError as exc:
                exc.last_valid_states = list(recent_valid)
                exc.args = (exc.report(),)
                raise

            X[i + 1] = s.vector()
            E[i + 1] = euler321_from_quat(s.quaternion_bn)
            recent_valid.append((float(t[i + 1]), s.vector().copy()))

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
