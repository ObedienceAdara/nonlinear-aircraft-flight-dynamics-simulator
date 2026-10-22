"""Numerical linearization of the 6-DOF model about an operating point.

The nonlinear model carries attitude as a quaternion. For linear analysis the
usual 12-state Euler form is used instead:

    x = [N, E, D, u, v, w, p, q, r, phi, theta, psi]
    u = [aileron, elevator, rudder, throttle]

``euler_state_derivative`` converts to the quaternion state, calls the same
``aircraft.derivative`` the simulator uses, and maps the quaternion rate back to
Euler-angle rates. ``linearize`` then builds A = df/dx and B = df/du by central
differences. Nothing here re-implements the physics, so the linear model cannot
drift away from the nonlinear one.

Euler kinematics are singular at theta = +/-90 deg; linearize away from that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .mathutils import quat_from_euler321
from .state import AircraftState, ControlInput, Environment

STATE_NAMES = ("N", "E", "D", "u", "v", "w", "p", "q", "r", "phi", "theta", "psi")
INPUT_NAMES = ("aileron", "elevator", "rudder", "throttle")

LON_STATES = ("u", "w", "q", "theta")
LON_INPUTS = ("elevator", "throttle")
LAT_STATES = ("v", "p", "r", "phi")
LAT_INPUTS = ("aileron", "rudder")


def euler_vector_from_state(state: AircraftState, euler_rad: Sequence[float]) -> np.ndarray:
    """12-state Euler vector from an AircraftState and its (phi, theta, psi)."""
    return np.concatenate([state.position_ned_m, state.velocity_body_m_s,
                           state.omega_body_rad_s, np.asarray(euler_rad, float)])


def state_from_euler_vector(x: np.ndarray) -> AircraftState:
    x = np.asarray(x, float)
    return AircraftState(x[0:3].copy(), x[3:6].copy(), x[6:9].copy(), quat_from_euler321(*x[9:12]))


def controls_to_vector(c: ControlInput) -> np.ndarray:
    return np.array([c.aileron, c.elevator, c.rudder, c.throttle], float)


def vector_to_controls(u: np.ndarray) -> ControlInput:
    return ControlInput(aileron=float(u[0]), elevator=float(u[1]), rudder=float(u[2]), throttle=float(u[3]))


def euler_rates(phi: float, theta: float, p: float, q: float, r: float) -> np.ndarray:
    """Euler-angle rates (phi_dot, theta_dot, psi_dot) from body rates, 3-2-1 sequence."""
    s, c = np.sin(phi), np.cos(phi)
    t, ct = np.tan(theta), np.cos(theta)
    return np.array([p + t * (q * s + r * c), q * c - r * s, (q * s + r * c) / ct])


def euler_state_derivative(aircraft, environment: Environment, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """xdot for the 12-state Euler form, using the model's own derivative()."""
    state = state_from_euler_vector(x)
    d = aircraft.derivative(state, vector_to_controls(u), environment)
    angle_rates = euler_rates(x[9], x[10], x[6], x[7], x[8])
    return np.concatenate([d.position_ned_m, d.velocity_body_m_s, d.omega_body_rad_s, angle_rates])


@dataclass(frozen=True)
class LinearModel:
    """xdot = A (x - x0) + B (u - u0), with f(x0, u0) = xdot0."""

    A: np.ndarray
    B: np.ndarray
    x0: np.ndarray
    u0: np.ndarray
    xdot0: np.ndarray
    state_names: tuple[str, ...] = STATE_NAMES
    input_names: tuple[str, ...] = INPUT_NAMES

    def _idx(self, names: Sequence[str], pool: Sequence[str]) -> list[int]:
        return [pool.index(n) for n in names]

    def subsystem(self, states: Sequence[str], inputs: Sequence[str]) -> "LinearModel":
        """Slice out a sub-model (for example the longitudinal block)."""
        si, ii = self._idx(states, self.state_names), self._idx(inputs, self.input_names)
        return LinearModel(
            A=self.A[np.ix_(si, si)], B=self.B[np.ix_(si, ii)],
            x0=self.x0[si], u0=self.u0[ii], xdot0=self.xdot0[si],
            state_names=tuple(states), input_names=tuple(inputs),
        )

    @property
    def longitudinal(self) -> "LinearModel":
        return self.subsystem(LON_STATES, LON_INPUTS)

    @property
    def lateral(self) -> "LinearModel":
        return self.subsystem(LAT_STATES, LAT_INPUTS)

    def coupling_ratio(self) -> float:
        """Size of the longitudinal-lateral cross terms relative to the diagonal blocks.

        Near zero for symmetric wings-level flight, where treating the two
        subsystems separately is valid.
        """
        li = self._idx(LON_STATES, self.state_names)
        ai = self._idx(LAT_STATES, self.state_names)
        cross = np.linalg.norm(self.A[np.ix_(li, ai)]) + np.linalg.norm(self.A[np.ix_(ai, li)])
        diag = np.linalg.norm(self.A[np.ix_(li, li)]) + np.linalg.norm(self.A[np.ix_(ai, ai)])
        return float(cross / diag)

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)


def linearize(
    aircraft,
    environment: Environment,
    x0: np.ndarray,
    u0: np.ndarray,
    *,
    step: float = 1e-6,
) -> LinearModel:
    """Central-difference Jacobians A = df/dx and B = df/du at (x0, u0).

    x0 is the 12-state Euler vector and u0 the 4-vector of controls. The point
    does not have to be a trim, but eigenvalues are only meaningful at one.
    """
    x0 = np.asarray(x0, float)
    u0 = np.asarray(u0, float)
    f0 = euler_state_derivative(aircraft, environment, x0, u0)

    A = np.empty((12, 12))
    for j in range(12):
        h = step * max(1.0, abs(x0[j]))
        dx = np.zeros(12)
        dx[j] = h
        A[:, j] = (euler_state_derivative(aircraft, environment, x0 + dx, u0)
                   - euler_state_derivative(aircraft, environment, x0 - dx, u0)) / (2 * h)

    B = np.empty((12, 4))
    for j in range(4):
        h = step * max(1.0, abs(u0[j]))
        du = np.zeros(4)
        du[j] = h
        B[:, j] = (euler_state_derivative(aircraft, environment, x0, u0 + du)
                   - euler_state_derivative(aircraft, environment, x0, u0 - du)) / (2 * h)

    return LinearModel(A=A, B=B, x0=x0, u0=u0, xdot0=f0)


def linearize_trim(aircraft, environment: Environment, trim_result, *, step: float = 1e-6) -> LinearModel:
    """Linearize about a ``TrimResult``."""
    x0 = euler_vector_from_state(trim_result.state, trim_result.euler_rad)
    u0 = controls_to_vector(trim_result.controls)
    return linearize(aircraft, environment, x0, u0, step=step)


def simulate_linear(
    model: LinearModel,
    input_of_t: Callable[[float], np.ndarray],
    duration_s: float,
    dt_s: float,
    dx0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 response of the perturbation model.

    ``input_of_t(t)`` returns the control *perturbation* du (not absolute
    controls), sized for ``model.input_names``. Returns (t, dx) with dx the state
    perturbation history, shape (n+1, n_states).

    Like ``Simulator.run``, the input is sampled once at the start of each step
    and held (zero-order hold) across the RK4 stages, so step and pulse inputs
    line up sample for sample with the nonlinear simulation.
    """
    n = int(round(duration_s / dt_s))
    t = np.linspace(0.0, n * dt_s, n + 1)
    dx = np.zeros((n + 1, model.A.shape[0]))
    if dx0 is not None:
        dx[0] = dx0

    for i in range(n):
        xi = dx[i]
        bu = model.B @ np.asarray(input_of_t(float(t[i])), float)

        def f(x: np.ndarray) -> np.ndarray:
            return model.A @ x + bu

        k1 = f(xi)
        k2 = f(xi + dt_s / 2 * k1)
        k3 = f(xi + dt_s / 2 * k2)
        k4 = f(xi + dt_s * k3)
        dx[i + 1] = xi + dt_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, dx