import numpy as np
import pytest

from aircraft6dof import AircraftState, ControlInput, Simulator, TrimCondition, linearize_trim
from aircraft6dof.linearize import (controls_to_vector, euler_rates, euler_state_derivative,
                                    euler_vector_from_state, linearize, simulate_linear, state_from_euler_vector)
from aircraft6dof.trim import trim
from aircraft6dof.mathutils import euler321_from_quat
from aircraft6dof.modes import all_modes, lateral_modes, longitudinal_modes

V = 55.0
H = 1000.0


@pytest.fixture(scope="module")
def level(aircraft, still_air):
    tr = trim(aircraft, still_air, TrimCondition(V, H))
    return tr, linearize_trim(aircraft, still_air, tr)


def test_euler_rates_match_quaternion_kinematics(aircraft, still_air):
    rng = np.random.default_rng(3)
    for _ in range(10):
        euler = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-3, 3)])
        x = np.concatenate([np.zeros(3), [50, 3, 4], rng.normal(size=3) * 0.2, euler])
        s = state_from_euler_vector(x)
        d = aircraft.derivative(s, ControlInput(0, 0, 0, 0.5), still_air)
        h = 1e-7
        q2 = s.quaternion_bn + h * d.quaternion_bn
        numeric = (euler321_from_quat(q2) - euler321_from_quat(s.quaternion_bn)) / h
        assert np.allclose(euler_rates(*x[9:11], *x[6:9])[:], numeric, atol=1e-5)


def test_trim_is_an_equilibrium_of_the_euler_form(aircraft, still_air, level):
    tr, lin = level
    assert np.max(np.abs(lin.xdot0[3:])) < 1e-9          # everything except N, E, D rates
    assert lin.xdot0[2] == pytest.approx(0.0, abs=1e-9)  # level: no climb


def test_jacobian_is_insensitive_to_step_size(aircraft, still_air, level):
    tr, lin = level
    x0, u0 = lin.x0, lin.u0
    other = linearize(aircraft, still_air, x0, u0, step=1e-5)
    assert np.allclose(lin.A, other.A, atol=1e-5, rtol=1e-4)
    assert np.allclose(lin.B, other.B, atol=1e-4, rtol=1e-4)


def test_linear_model_is_decoupled_at_symmetric_trim(level):
    _, lin = level
    assert lin.coupling_ratio() < 1e-6
    assert lin.longitudinal.A.shape == (4, 4) and lin.lateral.B.shape == (4, 2)


def test_modes_are_physically_sensible(level):
    _, lin = level
    lon, lat = longitudinal_modes(lin), lateral_modes(lin)
    sp, ph = lon["short_period"], lon["phugoid"]
    assert sp.stable and ph.stable
    assert sp.wn_rad_s > 5 * ph.wn_rad_s                  # time-scale separation
    assert 0.3 < sp.zeta < 0.9
    assert 0.0 < ph.zeta < 0.2
    dr, roll, spiral = lat["dutch_roll"], lat["roll_subsidence"], lat["spiral"]
    assert dr.stable and dr.zeta < 0.5
    assert roll.eigenvalue.real < -3.0
    assert abs(spiral.eigenvalue.real) < 0.1
    assert len(all_modes(lin)) == 5


def test_dutch_roll_matches_adverse_yaw_study_numbers(level):
    # Updated after the CG moment translation in the current nonlinear model.
    _, lin = level
    dr = lateral_modes(lin)["dutch_roll"]
    assert dr.zeta == pytest.approx(0.2385, abs=0.002)
    assert dr.wn_rad_s == pytest.approx(3.7425, abs=0.005)


def _linear_vs_nonlinear_errors(aircraft, still_air, tr, lin, amp_deg):
    """Per-state max error of the linear response, as a fraction of the nonlinear peak."""
    de = da = np.deg2rad(amp_deg)

    def du(t):                      # perturbation order: aileron, elevator, rudder, throttle
        elev = de if 1.0 <= t < 2.0 else (-de if 2.0 <= t < 3.0 else 0.0)
        return np.array([da if 5.0 <= t < 6.0 else 0.0, elev, 0.0, 0.0])

    def controls(t):
        d = du(t)
        return ControlInput(tr.controls.aileron + d[0], tr.controls.elevator + d[1],
                            tr.controls.rudder + d[2], tr.controls.throttle + d[3])

    T, dt = 10.0, 0.01
    hist = Simulator(aircraft).run(tr.state, controls, lambda t: still_air, duration_s=T, dt_s=dt)
    _, dx = simulate_linear(lin, du, T, dt)
    nl = np.column_stack([hist.state[:, 3:9], hist.euler_rad]) - np.concatenate([lin.x0[3:9], lin.x0[9:12]])
    lx = dx[:, 3:12]
    names = ["u", "v", "w", "p", "q", "r", "phi", "theta", "psi"]
    return {n: np.max(np.abs(nl[:, k] - lx[:, k])) / np.max(np.abs(nl[:, k]))
            for k, n in enumerate(names) if np.max(np.abs(nl[:, k])) > 1e-6}


def test_linear_model_tracks_nonlinear_simulation(aircraft, still_air, level):
    """Small elevator doublet + aileron pulse: linear and nonlinear responses agree."""
    tr, lin = level
    for name, err in _linear_vs_nonlinear_errors(aircraft, still_air, tr, lin, 0.25).items():
        assert err < 0.03, f"{name}: {100 * err:.2f}% of peak"


def test_linearization_error_shrinks_with_perturbation_size(aircraft, still_air, level):
    """A correct Jacobian leaves only second-order error: 4x smaller input -> ~4x smaller error."""
    tr, lin = level
    big = _linear_vs_nonlinear_errors(aircraft, still_air, tr, lin, 0.5)
    small = _linear_vs_nonlinear_errors(aircraft, still_air, tr, lin, 0.125)
    for name in big:
        assert small[name] < 0.5 * big[name], name


def test_simulate_linear_zero_input_stays_at_trim(level):
    _, lin = level
    t, dx = simulate_linear(lin, lambda t: np.zeros(4), 5.0, 0.01)
    assert np.max(np.abs(dx)) == 0.0


def test_round_trip_euler_vector(level):
    tr, lin = level
    x = euler_vector_from_state(tr.state, tr.euler_rad)
    back = state_from_euler_vector(x)
    assert np.allclose(back.velocity_body_m_s, tr.state.velocity_body_m_s)
    assert np.allclose(euler321_from_quat(back.quaternion_bn), tr.euler_rad)
    assert np.allclose(controls_to_vector(tr.controls)[1], tr.controls.elevator)