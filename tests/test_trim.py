import numpy as np
import pytest

from aircraft6dof import Environment, TrimCondition, TrimError
from aircraft6dof.trim import trim
from aircraft6dof.mathutils import dcm_body_to_ned_from_quat

V = 55.0
H = 1000.0


def air_relative_velocity_ned(result, env):
    C = dcm_body_to_ned_from_quat(result.state.quaternion_bn)
    return C @ result.state.velocity_body_m_s - env.wind_ned_m_s


def test_level_trim_is_steady(aircraft, still_air):
    r = trim(aircraft, still_air, TrimCondition(V, H))
    assert r.converged and r.residual_norm < 1e-9
    assert r.euler_rad[0] == 0.0
    assert r.euler_rad[1] == pytest.approx(r.alpha_rad, abs=1e-12)   # level flight: theta = alpha
    assert np.linalg.norm(r.state.velocity_body_m_s) == pytest.approx(V)
    assert r.state.position_ned_m[2] == -H
    assert 0.0 < r.controls.throttle < 1.0


@pytest.mark.parametrize("gamma_deg", [-4.0, 3.0, 6.0])
def test_climb_trim_hits_flight_path_angle(aircraft, still_air, gamma_deg):
    gamma = np.deg2rad(gamma_deg)
    r = trim(aircraft, still_air, TrimCondition(V, H, flight_path_rad=gamma))
    v_ned = air_relative_velocity_ned(r, still_air)
    assert -v_ned[2] == pytest.approx(V * np.sin(gamma), abs=1e-9)    # climb rate
    assert r.residual_norm < 1e-9
    # climbing needs more thrust than level flight
    level = trim(aircraft, still_air, TrimCondition(V, H))
    assert (r.controls.throttle > level.controls.throttle) == (gamma_deg > 0)


def test_coordinated_turn(aircraft, still_air):
    rate = np.deg2rad(5.0)
    r = trim(aircraft, still_air, TrimCondition(V, H, turn_rate_rad_s=rate))
    phi, theta, _ = r.euler_rad
    p, q, rr = r.state.omega_body_rad_s
    psi_dot = (q * np.sin(phi) + rr * np.cos(phi)) / np.cos(theta)
    assert psi_dot == pytest.approx(rate, abs=1e-12)
    assert r.residual_norm < 1e-8
    # bank angle close to the coordinated-turn value atan(V * psi_dot / g)
    assert phi == pytest.approx(np.arctan(V * rate / 9.80665), abs=np.deg2rad(1.5))
    # right turn needs positive roll
    assert phi > 0.0
    # tighter turn -> more bank and more elevator
    tighter = trim(aircraft, still_air, TrimCondition(V, H, turn_rate_rad_s=2 * rate))
    assert tighter.euler_rad[0] > phi


def test_steady_sideslip_holds_beta(aircraft, still_air):
    beta = np.deg2rad(4.0)
    r = trim(aircraft, still_air, TrimCondition(V, H, sideslip_rad=beta))
    vb = r.state.velocity_body_m_s
    assert np.arcsin(vb[1] / np.linalg.norm(vb)) == pytest.approx(beta, abs=1e-12)
    assert r.residual_norm < 1e-9
    assert abs(r.controls.rudder) > 0.0 and abs(r.controls.aileron) > 0.0


def test_trim_with_steady_wind_is_relative_to_air(aircraft, still_air):
    windy = Environment(
        wind_ned_m_s=np.array([8.0, -5.0, 0.0]),
        density_kg_m3=still_air.density_kg_m3,
        speed_of_sound_m_s=still_air.speed_of_sound_m_s,
        gravity_ned_m_s2=still_air.gravity_ned_m_s2,
    )
    r = trim(aircraft, windy, TrimCondition(V, H, heading_rad=np.deg2rad(30)))
    assert np.linalg.norm(air_relative_velocity_ned(r, windy)) == pytest.approx(V, abs=1e-9)
    calm = trim(aircraft, still_air, TrimCondition(V, H))
    assert r.alpha_rad == pytest.approx(calm.alpha_rad, abs=1e-9)
    assert r.controls.throttle == pytest.approx(calm.controls.throttle, abs=1e-9)


def test_wind_in_a_turn_is_rejected(aircraft, still_air):
    windy = Environment(wind_ned_m_s=np.array([5.0, 0.0, 0.0]), density_kg_m3=still_air.density_kg_m3,
                        gravity_ned_m_s2=still_air.gravity_ned_m_s2)
    with pytest.raises(ValueError):
        trim(aircraft, windy, TrimCondition(V, H, turn_rate_rad_s=0.05))


def test_unreachable_flight_path_is_an_error(aircraft, still_air):
    with pytest.raises((ValueError, TrimError)):
        trim(aircraft, still_air, TrimCondition(V, H, flight_path_rad=np.deg2rad(80.0)))


def test_impossible_trim_raises_with_last_iterate(aircraft, still_air):
    # A 35 degree climb needs far more thrust than the throttle can give; the
    # solver may reach it with throttle > 1 (flagged) or fail (TrimError).
    cond = TrimCondition(V, H, flight_path_rad=np.deg2rad(35.0))
    try:
        r = trim(aircraft, still_air, cond)
    except TrimError as exc:
        assert exc.result is not None and not exc.result.converged
    else:
        assert any("throttle" in w for w in r.warnings)


def test_bad_airspeed_rejected():
    with pytest.raises(ValueError):
        TrimCondition(0.0)
    with pytest.raises(ValueError):
        TrimCondition(float("nan"))