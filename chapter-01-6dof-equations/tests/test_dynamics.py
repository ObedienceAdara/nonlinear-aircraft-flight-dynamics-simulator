import numpy as np
import pytest

from aircraft6dof.dynamics import equations_of_motion


def test_zero_loads_and_zero_rates_are_equilibrium() -> None:
    state = np.zeros(6)
    inertia = np.diag([10.0, 20.0, 30.0])

    derivative = equations_of_motion(
        state,
        mass=100.0,
        inertia=inertia,
        force_body=np.zeros(3),
        moment_body=np.zeros(3),
    )

    np.testing.assert_allclose(derivative, np.zeros(6))


def test_pure_force_produces_linear_acceleration() -> None:
    state = np.zeros(6)
    inertia = np.diag([10.0, 20.0, 30.0])

    derivative = equations_of_motion(
        state,
        mass=100.0,
        inertia=inertia,
        force_body=np.array([200.0, -100.0, 50.0]),
        moment_body=np.zeros(3),
    )

    np.testing.assert_allclose(
        derivative[:3],
        np.array([2.0, -1.0, 0.5]),
    )
    np.testing.assert_allclose(derivative[3:], np.zeros(3))


def test_translational_rotation_coupling_matches_vector_equation() -> None:
    state = np.array([20.0, 3.0, -2.0, 0.1, -0.2, 0.3])
    inertia = np.diag([10.0, 20.0, 30.0])
    force = np.array([100.0, 30.0, -20.0])
    moment = np.zeros(3)

    expected_v_dot = force / 100.0 - np.cross(state[3:], state[:3])

    derivative = equations_of_motion(
        state,
        mass=100.0,
        inertia=inertia,
        force_body=force,
        moment_body=moment,
    )

    np.testing.assert_allclose(derivative[:3], expected_v_dot)


def test_rotational_equation_uses_angular_momentum_not_translational_velocity() -> None:
    state = np.array([50.0, 4.0, 2.0, 0.2, -0.1, 0.3])
    inertia = np.diag([100.0, 200.0, 300.0])
    force = np.zeros(3)
    moment = np.zeros(3)

    omega = state[3:]
    angular_momentum = inertia @ omega
    expected_omega_dot = np.linalg.solve(
        inertia,
        -np.cross(omega, angular_momentum),
    )

    derivative = equations_of_motion(
        state,
        mass=1_000.0,
        inertia=inertia,
        force_body=force,
        moment_body=moment,
    )

    np.testing.assert_allclose(derivative[3:], expected_omega_dot)


def test_invalid_mass_is_rejected() -> None:
    with pytest.raises(ValueError):
        equations_of_motion(
            np.zeros(6),
            mass=0.0,
            inertia=np.eye(3),
            force_body=np.zeros(3),
            moment_body=np.zeros(3),
        )


def test_invalid_inertia_is_rejected() -> None:
    with pytest.raises(ValueError):
        equations_of_motion(
            np.zeros(6),
            mass=100.0,
            inertia=np.array([[1.0, 2.0], [2.0, 1.0]]),
            force_body=np.zeros(3),
            moment_body=np.zeros(3),
        )
