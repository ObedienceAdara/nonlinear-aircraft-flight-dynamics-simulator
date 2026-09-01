import numpy as np

from aircraft6dof.integrators import rk4_step


def test_rk4_matches_exponential_growth() -> None:
    x0 = np.array([1.0])
    dt = 0.1

    x1 = rk4_step(
        x0,
        dt,
        derivative=lambda x: x,
    )

    expected = np.exp(dt)
    np.testing.assert_allclose(x1, np.array([expected]), rtol=1e-5, atol=1e-8)


def test_rk4_is_exact_for_constant_derivative_up_to_roundoff() -> None:
    x0 = np.array([2.0, -1.0])
    rate = np.array([0.5, -0.25])
    dt = 2.0

    x1 = rk4_step(
        x0,
        dt,
        derivative=lambda _: rate,
    )

    np.testing.assert_allclose(x1, x0 + rate * dt)


def test_rk4_rejects_non_positive_step() -> None:
    x0 = np.array([1.0])

    try:
        rk4_step(x0, 0.0, lambda x: x)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for dt=0.")
