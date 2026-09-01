import numpy as np

from aircraft6dof.integrators import rk4_step


def test_rk4_constant_derivative():
    x = np.array([1.0, 2.0])
    rate = np.array([0.5, -0.25])
    result = rk4_step(x, 2.0, lambda _: rate)
    np.testing.assert_allclose(result, x + 2.0 * rate)


def test_rk4_exponential():
    result = rk4_step(np.array([1.0]), 0.1, lambda x: x)
    np.testing.assert_allclose(result, [np.exp(0.1)], rtol=1e-5)
