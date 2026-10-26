import numpy as np
import pytest
import scipy.special as spp

from pcntoolkit.math_functions.shash import K, P, m1m2
from pcntoolkit.util.data_utils import clear_dedupe_cache


def _P_reference(q: np.ndarray) -> np.ndarray:
    frac = np.exp(1 / 4) / np.sqrt(8 * np.pi)
    return (spp.kv((q + 1) / 2, 1 / 4) + spp.kv((q - 1) / 2, 1 / 4)) * frac


def _m1m2_reference(epsilon: np.ndarray, delta: np.ndarray) -> tuple:
    p1 = _P_reference(1.0 / delta)
    p2 = _P_reference(2.0 / delta)
    eps_delta = epsilon / delta
    return np.sinh(eps_delta) * p1, (np.cosh(2 * eps_delta) * p2 - 1) / 2


def _deltas() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    n, s = 3000, 50
    return {
        "fixed": np.broadcast_to(rng.uniform(0.5, 2, (1, s)), (n, s)).copy(),
        "random_effect": rng.uniform(0.5, 2, (20, s))[rng.integers(0, 20, n)],
        "linear": rng.uniform(0.5, 2, (n, s)),
    }


@pytest.mark.parametrize("kind", ["fixed", "random_effect", "linear"])
def test_m1m2_bit_identical(kind):
    clear_dedupe_cache()
    delta = _deltas()[kind]
    epsilon = np.random.default_rng(1).normal(size=delta.shape)
    expected = _m1m2_reference(epsilon, delta)
    # The second call can use the dedupe cache.
    for _ in range(2):
        mean, raw_second = m1m2(epsilon, delta)
        assert np.array_equal(mean, expected[0])
        assert np.array_equal(raw_second, expected[1])


def test_P_and_K_bit_identical():
    delta = _deltas()["random_effect"]
    assert np.array_equal(P(delta), _P_reference(delta))
    assert np.array_equal(K(delta, 0.25), spp.kv(delta, 0.25))


def test_scalar_inputs():
    mean, raw_second = m1m2(0.3, 1.1)
    expected = _m1m2_reference(np.float64(0.3), np.float64(1.1))
    assert mean == expected[0] and raw_second == expected[1]
    assert np.ndim(P(1.5)) == 0
    assert K(1.5, 0.25) == spp.kv(1.5, 0.25)
