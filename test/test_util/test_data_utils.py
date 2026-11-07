import numpy as np
import pytest

from pcntoolkit.util import data_utils
from pcntoolkit.util.data_utils import clear_dedupe_cache, dedupe, elementwise


@pytest.fixture(autouse=True)
def empty_cache():
    clear_dedupe_cache()
    yield
    clear_dedupe_cache()


def _roundtrip(q: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    reduced, expand = dedupe(q, **kwargs)
    return reduced, expand(np.sqrt(reduced))


def test_dedupe_constant_axis():
    rng = np.random.default_rng(0)
    q = np.broadcast_to(rng.uniform(1, 2, (1, 30)), (50, 30)).copy()
    reduced, out = _roundtrip(q)
    assert reduced.size == 30
    assert np.array_equal(out, np.sqrt(q))


def test_dedupe_few_distinct():
    rng = np.random.default_rng(0)
    q = rng.uniform(1, 2, (5, 30))[rng.integers(0, 5, 200)]
    reduced, out = _roundtrip(q, sample_size=1000)
    assert reduced.size == 150
    assert np.array_equal(out, np.sqrt(q))


def test_dedupe_all_distinct_skips_unique():
    q = np.random.default_rng(0).uniform(1, 2, (200, 30))
    reduced, out = _roundtrip(q, sample_size=1000)
    assert reduced is q
    assert np.array_equal(out, np.sqrt(q))
    assert data_utils._index_cache is None


def test_dedupe_empty_and_small():
    for q in [np.zeros((0, 3)), np.array([1.0, 2.0])]:
        reduced, out = _roundtrip(q)
        assert np.array_equal(out, np.sqrt(q))


def _grouped(rng: np.random.Generator, idx: np.ndarray) -> np.ndarray:
    return rng.uniform(1, 2, (5, 30))[idx]


def test_dedupe_cache_reuses_positions_for_new_values():
    # Same grouping, new values (e.g. the next response variable).
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 5, 200)
    dedupe(_grouped(rng, idx), sample_size=1000)
    cached = data_utils._index_cache
    q2 = _grouped(rng, idx)
    reduced, out = _roundtrip(q2, sample_size=1000)
    assert data_utils._index_cache is cached
    assert reduced.size == 150
    assert np.array_equal(out, np.sqrt(q2))


def test_dedupe_cache_checks_grouping():
    # Same shape, but one value leaves its group: the cache must not be used.
    rng = np.random.default_rng(0)
    q1 = _grouped(rng, rng.integers(0, 5, 200))
    dedupe(q1, sample_size=1000)
    cached = data_utils._index_cache
    q2 = q1.copy()
    q2[1, 1] = 9.0
    reduced, out = _roundtrip(q2, sample_size=1000)
    assert data_utils._index_cache is not cached
    assert 9.0 in reduced
    assert np.array_equal(out, np.sqrt(q2))


def test_dedupe_keeps_signed_zeros_apart():
    rng = np.random.default_rng(0)
    q = rng.choice([0.0, -0.0, 1.0], size=(200, 30))
    reduced, expand = dedupe(q, sample_size=1000)
    with np.errstate(divide="ignore"):
        assert np.array_equal(expand(1 / reduced), 1 / q)


def test_dedupe_without_cache():
    rng = np.random.default_rng(0)
    q = _grouped(rng, rng.integers(0, 5, 200))
    reduced, out = _roundtrip(q, sample_size=1000, use_cache=False)
    assert reduced.size == 150
    assert np.array_equal(out, np.sqrt(q))
    assert data_utils._index_cache is None


def test_elementwise_chunks():
    a = np.random.default_rng(0).uniform(1, 2, (40, 30))
    b = a[:10]
    out_a, out_b = elementwise(np.sqrt, a, b, chunk_size=100)
    assert np.array_equal(out_a, np.sqrt(a))
    assert np.array_equal(out_b, np.sqrt(b))
