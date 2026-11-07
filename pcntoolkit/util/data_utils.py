"""Utilities for data operations.

This module is the shared home for programming operations (e.g., from NumPy,
pandas, xarray, itertools packages). The goal is to keep the main codebase
focused on Bayesian statistics while moving reusable programming operations
to a shared utility module.
"""

from __future__ import annotations

from functools import partial
from itertools import product
from typing import Callable, Generator

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike, NDArray


def iter_batch_combinations(
    batch_values: np.ndarray,
    unique_batch_effects: dict[str, list[str | int]],
    batch_dims: list[str],
) -> Generator[tuple[dict[str, str | int], np.ndarray]]:
    """Yield batch-effect combinations together with observation masks.

    Parameters
    ----------
    batch_values : np.ndarray
        Observed batch-effect values with shape
        ``(n_observations, n_batch_dims)``.
    unique_batch_effects : dict[str, list[str | int]]
        Allowed values for each batch-effect dimension.
    batch_dims : list[str]
        Ordered batch-effect dimensions used to interpret both
        ``batch_values`` and ``unique_batch_effects``.

    Yields
    ------
    tuple[dict[str, str | int], np.ndarray]
        A dictionary describing one batch-effect combination and a
        boolean mask that selects observations in that combination.

    Raises
    ------
    KeyError
        If ``unique_batch_effects`` is missing a requested batch
        dimension.
    """

    # Collect one ordered level list per dimension.
    # eg [['site1', 'site2'], ['M', 'F']]
    ordered_levels: list[list[str | int]] = []

    for batch_dim in batch_dims:
        # Fail fast if unique_batch_effects is missing for a requested
        # dimension.
        if batch_dim not in unique_batch_effects:
            raise KeyError(batch_dim)

        # Append the level list for this dimension.
        ordered_levels.append(list(unique_batch_effects[batch_dim]))

    # Enumerate every possible combination of levels.
    # eg ('site1', 'M'), ('site1', 'F'), ('site2', 'M'), ...
    for combo_values in product(*ordered_levels):
        # Start with all observations selected.
        mask = np.ones(batch_values.shape[0], dtype=bool)

        # Refine the mask one dimension at a time.
        for dim_idx, expected_value in enumerate(combo_values):
            # Keep only observations matching the current level.
            mask &= batch_values[:, dim_idx] == expected_value

        # Skip combinations that are absent from the observed data.
        if not mask.any():
            continue

        # Build a readable description of the current combination.
        # eg {'site': 'site1', 'sex': 'M'}
        combination = dict(zip(batch_dims, combo_values))

        # Yield: return one combination at a time (memory-efficient)
        # instead of returning all of them in a list.
        yield combination, mask


# dedupe uses np.unique only on arrays larger than UNIQUE_SAMPLE_SIZE, and only
# if at most MAX_UNIQUE_FRACTION of a sample of that size is distinct.
UNIQUE_SAMPLE_SIZE = 100_000
MAX_UNIQUE_FRACTION = 0.5

# Number of elements per dask chunk in elementwise
CHUNK_SIZE = 1_000_000


def elementwise(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *arrays: NDArray[np.float64],
    chunk_size: int = CHUNK_SIZE,
) -> tuple[NDArray[np.float64], ...]:
    """Apply an elementwise function to each array, on dask's threads.

    Arrays with at most ``chunk_size`` elements are computed directly.
    Larger arrays are split into chunks of ``chunk_size`` elements, and all
    chunks of all arrays are computed in one call to dask.

    Parameters
    ----------
    func : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        Function that works value by value and keeps the shape of its input.
    *arrays : NDArray[np.float64]
        Input arrays.
    chunk_size : int, optional
        Number of elements per dask chunk. Default is ``CHUNK_SIZE``.

    Returns
    -------
    tuple[NDArray[np.float64], ...]
        ``func(a)`` for each array ``a``, with the shape of ``a``.
    """
    if all(a.size <= chunk_size for a in arrays):
        return tuple(func(a) for a in arrays)
    lazy = [
        da.map_blocks(
            func, da.from_array(a.reshape(-1), chunks=chunk_size), dtype=float
        )
        for a in arrays
    ]
    results = da.compute(*lazy)
    return tuple(r.reshape(a.shape) for r, a in zip(results, arrays, strict=True))


# Positions of the distinct values that dedupe found last:
# (shape, flat index of the first copy of each distinct value, inverse indices)
_index_cache: (
    tuple[tuple[int, ...], NDArray[np.intp], NDArray[np.unsignedinteger]] | None
) = None


def clear_dedupe_cache() -> None:
    """Remove the positions of distinct values that ``dedupe`` keeps."""
    global _index_cache
    _index_cache = None


def _same_grouping(
    flat: NDArray[np.float64],
    index: NDArray[np.intp],
    inverse: NDArray[np.unsignedinteger],
) -> bool:
    """Return True if ``flat[index][inverse]`` gives back ``flat`` bit for bit.

    The comparison is on the bits, so ``-0.0`` and ``0.0``, or two NaNs with
    different bits, are not put in the same group.
    """
    return np.array_equal(
        flat[index][inverse].view(np.uint64).reshape(-1), flat.view(np.uint64)
    )


def _unique_positions(
    q: NDArray[np.float64], use_cache: bool
) -> tuple[NDArray[np.intp], NDArray[np.unsignedinteger]] | None:
    """Return the positions of the distinct values of ``q``, or None.

    The positions from the last call are used again if the values of ``q``
    are grouped in the same way, also if the values themselves are new.
    Returns None if ``np.unique`` would put different bits in one group.
    """
    global _index_cache
    flat = q.reshape(-1)
    cached = _index_cache if use_cache else None
    if cached is not None and cached[0] == q.shape:
        if _same_grouping(flat, cached[1], cached[2]):
            return cached[1], cached[2]
    _, index, inverse = np.unique(flat, return_index=True, return_inverse=True)
    # The smallest index type uses less memory: uint16 for 8 000 values.
    inverse = inverse.reshape(q.shape).astype(np.min_scalar_type(index.size - 1))
    if not _same_grouping(flat, index, inverse):
        return None
    if use_cache:
        _index_cache = (q.shape, index, inverse)
    return index, inverse


def _broadcast_copy(
    r: NDArray[np.float64],
    inner: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Expand ``r`` with ``inner``, then copy it along the reduced axis."""
    return np.broadcast_to(inner(r), shape).copy()


def dedupe(
    q: ArrayLike,
    sample_size: int = UNIQUE_SAMPLE_SIZE,
    max_unique_fraction: float = MAX_UNIQUE_FRACTION,
    use_cache: bool = True,
) -> tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    """Reduce ``q`` to its repeated values, for an elementwise function.

    If most values of ``q`` repeat, an expensive elementwise function needs
    to be computed only once per distinct value. For example, in SHASH
    models delta is often the same for all subjects (one value per posterior
    sample), or takes one value per batch effect level. Because the function
    is elementwise, ``expand(f(reduced))`` is bit-identical to ``f(q)``.

    ``q`` is reduced in one of these ways:

    1. If ``q`` is constant along an axis, keep one slice.
    2. If ``q`` has more than ``sample_size`` elements, and at most
       ``max_unique_fraction`` of a sample of ``sample_size`` elements is
       distinct, keep each distinct value once (``np.unique``).
    3. Otherwise, keep all values.

    The sample in step 2 makes sure that input where all values are distinct
    does not pay for a full sort.

    Step 2 keeps the positions of the distinct values for the next call.
    The next array often has the same grouping, also if its values are new:
    for example, ``compute_centiles`` computes the SHASH moments of delta
    once per centile and per response variable, and a random-effect delta
    always has one value per batch effect level and posterior sample. The
    kept positions are used only if they give back the new array bit for
    bit (about 0.09 s for 8 million values, compared to about 1.5 s for
    ``np.unique``). Thus the cache never gives a wrong result, and it is not
    necessary to clear it when a new model is made. ``clear_dedupe_cache``
    frees the memory.

    Parameters
    ----------
    q : ArrayLike
        Input array.
    sample_size : int, optional
        Minimum array size for ``np.unique``, and size of the sample that
        estimates the number of distinct values. Default is
        ``UNIQUE_SAMPLE_SIZE``.
    max_unique_fraction : float, optional
        Use ``np.unique`` only if at most this fraction of the sample is
        distinct. Default is ``MAX_UNIQUE_FRACTION``.
    use_cache : bool, optional
        Use and keep the positions of the distinct values from the last
        call. Default is True.

    Returns
    -------
    reduced : NDArray[np.float64]
        The values to compute the function on.
    expand : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        Maps the function of ``reduced`` back to the shape of ``q``.
    """
    q = np.asarray(q, dtype=np.float64)
    shape = q.shape
    if q.size == 0:
        return q, lambda r: r
    for axis in range(q.ndim):
        first = np.take(q, [0], axis=axis)
        if shape[axis] > 1 and np.array_equal(q, np.broadcast_to(first, shape)):
            reduced, inner = dedupe(first, sample_size, max_unique_fraction, use_cache)
            return reduced, partial(_broadcast_copy, inner=inner, shape=shape)
    if q.size > sample_size:
        sample = q.reshape(-1)[:: q.size // sample_size]
        if np.unique(sample).size <= max_unique_fraction * sample.size:
            positions = _unique_positions(q, use_cache)
            if positions is not None:
                index, inverse = positions
                return q.reshape(-1)[index], lambda r: r[inverse]
    return q, lambda r: r
