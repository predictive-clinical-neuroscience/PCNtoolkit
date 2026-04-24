"""Utilities for reusable array and iterator operations.

This module is the shared home for programming operations (e.g., from NumPy,
pandas, xarray, itertools packages). The goal is to keep the main codebase
focused on Bayesian statistics while moving reusable programming operations
like these to a shared utility module.
"""

from __future__ import annotations
from itertools import product
from typing import Iterator
import numpy as np


def iter_batch_combinations(
    batch_values: np.ndarray,
    unique_batch_effects: dict[str, list[str | int]],
    batch_dims: list[str],
) -> Iterator[tuple[dict[str, str | int], np.ndarray]]:
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

        yield combination, mask
