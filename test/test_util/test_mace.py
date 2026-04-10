"""
Tests for the MACE (Mean Absolute Centile Error) metric.

Verifies that MACE is computed per combination of
batch-effect dimensions (Cartesian product), not per
individual batch-effect level.
"""

from itertools import product as real_product
from unittest.mock import patch

import numpy as np
import xarray as xr

from pcntoolkit.util.evaluator import Evaluator
from test.fixtures.evaluator_fixtures import create_test_data
from test.fixtures.evaluator_fixtures import BATCH_CONFIGS


def generate_centile_data(data):
    # Select single response variable, as evaluate_mace
    # does internally before calling _evaluate_mace
    data_with_centiles = data.sel({"response_vars": "test_metric"})

    # Generate synthetic centile data
    centile_levels = np.linspace(0.05, 0.95, 19)
    n_obs = len(data_with_centiles.observations)
    y_vals = data_with_centiles["Y"].values.ravel()
    centile_matrix = np.tile(
        np.quantile(y_vals, centile_levels)[:, np.newaxis],
        (1, n_obs),
    )
    data_with_centiles["centiles"] = xr.DataArray(
        centile_matrix,
        dims=("centile", "observations"),
        coords={"centile": centile_levels},
    )

    return data_with_centiles


def test_001_mace_should_average24Combos_when_fourBatchDims() -> None:
    """With 4 batch dimensions (2x3x2x2 = 24 combos), MACE should 
    average across all combos."""
    n_batch_effects = 4

    # Deterministic dataset with 4 batch dims (24 combos)
    data = create_test_data(n_batch_effects=n_batch_effects)

    # Add synthetic centile curves to the data
    data_with_centiles = generate_centile_data(data)

    # Expected combos: sorted Cartesian product of all
    # level lists in the 4-dimension config
    expected_combos = sorted(
        real_product(
            *[vals for _, vals in BATCH_CONFIGS[n_batch_effects]]
        )
    )

    # Spy on combos
    visited: list[tuple] = []

    def recording_product(*iterables):
        """Record every combo produced by product."""
        combos = list(real_product(*iterables))
        visited.extend(combos)
        return combos

    # Run _evaluate_macet
    evaluator = Evaluator()
    with patch(
        "pcntoolkit.util.evaluator.product",
        side_effect=recording_product,
    ):
        actual = evaluator._evaluate_mace(data_with_centiles)

    # Check that all 24 combos were averaged
    assert sorted(visited) == expected_combos
    # Check MACE value
    assert 0.0 <= actual <= 1.0


def test_002_mace_should_averageAllSubjects_when_noBatchEffects() -> None:
    """Without batch effects, MACE uses all subjects."""
    # Arrange - deterministic dataset, no batch effects
    data = create_test_data(n_batch_effects=0)

    # Add synthetic centile curves to the data
    data_with_centiles = generate_centile_data(data)

    # Run _evaluate_macet
    evaluator = Evaluator()
    actual = evaluator._evaluate_mace(data_with_centiles)

    # Check MACE value
    assert 0.0 <= actual <= 1.0
