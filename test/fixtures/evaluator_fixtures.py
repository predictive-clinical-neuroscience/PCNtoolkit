"""
Shared helper functions for evaluator-related tests
(MACE, MSLL, etc.).

These utilities are plain functions rather than pytest
fixtures so that individual tests can control their
parameters directly.
"""

import numpy as np

from pcntoolkit.dataio.norm_data import NormData

# Predefined batch-effect configurations
BATCH_CONFIGS: dict[int, list[tuple[str, list[str]]]] = {
    # One dimension: site with two levels
    1: [
        ("site", ["sA", "sB"]),
    ],
    # Two dimensions: site and sex
    2: [
        ("site", ["sA", "sB"]),
        ("sex", ["M", "F", "O"]),
    ],
    # Three dimensions: site, sex, and scanner
    3: [
        ("site", ["sA", "sB"]),
        ("sex", ["M", "F", "O"]),
        ("scanner", ["sc1", "sc2"]),
    ],
    # Four dimensions: site, sex, scanner, and phase
    4: [
        ("site", ["sA", "sB"]),
        ("sex", ["M", "F", "O"]),
        ("scanner", ["sc1", "sc2"]),
        ("phase", ["p1", "p2"]),
    ],
}


def create_test_data(
    n_samples: int = 100,
    scale_factor: float = 1.0,
    seed: int = 42,
    n_batch_effects: int = 1,
) -> NormData:
    """Create random synthetic data for evaluator tests.

    Generates covariates, a linear response,
    and the requested number of batch-effect dimensions.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    scale_factor : float
        Factor to scale the response variable by
    seed : int
        Random seed for reproducibility
    n_batch_effects : int
        Number of batch-effect dimensions to include.
        Supported values: 0-4.

    Returns
    -------
    NormData
        Synthetic dataset with scaled response variables
    """
    # Fix the random seed for reproducibility
    np.random.seed(seed)
    # Draw age-like covariates uniformly in [20, 80]
    X = np.random.uniform(20, 80, (n_samples, 1))
    # Create a linear signal with Gaussian noise
    Y_base = (
        0.5 * X[:, 0]
        + np.random.normal(0, 5, n_samples)
    )
    # Apply scale factor and reshape to (n_samples, 1)
    Y = (Y_base * scale_factor).reshape(-1, 1)

    # Create batch effects
    if n_batch_effects == 0:
        # No batch effects: pass None to from_ndarrays
        batch_effects = None
        # Attrs without batch_effect_dims key
        my_attrs: dict[str, list[str]] = {
            "covariates": ["age"],
            "response_vars": ["test_metric"],
        }
    else:
        # Retrieve the predefined config for this count
        config = BATCH_CONFIGS[n_batch_effects]
        # Sample each dimension independently
        be_cols = [
            np.random.choice(vals, size=n_samples)
            for _, vals in config
        ]
        # Stack columns into (n_samples, n_dims) array
        batch_effects = np.column_stack(be_cols)
        # Include dimension names in attrs
        my_attrs = {
            "covariates": ["age"],
            "response_vars": ["test_metric"],
            "batch_effect_dims": [
                name for name, _ in config
            ],
        }

    # Build and return the NormData
    return NormData.from_ndarrays(
        name="test_data",
        X=X,
        Y=Y,
        batch_effects=batch_effects,
        subject_ids=np.arange(n_samples),
        attrs=my_attrs,
    )
