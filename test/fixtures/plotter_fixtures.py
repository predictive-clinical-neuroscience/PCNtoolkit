"""
Shared helper functions for plotter-related tests
"""

from pcntoolkit.normative_model import NormData
import numpy as np
import xarray as xr


def create_test_data_with_z(
    n: int = 20,
    seed: int = 0,
) -> NormData:
    """Build a tiny NormData that already contains Z-scores, so there is no 
    need to fit a model to generate them. This allows testing of the plotting
    functions in isolation from the modeling code and makes testing faster.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    NormData
        Minimal dataset with one covariate, one response variable,
        one batch-effect dimension, and pre-populated Z scores.
    """
    rng = np.random.default_rng(seed)
    # One covariate (age-like), one response variable, one site column.
    X = rng.uniform(20, 80, (n, 1))
    Y = rng.normal(0, 1, (n, 1))
    batch_effects = rng.choice(["A", "B"], size=(n, 1))

    # Create the base NormData object.
    data = NormData.from_ndarrays(
        name="tiny",
        X=X,
        Y=Y,
        batch_effects=batch_effects,
        subject_ids=np.arange(n),
        attrs={
            "covariates": ["age"],
            "response_vars": ["metric"],
            "batch_effect_dims": ["site"],
        },
    )

    # Inject synthetic Z-scores directly into the dataset.
    z_values = rng.standard_normal((n,))
    data["Z"] = xr.DataArray(
        # Shape must match (observations, response_vars).
        z_values.reshape(n, 1),
        dims=("observations", "response_vars"),
        coords={
            "observations": data.coords["observations"],
            "response_vars": ["metric"],
        },
    )

    return data
