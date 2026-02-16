"""
Test to verify that MSLL is computed correctly regardless of data scale.

This test addresses the issue where MSLL was scale-dependent because the baseline
log-likelihood was computed on unscaled Y values while the model log-likelihood
was computed on scaled Y values.

The fix ensures both are computed on the same (scaled) data.
"""

import numpy as np

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR


def create_test_data(n_samples: int = 100, scale_factor: float = 1.0, seed: int = 42):
    """Create synthetic test data with a specified scale factor.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    scale_factor : float
        Factor to scale the response variable by
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    NormData
        Synthetic dataset with scaled response variables
    """
    np.random.seed(seed)
    
    # Create covariates (age-like)
    X = np.random.uniform(20, 80, (n_samples, 1))
    
    # Create response variable with linear relationship + noise
    # Scale the response by scale_factor
    Y_base = 0.5 * X[:, 0] + np.random.normal(0, 5, n_samples)
    Y = (Y_base * scale_factor).reshape(-1, 1)
    
    # Create batch effects
    batch_effects = np.random.choice([0, 1], size=(n_samples, 1)).astype(str)
    
    # Create NormData
    data = NormData.from_ndarrays(
        name="test_data",
        X=X,
        Y=Y,
        batch_effects=batch_effects,
        subject_ids=np.arange(n_samples),
        attrs={
            "covariates": ["age"],
            "response_vars": ["test_metric"],
            "batch_effect_dims": ["site"],
        },
    )
    
    return data


def test_msll_remains_similar_when_scale_changes(tmp_path):
    """
    Test that MSLL values are similar when data is scaled differently,
    confirming that MSLL is scale-independent.
    
    Using the same random seed, data at different scales should produce
    similar MSLL values.
    """
    msll_values = {}
    
    for scale_factor in [1.0, 1e-4]:
        # Create data with specified scale (keep the same seed)
        data = create_test_data(n_samples=100, scale_factor=scale_factor, seed=123)
        
        # Create and fit model
        blr = BLR()
        model = NormativeModel(
            template_regression_model=blr,
            savemodel=False,
            evaluate_model=True,
            saveresults=False,
            saveplots=False,
            save_dir=str(tmp_path / f"scale_{scale_factor}"),
            inscaler="standardize",
            outscaler="standardize",
        )
        
        model.fit(data)
        
        msll = float(data.statistics.sel(response_vars="test_metric", statistic="MSLL").values)
        msll_values[scale_factor] = msll
        print(f"Scale factor: {scale_factor}, MSLL: {msll}")
    
    # If both fitted and baseline models are computed on scaled data then the
    # MSLL should remain similar
    msll_diff = abs(msll_values[1.0] - msll_values[1e-4])
    print(f"MSLL difference between scales: {msll_diff}")
    
    # Allow some numerical tolerance depending on the specs of the PC
    assert msll_diff < 1e-8, (
        f"MSLL differs too much between scales: "
        f"scale=1.0 -> {msll_values[1.0]}, scale=1e-4 -> {msll_values[1e-4]}"
    )