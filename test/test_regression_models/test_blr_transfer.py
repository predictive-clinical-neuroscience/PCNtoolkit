"""
Tests for BLR.transfer with few adaptation subjects (issue #621).

The new-site data are simulated from the reference prediction itself:

    y_i = ys_i + offset + scale * s_i * eps_i,   eps_i ~ N(0, 1),

with ys_i and s_i^2 the predictive mean and variance of the reference model.
This is the model that BLR.transfer assumes.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.stats import t as student_t

from pcntoolkit.math_functions.basis_function import BsplineBasisFunction
from pcntoolkit.regression_model.blr import BLR

TRAIN_BE_MAPS = {"site": {"0": 0, "1": 1}, "sex": {"0": 0, "1": 1}}
NEW_BE_MAPS = {"site": {"new": 0}, "sex": {"F": 0, "M": 1}}
OFFSET = {0: 0.4, 1: -0.3}
SCALE = {0: 1.3, 1: 0.8}


def _xr_inputs(
    x: np.ndarray, be: np.ndarray, y: np.ndarray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Wrap numpy arrays in the DataArrays that BLR expects."""
    obs = np.arange(x.shape[0])
    X = xr.DataArray(
        x[:, None],
        dims=("observations", "covariates"),
        coords={"observations": obs, "covariates": ["age"]},
    )
    B = xr.DataArray(
        be,
        dims=("observations", "batch_effect_dims"),
        coords={"observations": obs, "batch_effect_dims": ["site", "sex"]},
    )
    Y = xr.DataArray(y, dims=("observations",), coords={"observations": obs})
    return X, B, Y


@pytest.fixture(scope="module")
def reference_blr() -> BLR:
    """A BLR with constant noise, fitted on 1000 simulated subjects of 2 sites."""
    rng = np.random.default_rng(621)
    n = 1000
    x = rng.uniform(-1.5, 1.5, n)
    be = np.stack([rng.integers(0, 2, n), rng.integers(0, 2, n)], axis=1)
    y = 0.5 * x + 0.3 * x**2 + 0.5 * rng.standard_normal(n)
    blr = BLR(
        basis_function_mean=BsplineBasisFunction(basis_column=0, nknots=5, degree=3)
    )
    X, B, Y = _xr_inputs(x, be, y)
    blr.fit(X, B, TRAIN_BE_MAPS, Y)
    return blr


def _new_site(
    rng: np.random.Generator, blr: BLR, n_per_sex: tuple[int, int]
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Simulate healthy subjects of a new site from the reference prediction."""
    sex = np.repeat([0, 1], n_per_sex)
    x = rng.uniform(-1.5, 1.5, sex.size)
    # Reference prediction at the batch that BLR.transfer uses (first levels)
    ys, s2 = blr.ys_s2(x[:, None], np.zeros((x.size, 2), dtype=int))
    offset = np.vectorize(OFFSET.get)(sex)
    scale = np.vectorize(SCALE.get)(sex)
    y = ys + offset + scale * np.sqrt(s2) * rng.standard_normal(x.size)
    be = np.stack([np.zeros(x.size, dtype=int), sex], axis=1)
    return _xr_inputs(x, be, y)


def test_transfer_raises_when_a_cell_has_one_subject(reference_blr: BLR) -> None:
    """A cell with a single adaptation subject has no estimable scale.

    Before this check, such a cell got a scale of 0 and infinite Z-scores.
    """
    rng = np.random.default_rng(1)
    X, B, Y = _new_site(rng, reference_blr, (10, 1))
    with pytest.raises(ValueError, match=r"site=new, sex=M \(n=1\)"):
        reference_blr.transfer(X, B, NEW_BE_MAPS, Y)


def test_transfer_accepts_a_cell_absent_from_the_data(reference_blr: BLR) -> None:
    """A level listed in be_maps but with no subject is skipped, not an error."""
    rng = np.random.default_rng(2)
    X, B, Y = _new_site(rng, reference_blr, (10, 0))
    transferred = reference_blr.transfer(X, B, NEW_BE_MAPS, Y)
    assert list(transferred.correction_coefficients) == ["(0, 0)"]


def test_predict_raises_on_a_cell_absent_from_transfer(reference_blr: BLR) -> None:
    """Predicting a combination without adaptation subjects names it (#644)."""
    rng = np.random.default_rng(4)
    X, B, Y = _new_site(rng, reference_blr, (10, 0))
    transferred = reference_blr.transfer(X, B, NEW_BE_MAPS, Y)
    Xt, Bt, Yt = _new_site(rng, reference_blr, (5, 5))
    with pytest.raises(ValueError, match=r"No subjects for site=new, sex=M"):
        transferred.forward(Xt, Bt, Yt)


def test_transfer_correction_coefficients(reference_blr: BLR) -> None:
    """The stored offset and scale factor match their definition.

    Offset: mean residual. Scale factor: unbiased SD of the residuals (ddof=1)
    over the mean predicted SD, times sqrt(1 + 1/n) for the uncertainty of the
    offset.
    """
    rng = np.random.default_rng(3)
    n = 10
    X, B, Y = _new_site(rng, reference_blr, (n, n))
    transferred = reference_blr.transfer(X, B, NEW_BE_MAPS, Y)

    ys, s2 = reference_blr.ys_s2(X.values, np.zeros((2 * n, 2), dtype=int))
    for sex in (0, 1):
        mask = B.values[:, 1] == sex
        residuals = Y.values[mask] - ys[mask]
        expected_factor = (
            np.std(residuals, ddof=1) / np.mean(np.sqrt(s2[mask])) * np.sqrt(1 + 1 / n)
        )
        offset, factor = transferred.correction_coefficients[str((0, sex))]
        assert offset == pytest.approx(np.mean(residuals))
        assert factor == pytest.approx(expected_factor)


def test_transfer_zscores_follow_student_t(reference_blr: BLR) -> None:
    """Z-scores of healthy new-site subjects follow a Student t(n - 1).

    With n = 10 adaptation subjects per cell and a constant reference
    variance, the expected share of |Z| > 1.96 is 2 * t.sf(1.96, 9) = 0.082.
    Treating the estimated offset and scale as exact gave 0.110.

    Tolerance: 300 sites x 100 test subjects. The standard error of the share
    is about 0.003 (subjects of a site share the same estimates), so 0.012 is
    about 4 standard errors and still excludes 0.110.
    """
    rng = np.random.default_rng(20261112)
    n = 10
    z_all = []
    for _ in range(300):
        Xa, Ba, Ya = _new_site(rng, reference_blr, (n, n))
        transferred = reference_blr.transfer(Xa, Ba, NEW_BE_MAPS, Ya)
        Xt, Bt, Yt = _new_site(rng, reference_blr, (50, 50))
        z_all.append(transferred.forward(Xt, Bt, Yt).values)
    z = np.concatenate(z_all)

    expected = 2 * student_t.sf(1.96, n - 1)
    assert expected == pytest.approx(0.082, abs=0.001)
    assert abs(np.mean(np.abs(z) > 1.96) - expected) < 0.012
