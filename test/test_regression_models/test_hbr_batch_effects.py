"""Regression tests for invalid encoded batch effects in HBR (#590)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pcntoolkit import NormData
from pcntoolkit.regression_model.hbr import HBR


@pytest.fixture
def hbr() -> HBR:
    """Load committed posterior samples without running a new sampler."""
    folder = Path(__file__).parents[1] / "resources/pretrained_hbr/model/response_var_0"
    saved = json.loads((folder / "regression_model.json").read_text())["model"]
    return HBR.from_dict(copy.deepcopy(saved), path=str(folder))


@pytest.fixture
def prediction_data() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Four observations containing both fitted site codes."""
    observations = np.arange(4)
    X = xr.DataArray(
        np.linspace(-1, 1, 4)[:, None],
        dims=("observations", "covariates"),
        coords={"observations": observations, "covariates": ["covariate_0"]},
    )
    be = xr.DataArray(
        np.array([[0], [1], [0], [1]]),
        dims=("observations", "batch_effect_dims"),
        coords={"observations": observations, "batch_effect_dims": ["site"]},
    )
    Y = xr.DataArray(
        np.ones(4), dims="observations", coords={"observations": observations}
    )
    return X, be, Y


def _predict(
    hbr: HBR,
    operation: str,
    X: xr.DataArray,
    be: xr.DataArray,
    Y: xr.DataArray,
) -> xr.DataArray:
    """Call each public prediction path with the same data."""
    if operation == "compute_yhat":
        data = NormData.from_ndarrays("predict", X.values, Y.values[:, None])
        return hbr.compute_yhat(data, "response_var_0", X, be)
    return getattr(hbr, operation)(X, be, Y)


@pytest.mark.parametrize(
    "operation", ["forward", "backward", "elemwise_logp", "compute_yhat"]
)
@pytest.mark.parametrize(
    "invalid_code", [-1, 2, -1.0, 2.0, 0.5, np.nan, np.inf, -np.inf]
)
def test_hbr_rejects_unknown_batch_effect_codes(
    hbr: HBR,
    prediction_data: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    operation: str,
    invalid_code: int | float,
) -> None:
    """Unknown codes must fail before they reach compiled array indexing."""
    X, be, Y = prediction_data
    bad = be.astype(type(invalid_code))
    bad.values[1, 0] = invalid_code

    with pytest.raises(ValueError, match="Invalid codes .* batch effect 'site'") as exc:
        _predict(hbr, operation, X, bad, Y)
    assert "Expected one of [0, 1]" in str(exc.value)


@pytest.mark.parametrize(
    "operation", ["forward", "backward", "elemwise_logp", "compute_yhat"]
)
def test_hbr_accepts_known_batch_effect_codes(
    hbr: HBR,
    prediction_data: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    operation: str,
) -> None:
    """Both fitted site codes must still produce finite predictions."""
    result = _predict(hbr, operation, *prediction_data)

    assert result.shape == (4,)
    assert np.isfinite(result.values).all()


@pytest.mark.parametrize("invalid_code", [-1, 2, 0.5, np.nan, np.inf])
def test_hbr_rejects_invalid_code_before_updating_cached_model(
    hbr: HBR,
    prediction_data: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    invalid_code: int | float,
) -> None:
    """An invalid logp request must leave the existing model data unchanged."""
    X, be, Y = prediction_data
    expected = hbr.elemwise_logp(X, be, Y)
    bad = be.astype(type(invalid_code))
    bad.values[1, 0] = invalid_code
    new_observations = X.observations.values + 10
    bad = bad.assign_coords(observations=new_observations)

    with pytest.raises(ValueError, match="Invalid codes .* batch effect 'site'"):
        hbr.elemwise_logp(
            (X + 10).assign_coords(observations=new_observations),
            bad,
            (Y + 10).assign_coords(observations=new_observations),
        )

    np.testing.assert_array_equal(
        hbr.pymc_model["site_data"].get_value(), be.values[:, 0]
    )
    np.testing.assert_array_equal(hbr.pymc_model["Y"].get_value(), Y.values)
    np.testing.assert_array_equal(
        hbr.pymc_model.coords["observations"], X.observations.values
    )
    xr.testing.assert_allclose(hbr.elemwise_logp(X, be, Y), expected)


@pytest.mark.parametrize("operation", ["create", "update"])
def test_likelihood_validates_each_batch_effect_dimension(
    hbr: HBR,
    prediction_data: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    operation: str,
) -> None:
    """An unknown code in the second batch dimension must also be rejected."""
    X, be, Y = prediction_data
    be = xr.concat([be, be], dim="batch_effect_dims").assign_coords(
        batch_effect_dims=["site", "sex"]
    )
    be_maps = {"site": {"a": 0, "b": 1}, "sex": {"female": 0, "male": 1}}
    likelihood = hbr.likelihood
    model = likelihood.compile(X, be, be_maps, Y)
    bad = be.copy(deep=True)
    bad.loc[{"batch_effect_dims": "site"}] = 1 - bad.sel(batch_effect_dims="site")
    bad.loc[{"observations": 1, "batch_effect_dims": "sex"}] = 2

    with pytest.raises(ValueError, match="Invalid codes .* batch effect 'sex'"):
        if operation == "create":
            likelihood.create_model_with_data(X, bad, be_maps, Y)
        else:
            likelihood.update_data(model, X + 10, bad, be_maps, Y + 10)

    for dimension in ("site", "sex"):
        np.testing.assert_array_equal(
            model[f"{dimension}_data"].get_value(),
            be.sel(batch_effect_dims=dimension).values,
        )
    np.testing.assert_array_equal(model["Y"].get_value(), Y.values)
