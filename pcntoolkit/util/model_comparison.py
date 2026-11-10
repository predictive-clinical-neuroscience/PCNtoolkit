import copy

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.hbr import HBR, MCMC
from pcntoolkit.util.output import Output, Warnings

# fit() scales and unscales data, leading to rounding errors which we should
# tolerate
SCALE_TOLERANCE = 1e-9


def compare_hbr_models(
    models: dict[str, str], data: NormData
) -> dict[str, pd.DataFrame]:
    """Compare HBR models with PSIS-LOO (``arviz.compare``).

    Each model is scored per response variable by its ELPD (expected log
    predictive density): how well it predicts unseen subjects. Higher is better.

    Parameters
    ----------
    models : dict[str, str]
        Mapping model name to the directory the model was saved in. Models
        that cannot be loaded are skipped.
    data : NormData
        The data all models were fitted on (PSIS-LOO is only valid on it).
        It is not modified.

    Returns
    -------
    dict[str, pd.DataFrame]
        The ``arviz.compare`` table for each response variable that is in
        ``data`` and in every model.

    Raises
    ------
    TypeError
        If a model is not an HBR model.
    ValueError
        If no model could be loaded, no response variable is shared, ``data``
        is not compatible with a model, or the models scale a response
        variable differently.
    """
    loaded_models: dict[str, NormativeModel] = {}
    for k, v in models.items():
        try:
            loaded_models[k] = NormativeModel.load(v)
        except Exception as e:
            print("Cannot load model at location:", v, e)
    if not loaded_models:
        raise ValueError("None of the models could be loaded.")

    for name, model in loaded_models.items():
        _check_is_hbr(name, model)

    response_vars = _shared_response_vars(loaded_models, data)
    # check_compatibility rejects response variables a model does not have
    data = data.sel({"response_vars": response_vars})

    for name, model in loaded_models.items():
        if not model.check_compatibility(data):
            raise ValueError(
                f"The data is not compatible with model '{name}'. Pass the data "
                "the models were fitted on."
            )

    # preprocess works in place, so copy first
    scaled = {}
    for name, model in loaded_models.items():
        scaled[name] = copy.deepcopy(data)
        model.preprocess(scaled[name])

    comparisons = {}
    for respvar in response_vars:
        _check_same_response_scale(scaled, respvar)
        for name, model in loaded_models.items():
            _warn_if_not_mcmc(name, model, respvar)
        traces = {
            name: _log_likelihood_tree(model, scaled[name], respvar)
            for name, model in loaded_models.items()
        }
        comparisons[respvar] = az.compare(traces)
    return comparisons


def _log_likelihood_tree(
    model: NormativeModel, scaled_data: NormData, respvar: str
) -> xr.DataTree:
    """Posterior and pointwise log-likelihood of preprocessed data."""
    X, be, _, Y, _ = model.extract_data(scaled_data.sel({"response_vars": respvar}))
    regression_model = model[respvar]
    # New PyMC model, so the regression model's own one is left untouched
    pymc_model = regression_model.likelihood.compile(X, be, regression_model.be_maps, Y)
    log_likelihood = pm.compute_log_likelihood(
        regression_model.idata,
        model=pymc_model,
        extend_inferencedata=False,
        progressbar=False,
    )
    return xr.DataTree.from_dict(
        {
            "posterior": regression_model.idata["posterior"].dataset,
            "log_likelihood": log_likelihood,
        }
    )


def _warn_if_not_mcmc(name: str, model: NormativeModel, respvar: str) -> None:
    """Warn if not fitted with MCMC: PSIS-LOO assumes exact posterior draws."""
    method = model[respvar].inference_method
    if not isinstance(method, MCMC):
        Output.warning(Warnings.MODEL_COMPARISON_NO_MCMC, model_name=name)


def _check_is_hbr(name: str, model: NormativeModel) -> None:
    """Raise if a model has a non-HBR regression model."""
    for regression_model in model.regression_models.values():
        if not isinstance(regression_model, HBR):
            raise TypeError(
                f"Model '{name}' is a {type(regression_model).__name__} model. "
                "compare_hbr_models only compares HBR models."
            )


def _shared_response_vars(
    models: dict[str, NormativeModel], data: NormData
) -> list[str]:
    """Response variables in ``data`` and in every model; warn about the rest."""
    in_data = [str(respvar) for respvar in data.response_vars.values]
    shared = [
        respvar
        for respvar in in_data
        if all(respvar in model.response_vars for model in models.values())
    ]
    skipped = set(in_data).union(
        *(model.response_vars for model in models.values())
    ) - set(shared)
    if skipped:
        Output.warning(
            Warnings.MODEL_COMPARISON_SKIPPED_RESPONSE_VARS,
            response_vars=", ".join(sorted(skipped)),
        )
    if not shared:
        raise ValueError(
            "No response variable is in the data and in every model, so there "
            "is nothing to compare."
        )
    return shared


def _check_same_response_scale(scaled: dict[str, NormData], respvar: str) -> None:
    """Raise if models scale ``respvar`` differently (beyond rounding)."""
    transformed = {
        name: model_data.Y.sel(response_vars=respvar).to_numpy()
        for name, model_data in scaled.items()
    }
    reference_name, reference_values = next(iter(transformed.items()))
    for name, values in transformed.items():
        if not np.allclose(
            values,
            reference_values,
            rtol=SCALE_TOLERANCE,
            atol=SCALE_TOLERANCE,
            equal_nan=True,
        ):
            raise ValueError(
                f"Models '{reference_name}' and '{name}' do not transform or scale "
                f"'{respvar}' identically, so their log-likelihoods are not on the "
                "same scale. Fit all models on the same data with the same "
                "outscaler and y_transform."
            )
