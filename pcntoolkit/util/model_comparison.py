import copy

import arviz as az
import pandas as pd
import pymc as pm
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.util.output import Output, Warnings


def compare_hbr_models(
    models: dict[str, str], data: NormData
) -> dict[str, pd.DataFrame]:
    """Compare HBR models by their estimated out-of-sample predictive accuracy.

    For each response variable, every model is scored by its expected log
    predictive density (ELPD): how much probability the model gives to a new
    subject's value, on average. Higher is better. The ELPD is estimated with
    Pareto-smoothed importance sampling leave-one-out cross-validation
    (PSIS-LOO, ``arviz.compare``): it approximates refitting the model with
    each subject left out, using only the posterior samples (the parameter
    values drawn during fitting).

    PSIS-LOO is only valid on the observations the posterior was fitted on:
    leaving out subject i means removing subject i's contribution from that
    posterior. Saved models contain the posterior samples but not the
    training data, so the data the models were fitted on must be passed here.
    All models are scored on the same observations, so their pointwise
    values are paired and the differences (``elpd_diff``, ``dse``) and the
    stacking weights are meaningful.

    Parameters
    ----------
    models : dict[str, str]
        Mapping from model name to the directory the model was saved in.
        Models that cannot be loaded are reported and skipped.
    data : NormData
        The data that all models were fitted on (for example ``train`` in
        ``model.fit(train)``). It is not modified.

    Returns
    -------
    dict[str, pd.DataFrame]
        For each response variable, the table returned by
        ``arviz.compare``.

    Raises
    ------
    ValueError
        If no model could be loaded, if ``data`` is not compatible with a
        model, or if the models do not transform and scale the response
        variable identically. The log-likelihoods are computed on the scale
        the models were fitted on (after ``y_transform`` and the outscaler),
        so they are only comparable when that scale is the same.
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
        if not model.check_compatibility(data):
            raise ValueError(
                f"The data is not compatible with model '{name}'. Pass the data "
                "the models were fitted on."
            )

    comparisons = {}
    for respvar in next(iter(loaded_models.values())).response_vars:
        _check_same_response_scale(loaded_models, respvar)
        for name, model in loaded_models.items():
            _warn_if_approximate_posterior(name, model, respvar)
        traces = {
            name: _log_likelihood_tree(model, data, respvar)
            for name, model in loaded_models.items()
        }
        comparisons[respvar] = az.compare(traces)
    return comparisons


def _log_likelihood_tree(
    model: NormativeModel, data: NormData, respvar: str
) -> xr.DataTree:
    """Posterior samples and pointwise log-likelihood of ``data`` for one response var.

    Parameters
    ----------
    model : NormativeModel
        A fitted normative model with an HBR regression model.
    data : NormData
        The data the model was fitted on.
    respvar : str
        The response variable.

    Returns
    -------
    xr.DataTree
        A tree with a ``posterior`` group and a ``log_likelihood`` group of
        dimensions (chain, draw, observations), as expected by
        ``arviz.compare``.
    """
    # preprocess scales the data in place
    data = copy.deepcopy(data)
    model.preprocess(data)
    X, be, _, Y, _ = model.extract_data(data.sel({"response_vars": respvar}))
    regression_model = model[respvar]
    # A new PyMC model holding the given data, so the regression model's own
    # pymc_model (which holds whatever data was predicted last) is left untouched.
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


def _warn_if_approximate_posterior(
    name: str, model: NormativeModel, respvar: str
) -> None:
    """Warn if the posterior samples of a model come from an approximation.

    PSIS-LOO assumes draws from the exact posterior. With ADVI, Pathfinder or
    Laplace draws it would need an extra correction for the approximation,
    which is not applied here.

    Parameters
    ----------
    name : str
        The model name used in the comparison.
    model : NormativeModel
        The loaded model.
    respvar : str
        The response variable.
    """
    method = getattr(model[respvar], "inference_method", "mcmc")
    # A string on dev; an InferenceMethod object once #597 is merged
    method_name = method if isinstance(method, str) else type(method).__name__
    if method_name.lower() != "mcmc":
        Output.warning(
            Warnings.MODEL_COMPARISON_APPROXIMATE_POSTERIOR,
            model_name=name,
            inference_method=method_name,
        )


def _check_same_response_scale(models: dict[str, NormativeModel], respvar: str) -> None:
    """Raise if the models do not map the response variable to the same scale.

    Parameters
    ----------
    models : dict[str, NormativeModel]
        The loaded models, by name.
    respvar : str
        The response variable.

    Raises
    ------
    ValueError
        If the models differ in ``y_transform`` or in their fitted outscaler
        for ``respvar``.
    """
    scales = {
        name: (model.y_transform, model.outscalers[respvar].to_dict())
        for name, model in models.items()
    }
    reference_name, reference_scale = next(iter(scales.items()))
    for name, scale in scales.items():
        if scale != reference_scale:
            raise ValueError(
                f"Models '{reference_name}' and '{name}' do not transform or scale "
                f"'{respvar}' identically, so their log-likelihoods are not on the "
                "same scale. Fit all models on the same data with the same "
                "outscaler and y_transform."
            )
