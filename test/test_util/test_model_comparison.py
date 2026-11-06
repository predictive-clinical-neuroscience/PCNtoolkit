"""
Tests for compare_hbr_models.

PSIS-LOO estimates how well a model predicts new subjects from its posterior
samples, by removing each fitted subject in turn. It is only valid on the
observations the posterior was fitted on. compare_hbr_models used to score
each model on a few points synthesized from that same model, which ranks
models by how narrow their predictive distribution is rather than by how well
they fit the data. These tests check that every model is scored on the fit
data, that a model whose predictive distribution is far too narrow for the
data is ranked last, and that the result does not depend on the global random
state.
"""

import copy

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.likelihood import NormalLikelihood
from pcntoolkit.math_functions.prior import make_prior
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.hbr import HBR
from pcntoolkit.util.model_comparison import (
    _log_likelihood_tree,
    compare_hbr_models,
)

N_OBS = 120
DRAWS = 100
TUNE = 100
CHAINS = 2
RESPVAR = "y"


def make_data() -> NormData:
    """Linear effect of age, a site offset and unit residual noise."""
    rng = np.random.default_rng(7)
    age = rng.uniform(20, 80, N_OBS)
    site = rng.choice(["A", "B"], N_OBS)
    y = 0.05 * age + 0.5 * (site == "B") + rng.normal(0.0, 1.0, N_OBS)
    df = pd.DataFrame(
        {"age": age, "site": site, "y": y, "sub": [f"s{i}" for i in range(N_OBS)]}
    )
    return NormData.from_dataframe(
        "toy",
        df,
        covariates=["age"],
        batch_effects=["site"],
        response_vars=["y"],
        subject_ids="sub",
    )


def make_model(sigma, save_dir: str) -> NormativeModel:
    mu = make_prior(
        linear=True,
        intercept=make_prior(random=True),
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
    )
    hbr = HBR(
        "template",
        likelihood=NormalLikelihood(mu=mu, sigma=sigma),
        nuts_sampler="nutpie",
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=CHAINS,
        progressbar=False,
    )
    return NormativeModel(
        hbr,
        savemodel=True,
        evaluate_model=False,
        saveresults=False,
        saveplots=False,
        save_dir=save_dir,
        inscaler="standardize",
        outscaler="standardize",
    )


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Two models fitted on the same data and saved to disk.

    "free" estimates the residual standard deviation. "too_narrow" forces it
    into [0.045, 0.05] on the standardized scale, about sixteen times smaller
    than the true value (about 0.8): on its own synthetic draws it would look
    better than "free", on the data it is far worse.
    """
    data = make_data()
    sigmas = {
        "free": make_prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
            mapping="softplus",
            mapping_params=(0.0, 3.0),
        ),
        "too_narrow": make_prior(dist_name="Uniform", dist_params=(0.045, 0.05)),
    }
    paths, in_memory = {}, {}
    for name, sigma in sigmas.items():
        path = str(tmp_path_factory.mktemp(name))
        model = make_model(sigma, path)
        model.fit(data.copy(deep=True))
        paths[name] = path
        in_memory[name] = model
    return data, paths, in_memory


def test_log_likelihood_is_evaluated_on_fit_data(fitted):
    data, paths, in_memory = fitted
    loaded = NormativeModel.load(paths["free"])
    tree = _log_likelihood_tree(loaded, data, RESPVAR)
    loglik = tree["log_likelihood"]["Yhat"]
    assert loglik.sizes["observations"] == N_OBS
    assert loglik.sizes["chain"] == CHAINS
    assert loglik.sizes["draw"] == DRAWS

    # Reference: the PyMC model built during fit, which holds the fit data.
    reference_model = in_memory["free"][RESPVAR]
    reference = pm.compute_log_likelihood(
        reference_model.idata,
        model=reference_model.pymc_model,
        extend_inferencedata=False,
        progressbar=False,
    )
    np.testing.assert_allclose(loglik.values, reference["Yhat"].values, rtol=1e-10)


def test_too_narrow_model_is_ranked_last(fitted):
    data, paths, _ = fitted
    table = compare_hbr_models(paths, data)[RESPVAR]
    assert list(table.index) == ["free", "too_narrow"]
    assert table.loc["too_narrow", "elpd_diff"] < -5 * table.loc["too_narrow", "dse"]
    # Stacking weights are not model probabilities: "too_narrow" keeps a small
    # weight because it predicts the few subjects closest to the mean better.
    assert table.loc["free", "weight"] > table.loc["too_narrow", "weight"]


def test_result_does_not_depend_on_global_random_state(fitted):
    data, paths, _ = fitted
    np.random.seed(0)
    first = compare_hbr_models(paths, data)[RESPVAR]
    np.random.seed(1)
    second = compare_hbr_models(paths, data)[RESPVAR]
    pd.testing.assert_frame_equal(first, second)


def test_data_is_not_modified(fitted):
    data, paths, _ = fitted
    before = copy.deepcopy(data)
    compare_hbr_models(paths, data)
    np.testing.assert_array_equal(data.Y.values, before.Y.values)
    np.testing.assert_array_equal(data.X.values, before.X.values)


def test_models_on_different_scales_are_rejected(fitted, monkeypatch):
    data, paths, _ = fitted
    original_load = NormativeModel.load

    def load_with_other_outscaler(path):
        model = original_load(path)
        if path == paths["too_narrow"]:
            model.outscalers[RESPVAR].s = model.outscalers[RESPVAR].s * 2
        return model

    monkeypatch.setattr(NormativeModel, "load", load_with_other_outscaler)
    with pytest.raises(ValueError, match="identically"):
        compare_hbr_models(paths, data)


def test_approximate_posterior_triggers_a_warning(fitted, monkeypatch):
    data, paths, _ = fitted
    original_load = NormativeModel.load

    def load_as_advi(path):
        model = original_load(path)
        if path == paths["too_narrow"]:
            model[RESPVAR].inference_method = "advi"
        return model

    monkeypatch.setattr(NormativeModel, "load", load_as_advi)
    with pytest.warns(UserWarning, match="inference_method='advi'"):
        compare_hbr_models(paths, data)
