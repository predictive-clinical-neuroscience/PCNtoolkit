from __future__ import annotations

import inspect
import json
import logging

import pytest

from pcntoolkit.math_functions.likelihood import NormalLikelihood
from pcntoolkit.math_functions.prior import (
    CenteredRandomPrior,
    LinearPrior,
    Prior,
    RandomPrior,
    make_prior,
    prior_from_args,
)
from pcntoolkit.regression_model.hbr import (  # noqa: F401, F403
    HBR,
)
from test.fixtures.data_fixtures import *  # noqa: F401, F403
from test.fixtures.hbr_model_fixtures import *  # noqa: F401, F403
from test.fixtures.norm_data_fixtures import *  # noqa: F401, F403
from test.fixtures.path_fixtures import *

logging.basicConfig(level=logging.INFO)
"""
This file contains tests for the HBR class in the PCNtoolkit.
The tests cover:
1. Creating HBR objects from arguments
2. Converting HBR objects to dictionaries
3. Creating HBR objects from dictionaries
"""


@pytest.mark.parametrize(
    "args",
    [
        {"likelihood": "Normal", "linear_mu": False, "random_mu": False},
        {"likelihood": "Normal", "linear_mu": False, "random_mu": True},
        {
            "likelihood": "Normal",
            "linear_mu": False,
            "random_mu": True,
            "centered_mu": True,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": False,
            "random_intercept_mu": False,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "random_intercept_mu": False,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "centered_slope_mu": True,
            "random_intercept_mu": False,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "random_intercept_mu": True,
        },
        {
            "likelihood": "Normal",
            "linear_mu": True,
            "random_slope_mu": True,
            "centered_slope_mu": True,
            "random_intercept_mu": True,
            "centered_intercept_mu": True,
        },
    ],
)
def test_hbr_to_and_from_dict_and_args(sample_args, args):
    # Testing from args first
    hbr = HBR.from_args("test_name", sample_args | args)
    assert hbr.draws == sample_args.get("draws")
    assert hbr.tune == sample_args.get("tune")
    assert hbr.cores == sample_args.get("cores")
    assert isinstance(hbr.likelihood, NormalLikelihood)
    if args.get("linear_mu", False):
        assert isinstance(hbr.likelihood.mu, LinearPrior)
        if args.get("random_slope_mu", False):
            assert isinstance(hbr.likelihood.mu.slope, RandomPrior)
        else:
            assert isinstance(hbr.likelihood.mu.slope, Prior)
        if args.get("random_intercept_mu", False):
            assert isinstance(hbr.likelihood.mu.intercept, RandomPrior)
        else:
            assert isinstance(hbr.likelihood.mu.intercept, Prior)
    assert hbr.is_from_dict
    assert isinstance(hbr.likelihood.sigma, LinearPrior)

    # Testing to_dict
    hbr_dict = hbr.to_dict()
    assert hbr_dict["draws"] == sample_args.get("draws")
    assert hbr_dict["tune"] == sample_args.get("tune")
    assert hbr_dict["cores"] == sample_args.get("cores")
    assert hbr_dict["likelihood"]["name"] == "Normal"
    assert args.get("linear_mu", False) == (hbr_dict["likelihood"]["mu"]["type"] == "LinearPrior")
    if args.get("linear_mu", False):
        assert hbr_dict["likelihood"]["mu"]["type"] == "LinearPrior"
        # A centered random effect serializes under its own type name.
        expected_slope_type = "CenteredRandomPrior" if args.get("centered_slope_mu", False) else "RandomPrior"
        expected_intercept_type = "CenteredRandomPrior" if args.get("centered_intercept_mu", False) else "RandomPrior"
        assert (hbr_dict["likelihood"]["mu"]["slope"]["type"] == expected_slope_type) == args.get("random_slope_mu", False)
        assert (hbr_dict["likelihood"]["mu"]["intercept"]["type"] == expected_intercept_type) == args.get(
            "random_intercept_mu", False
        )
    assert hbr.is_from_dict
    assert hbr_dict["likelihood"]["sigma"]["type"] == "LinearPrior"

    # Testing from_dict
    del hbr
    hbr = HBR.from_dict(hbr_dict)
    assert hbr.draws == sample_args.get("draws")
    assert hbr.tune == sample_args.get("tune")
    assert hbr.cores == sample_args.get("cores")
    assert isinstance(hbr.likelihood, NormalLikelihood)
    if args.get("linear_mu", False):
        assert isinstance(hbr.likelihood.mu, LinearPrior)
        if args.get("random_slope_mu", False):
            assert isinstance(hbr.likelihood.mu.slope, RandomPrior)
        else:
            assert isinstance(hbr.likelihood.mu.slope, Prior)
        if args.get("random_intercept_mu", False):
            assert isinstance(hbr.likelihood.mu.intercept, RandomPrior)
        else:
            assert isinstance(hbr.likelihood.mu.intercept, Prior)

    assert hbr.is_from_dict


@pytest.fixture(scope="module")
def extract_data(fitted_norm_hbr_model: NormativeModel, norm_data_from_arrays: NormData):
    fitted_norm_hbr_model.saveplots = False
    fitted_norm_hbr_model.saveresults = True
    fitted_norm_hbr_model.savemodel = False
    fitted_norm_hbr_model.evaluate_model = True
    fitted_norm_hbr_model.save_dir = os.path.join(gettempdir(), "pcntoolkit_tests", "save_load_test", "hbr", "results")
    if os.path.exists(fitted_norm_hbr_model.save_dir):
        shutil.rmtree(fitted_norm_hbr_model.save_dir)
    os.makedirs(fitted_norm_hbr_model.save_dir, exist_ok=True)
    fitted_norm_hbr_model.predict(norm_data_from_arrays)
    responsevar = fitted_norm_hbr_model.response_vars[0]
    resp_model: HBR = fitted_norm_hbr_model[responsevar]  # type: ignore
    return resp_model.pymc_model, *fitted_norm_hbr_model.extract_data(norm_data_from_arrays.sel(response_vars=responsevar))[:-1]


def test_normal_fixed_prior(extract_data):
    prior: Prior = make_prior("theta", dist_name="Normal", dist_params=(0, 10))  # type: ignore
    samples = prior.compile(*extract_data)
    assert prior.name == "theta"
    assert prior.dims is None
    assert prior.dist_name == "Normal"
    assert prior.dist_params == (0, 10)
    assert tuple(samples.shape.eval()) == ()


def test_normal_fixed_prior_with_covariate_dim(extract_data):
    prior: Prior = make_prior("fixed3", dist_name="Normal", dims=("mu_covariates",))  # type: ignore
    samples = prior.compile(*extract_data)
    assert prior.name == "fixed3"
    assert prior.dims == ("mu_covariates",)
    assert prior.dist_name == "Normal"
    assert prior.dist_params == (0, 10)
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["mu_covariates"])


def test_random_prior(extract_data):
    prior: RandomPrior = make_prior(name="mu", random=True)  # type: ignore
    samples = prior.compile(*extract_data)
    assert prior.name == "mu"
    assert prior.dims is None
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_random_prior_with_covariate_dim(extract_data):
    prior: RandomPrior = make_prior("test_random3", random=True, dims=("mu_covariates",))  # type: ignore
    samples = prior.compile(*extract_data)
    assert prior.name == "test_random3"
    assert len(samples.shape.eval()) == 2
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])
    assert samples.shape.eval()[1] == len(extract_data[0].coords["mu_covariates"])


def test_centered_random_prior_compiles_centered(extract_data):
    """Test that offests are not a Deterministic (eg that offsets are not equal to 
    normalised_offets*site_scale) and that there is no normalized_offsets exists"""
    model = extract_data[0]
    prior: CenteredRandomPrior = make_prior(name="mu13", random=True, centered=True)  # type: ignore
    samples = prior.compile(*extract_data)

    free_rvs = {v.name for v in model.free_RVs}
    deterministics = {v.name for v in model.deterministics}
    for be_i in model.coords["batch_effect_dims"]:
        assert f"{be_i}_offset_mu13" in free_rvs
        assert f"{be_i}_offset_mu13" not in deterministics
        # The unit-scale draw (normalized_{be}_offset_mu13) belongs to the non-centered 
        # form only.
        assert f"normalized_{be_i}_offset_mu13" not in free_rvs

    assert samples.shape.eval()[0] == len(model.coords["observations"])


def test_random_prior_compiles_non_centered(extract_data):
    """Test that offests are Deterministic (eg that offsets are equal to 
    normalised_offets*site_scale) and that there is a normalized_offsets parameter"""
    model = extract_data[0]
    prior: RandomPrior = make_prior(name="mu14", random=True)  # type: ignore
    prior.compile(*extract_data)

    free_rvs = {v.name for v in model.free_RVs}
    deterministics = {v.name for v in model.deterministics}
    for be_i in model.coords["batch_effect_dims"]:
        assert f"normalized_{be_i}_offset_mu14" in free_rvs
        assert f"{be_i}_offset_mu14" in deterministics


def test_centered_random_prior_transfer_stays_centered():
    """
    RandomPrior.transfer hardcodes RandomPrior(...), so inheriting it would
    silently downgrade a centered prior to the non-centered parameterisation.
    """
    assert CenteredRandomPrior.transfer is not RandomPrior.transfer
    assert CenteredRandomPrior._compile is not RandomPrior._compile
    assert "CenteredRandomPrior(" in inspect.getsource(CenteredRandomPrior.transfer)


def test_linear_prior(extract_data):
    prior: LinearPrior = make_prior("test_linear1", linear=True)  # type: ignore
    samples = prior.compile(*extract_data)
    assert prior.name == "test_linear1"
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_linear_prior_with_random_slope(extract_data):
    slope: RandomPrior = make_prior("test_slope", random=True)  # type: ignore
    prior: LinearPrior = make_prior("test_linear2", linear=True, slope=slope)  # type: ignore
    samples = prior.compile(*extract_data)
    assert prior.name == "test_linear2"
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_linear_prior_with_random_intercept(extract_data):
    intercept: RandomPrior = make_prior("test_intercept", random=True)  # type: ignore
    prior: LinearPrior = make_prior("test_linear3", linear=True, intercept=intercept)  # type: ignore
    samples = prior.compile(*extract_data)
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_linear_prior_with_random_intercept_and_slope(extract_data):
    intercept: RandomPrior = make_prior("test_intercept", random=True)  # type: ignore
    slope: RandomPrior = make_prior("test_slope", random=True)  # type: ignore
    prior: LinearPrior = make_prior("test_linear4", linear=True, intercept=intercept, slope=slope)  # type: ignore
    samples = prior.compile(*extract_data)
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_priors_from_args_single(extract_data):
    my_new_prior_name = "mu2"
    prior_dict = {
        f"dist_name_{my_new_prior_name}": "Normal",
        f"dist_params_{my_new_prior_name}": (0, 1),
        f"random_{my_new_prior_name}": False,
        f"linear_{my_new_prior_name}": False,
        "intercept": None,
        "slope": None,
    }
    mu: Prior = prior_from_args(my_new_prior_name, prior_dict)  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims is None
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert isinstance(mu, Prior)
    assert tuple(samples.shape.eval()) == ()


def test_prior_from_args_single_with_covariate_dim(extract_data):
    my_new_prior_name = "mu3"
    prior_dict = {
        f"dist_name_{my_new_prior_name}": "Normal",
        f"dist_params_{my_new_prior_name}": (0, 1),
        f"random_{my_new_prior_name}": False,
        f"centered_{my_new_prior_name}": False,
        f"linear_{my_new_prior_name}": False,
        "intercept": None,
        "slope": None,
    }
    mu: Prior = prior_from_args(my_new_prior_name, prior_dict, dims=("mu_covariates",))  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims == ("mu_covariates",)
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert isinstance(mu, Prior)
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["mu_covariates"])


def test_two_priors_from_args(extract_data):
    my_new_prior_name = "mu4"
    my_new_prior_name_sigma = "sigma4"
    prior_dict = {
        f"dist_name_{my_new_prior_name}": "Normal",
        f"dist_params_{my_new_prior_name}": (0, 1),
        f"dist_name_{my_new_prior_name_sigma}": "LogNormal",
        f"dist_params_{my_new_prior_name_sigma}": (2.0,),
        f"random_{my_new_prior_name}": False,
        f"centered_{my_new_prior_name}": False,
        f"linear_{my_new_prior_name}": False,
        "intercept": None,
        "slope": None,
        f"random_{my_new_prior_name_sigma}": False,
        f"centered_{my_new_prior_name_sigma}": False,
        f"linear_{my_new_prior_name_sigma}": False,
    }
    mu: Prior = prior_from_args(my_new_prior_name, prior_dict)  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims is None
    assert mu.dist_name == "Normal"
    assert mu.dist_params == (0, 1)
    assert tuple(samples.shape.eval()) == ()

    sigma: Prior = prior_from_args(my_new_prior_name_sigma, prior_dict)  # type: ignore
    samples = sigma.compile(*extract_data)
    assert sigma.name == my_new_prior_name_sigma
    assert sigma.dims is None
    assert sigma.dist_name == "LogNormal"
    assert sigma.dist_params == (2.0,)
    assert tuple(samples.shape.eval()) == ()


def test_prior_from_args_random_with_covariate_dim(extract_data):
    my_new_prior_name = "mu7"
    prior_dict = {
        f"dist_name_mu_{my_new_prior_name}": "Normal",
        f"dist_params_mu_{my_new_prior_name}": (0, 4),
        f"dist_name_sigma_{my_new_prior_name}": "HalfNormal",
        f"dist_params_sigma_{my_new_prior_name}": (1.0,),
        f"random_{my_new_prior_name}": True,
        f"linear_{my_new_prior_name}": False,
        f"intercept_{my_new_prior_name}": None,
        f"slope_{my_new_prior_name}": None,
    }
    mu: RandomPrior = prior_from_args(my_new_prior_name, prior_dict, dims=("mu_covariates",))  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims == ("mu_covariates",)
    assert mu.mu.name == f"mu_{my_new_prior_name}"
    assert mu.mu.dims == ("mu_covariates",)
    assert isinstance(mu.mu, Prior)
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 4)
    assert mu.sigma.name == f"sigma_{my_new_prior_name}"
    assert mu.sigma.dims == ("mu_covariates",)
    assert isinstance(mu.sigma, Prior)
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)

    assert len(samples.shape.eval()) == 2
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])
    assert samples.shape.eval()[1] == len(extract_data[0].coords["mu_covariates"])


def test_prior_from_args_random(extract_data):
    my_new_prior_name = "mu8"
    prior_dict = {
        f"dist_name_{my_new_prior_name}": "Normal",
        f"dist_params_mu_{my_new_prior_name}": (0, 5),
        f"random_{my_new_prior_name}": True,
        f"linear_{my_new_prior_name}": False,
        f"intercept_{my_new_prior_name}": None,
        f"slope_{my_new_prior_name}": None,
    }
    mu: RandomPrior = prior_from_args(my_new_prior_name, prior_dict)  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims is None
    assert isinstance(mu, RandomPrior)
    assert isinstance(mu.mu, Prior)
    assert mu.mu.name == f"mu_{my_new_prior_name}"
    assert mu.mu.dims is None
    assert mu.mu.dist_name == "Normal"
    assert mu.mu.dist_params == (0, 5)
    assert isinstance(mu.sigma, Prior)
    assert mu.sigma.name == f"sigma_{my_new_prior_name}"
    assert mu.sigma.dims is None
    assert mu.sigma.dist_name == "HalfNormal"
    assert mu.sigma.dist_params == (1.0,)

    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_prior_from_args_linear(extract_data):
    my_new_prior_name = "mu9"
    prior_dict = {
        f"linear_{my_new_prior_name}": True,
        f"random_intercept_{my_new_prior_name}": True,
        f"random_slope_{my_new_prior_name}": False,
    }
    mu: LinearPrior = prior_from_args(my_new_prior_name, prior_dict)  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomPrior)
    assert mu.intercept.name == f"intercept_{my_new_prior_name}"
    assert mu.intercept.dims is None
    assert isinstance(mu.slope, Prior)
    assert mu.slope.name == f"slope_{my_new_prior_name}"
    assert mu.slope.dims == (f"{my_new_prior_name}_covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)
    assert len(samples.shape.eval()) == 1
    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_prior_from_args_linear_with_random_slope(extract_data):
    my_new_prior_name = "mu10"
    prior_dict = {
        f"linear_{my_new_prior_name}": True,
        f"random_intercept_{my_new_prior_name}": True,
        f"random_slope_{my_new_prior_name}": True,
    }
    mu: LinearPrior = prior_from_args(my_new_prior_name, prior_dict)  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomPrior)
    assert isinstance(mu.slope, RandomPrior)
    assert mu.intercept.name == f"intercept_{my_new_prior_name}"
    assert mu.intercept.dims is None
    assert mu.slope.name == f"slope_{my_new_prior_name}"
    assert mu.slope.dims == (f"{my_new_prior_name}_covariates",)
    assert isinstance(mu.slope.mu, Prior)
    assert mu.slope.mu.name == f"mu_slope_{my_new_prior_name}"
    assert mu.slope.mu.dims == (f"{my_new_prior_name}_covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, Prior)
    assert mu.slope.sigma.name == f"sigma_slope_{my_new_prior_name}"
    assert mu.slope.sigma.dims == (f"{my_new_prior_name}_covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)

    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_prior_from_args_linear_with_random_intercept(extract_data):
    my_new_prior_name = "mu11"
    prior_dict = {
        f"linear_{my_new_prior_name}": True,
        f"random_intercept_{my_new_prior_name}": True,
        f"random_slope_{my_new_prior_name}": False,
    }
    mu: LinearPrior = prior_from_args(my_new_prior_name, prior_dict)  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims is None
    assert isinstance(mu.slope, Prior)
    assert mu.slope.name == f"slope_{my_new_prior_name}"
    assert mu.slope.dims == (f"{my_new_prior_name}_covariates",)
    assert mu.slope.dist_name == "Normal"
    assert mu.slope.dist_params == (0, 1)

    assert isinstance(mu.intercept, RandomPrior)
    assert isinstance(mu.intercept.mu, Prior)
    assert mu.intercept.mu.name == f"mu_intercept_{my_new_prior_name}"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, Prior)
    assert mu.intercept.sigma.name == f"sigma_intercept_{my_new_prior_name}"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)

    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_prior_from_args_linear_with_random_intercept_and_slope(extract_data):
    my_new_prior_name = "mu12"
    prior_dict = {
        f"linear_{my_new_prior_name}": True,
        f"random_intercept_{my_new_prior_name}": True,
        f"random_slope_{my_new_prior_name}": True,
    }
    mu: LinearPrior = prior_from_args(my_new_prior_name, prior_dict)  # type: ignore
    samples = mu.compile(*extract_data)
    assert mu.name == my_new_prior_name
    assert mu.dims is None
    assert isinstance(mu.intercept, RandomPrior)
    assert mu.intercept.name == f"intercept_{my_new_prior_name}"
    assert mu.intercept.dims is None
    assert isinstance(mu.intercept.mu, Prior)
    assert mu.intercept.mu.name == f"mu_intercept_{my_new_prior_name}"
    assert mu.intercept.mu.dims is None
    assert mu.intercept.mu.dist_name == "Normal"
    assert mu.intercept.mu.dist_params == (0, 1)
    assert isinstance(mu.intercept.sigma, Prior)
    assert mu.intercept.sigma.name == f"sigma_intercept_{my_new_prior_name}"
    assert mu.intercept.sigma.dims is None
    assert mu.intercept.sigma.dist_name == "HalfNormal"
    assert mu.intercept.sigma.dist_params == (1.0,)
    assert isinstance(mu.slope, RandomPrior)
    assert mu.slope.name == f"slope_{my_new_prior_name}"
    assert mu.slope.dims == (f"{my_new_prior_name}_covariates",)
    assert isinstance(mu.slope.mu, Prior)
    assert mu.slope.mu.name == f"mu_slope_{my_new_prior_name}"
    assert mu.slope.mu.dims == (f"{my_new_prior_name}_covariates",)
    assert mu.slope.mu.dist_name == "Normal"
    assert mu.slope.mu.dist_params == (0, 1)
    assert isinstance(mu.slope.sigma, Prior)
    assert mu.slope.sigma.name == f"sigma_slope_{my_new_prior_name}"
    assert mu.slope.sigma.dims == (f"{my_new_prior_name}_covariates",)
    assert mu.slope.sigma.dist_name == "HalfNormal"
    assert mu.slope.sigma.dist_params == (1.0,)

    assert samples.shape.eval()[0] == len(extract_data[0].coords["observations"])


def test_to_dict(extract_data):
    prior_dict = {
        "linear_mu": True,
        "random_intercept_mu": True,
        "centered_intercept_mu": True,
        "random_slope_mu": True,
        "centered_slope_mu": True,
    }
    mu = prior_from_args("mu", prior_dict)
    my_dict = mu.to_dict()
    assert my_dict["name"] == "mu"
    assert my_dict["intercept"]["name"] == "intercept_mu"
    assert my_dict["intercept"]["mu"]["name"] == "mu_intercept_mu"
    assert my_dict["intercept"]["mu"]["dist_name"] == "Normal"
    assert my_dict["intercept"]["mu"]["dist_params"] == (0, 1)
    assert my_dict["intercept"]["sigma"]["name"] == "sigma_intercept_mu"
    assert my_dict["intercept"]["sigma"]["dist_name"] == "HalfNormal"
    assert my_dict["intercept"]["sigma"]["dist_params"] == (1.0,)
    assert my_dict["slope"]["name"] == "slope_mu"
    assert my_dict["slope"]["dims"] == ("covariates",)

    # Check if json.dumps does not throw an error
    json.dumps(my_dict)
