"""Tests for the likelihood functions.

Likelihoods map response variables to Z-space (``forward``) and back
(``backward``). This file covers that mapping, plus serialization, for each
likelihood.

"""

import json

import numpy as np
import pytest
import scipy.stats as stats

from pcntoolkit.math_functions.likelihood import (
    Likelihood,
    ZeroInflatedNegativeBinomialLikelihood,
)
from pcntoolkit.math_functions.prior import make_prior
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.hbr import HBR

# Parameters of the ZINB used throughout: a mean of 5 counts, a shape of 2, and
# 70% of the draws coming from the negative binomial rather than the point mass
# at zero.
MU, ALPHA, PSI = 5.0, 2.0, 0.7
N_OBS = 20


def assert_serialization_roundtrip(
    likelihood: Likelihood, name: str, var_names: list[str]
) -> None:
    """Check that a likelihood survives a to_dict/from_dict roundtrip.

    Models are saved as json, so the dict must also be json-serializable.
    Shared by all likelihoods.

    Parameters
    ----------
    likelihood : Likelihood
        The likelihood to roundtrip.
    name : str
        The name the likelihood is registered under, e.g. "ZINB".
    var_names : list[str]
        The variable names the likelihood is expected to expose.
    """
    dct = likelihood.to_dict()
    assert dct["name"] == name
    json.dumps(dct)

    loaded = Likelihood.from_dict(dct)
    assert isinstance(loaded, type(likelihood))
    assert loaded.name == name
    assert loaded.get_var_names() == var_names


# --------------------------------------------------------------------------- #
# ZINB
# --------------------------------------------------------------------------- #


def zinb_likelihood() -> ZeroInflatedNegativeBinomialLikelihood:
    """Build a ZINB likelihood.

    The priors are placeholders: forward and backward take parameter arrays
    directly, so the priors they would have been sampled from are irrelevant here.
    """
    return ZeroInflatedNegativeBinomialLikelihood(
        make_prior("mu"), make_prior("alpha"), make_prior("psi")
    )


def zinb_params(n: int = N_OBS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (mu, alpha, psi) as ``(observations, sample)`` arrays.

    This is the shape HBR hands the likelihood, one column per posterior sample.
    """
    return (
        np.full((n, 1), MU),
        np.full((n, 1), ALPHA),
        np.full((n, 1), PSI),
    )


def test_zinb_forward_backward_roundtrip():
    """Counts must survive a forward/backward roundtrip exactly.

    ZINB is discrete, so unlike the continuous likelihoods the roundtrip is
    exact rather than approximate.
    """
    likelihood = zinb_likelihood()
    params = zinb_params()
    Y = np.arange(N_OBS, dtype=float).reshape(-1, 1)

    Z = likelihood.forward(*params, Y=Y, rng=np.random.default_rng(42))
    Y_prime = likelihood.backward(*params, Z=Z)

    assert np.array_equal(Y_prime, Y)


def test_zinb_forward_is_reproducible_with_rng():
    """forward draws a random quantile, so it only repeats when given a seed."""
    likelihood = zinb_likelihood()
    params = zinb_params()
    Y = np.arange(N_OBS, dtype=float).reshape(-1, 1)

    Z_seeded = likelihood.forward(*params, Y=Y, rng=np.random.default_rng(1))
    Z_same_seed = likelihood.forward(*params, Y=Y, rng=np.random.default_rng(1))
    Z_unseeded = likelihood.forward(*params, Y=Y)

    assert np.array_equal(Z_seeded, Z_same_seed)
    assert not np.allclose(Z_seeded, Z_unseeded)


def test_zinb_forward_is_monotone_in_y():
    """Higher counts must map to higher Z-scores."""
    likelihood = zinb_likelihood()
    Y = np.arange(N_OBS, dtype=float).reshape(-1, 1)

    Z = likelihood.forward(*zinb_params(), Y=Y, rng=np.random.default_rng(0))

    assert np.all(np.diff(Z.ravel()) >= 0)
    assert np.all(np.isfinite(Z))


def test_zinb_backward_is_monotone_in_z():
    """Higher Z-scores must map back to higher counts."""
    likelihood = zinb_likelihood()
    Z = np.linspace(-3, 3, N_OBS).reshape(-1, 1)

    Y = likelihood.backward(*zinb_params(), Z=Z)

    assert np.all(np.diff(Y.ravel()) >= 0)


@pytest.mark.parametrize(
    "invalid_value, match",
    [
        (-1.0, "non-negative integer counts"),
        (2.5, "non-negative integer counts"),
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_zinb_forward_rejects_invalid_y(invalid_value, match):
    """Y must be finite non-negative integers, since ZINB models counts."""
    likelihood = zinb_likelihood()
    Y = np.array([[1.0], [invalid_value]])

    with pytest.raises(ValueError, match=match):
        likelihood.forward(*zinb_params(n=2), Y=Y)


def test_zinb_backward_returns_zero_below_point_mass():
    """Z-scores below the zero-inflated point mass all map to a count of zero."""
    likelihood = zinb_likelihood()
    params = zinb_params()

    # Everything at or below F(0) falls in the point mass at zero.
    z_at_point_mass = stats.norm.ppf(likelihood._cdf(0, *params))

    below = likelihood.backward(*params, Z=z_at_point_mass - 0.1)
    above = likelihood.backward(*params, Z=z_at_point_mass + 0.5)

    assert np.all(below == 0.0)
    assert np.all(above > 0.0)


def test_zinb_backward_warns_on_saturated_quantile():
    """Z-scores too extreme to have a finite count warn and return inf."""
    likelihood = zinb_likelihood()

    with pytest.warns(UserWarning, match="no finite count exists"):
        Y = likelihood.backward(*zinb_params(), Z=np.full((N_OBS, 1), 9.0))

    assert np.all(np.isinf(Y))


def test_zinb_cdf_matches_nbinom_without_zero_inflation():
    """With psi=1 there is no zero inflation, so the CDF is a plain nbinom."""
    likelihood = zinb_likelihood()
    mu, alpha, _ = zinb_params()
    no_inflation = np.ones((N_OBS, 1))
    y = np.arange(N_OBS).reshape(-1, 1)

    n, p = likelihood._nb_params(mu, alpha)

    assert np.allclose(
        likelihood._cdf(y, mu, alpha, no_inflation), stats.nbinom.cdf(y, n, p)
    )


def test_zinb_cdf_is_zero_below_zero():
    """Counts cannot be negative, so no probability mass sits below zero."""
    likelihood = zinb_likelihood()

    assert np.all(likelihood._cdf(np.array([[-1.0]]), *zinb_params(n=1)) == 0.0)


def test_zinb_yhat_accounts_for_zero_inflation():
    """The expected count is the nbinom mean scaled by the non-inflated share."""
    likelihood = zinb_likelihood()

    yhat = likelihood.yhat(*zinb_params())

    assert np.allclose(yhat, PSI * MU)


def test_zinb_serialization_roundtrip():
    assert_serialization_roundtrip(
        zinb_likelihood(),
        name="ZINB",
        var_names=["mu_samples", "alpha_samples", "psi_samples"],
    )


def test_zinb_from_args():
    """A ZINB can be built from a plain args dict, as the CLI does."""
    likelihood = Likelihood.from_args({"likelihood": "ZINB"})

    assert isinstance(likelihood, ZeroInflatedNegativeBinomialLikelihood)
    assert likelihood.mu.name == "mu"
    assert likelihood.alpha.name == "alpha"
    assert likelihood.psi.name == "psi"


@pytest.mark.parametrize("outscaler", ["standardize", "minmax", "robminmax"])
def test_zinb_rejects_scaling_outscaler(outscaler):
    """Scaling Y would leave it non-integer, which a ZINB cannot model."""
    with pytest.raises(ValueError, match="outscaler"):
        NormativeModel(
            HBR("test_zinb", likelihood=zinb_likelihood()), outscaler=outscaler
        )


@pytest.mark.parametrize("y_transform", ["log", "log1p"])
def test_zinb_rejects_y_transform(y_transform):
    """Transforming Y would leave it non-integer, which a ZINB cannot model."""
    with pytest.raises(ValueError, match="y_transform"):
        NormativeModel(
            HBR("test_zinb", likelihood=zinb_likelihood()),
            outscaler="none",
            y_transform=y_transform,
        )


@pytest.mark.parametrize("outscaler", ["none", "id"])
def test_zinb_accepts_unscaled_outscaler(outscaler):
    """Both spellings of "do not scale" must be accepted."""
    model = NormativeModel(
        HBR("test_zinb", likelihood=zinb_likelihood()), outscaler=outscaler
    )

    assert model.outscaler == outscaler


def test_non_count_likelihood_may_scale_y():
    """Only count likelihoods are restricted; the rest still scale Y by default."""
    model = NormativeModel(HBR("test_normal"), outscaler="standardize")

    assert model.outscaler == "standardize"
