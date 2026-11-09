import os

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest

from pcntoolkit.math_functions.factorize import *


@pytest.mark.parametrize("mu, sigma", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_normal(mu, sigma):
    samples = np.random.normal(mu, sigma, 100000)
    approximation = factorize_normal(samples, 1)
    tight_approximation = factorize_normal(samples, 0.5)
    loose_approximation = factorize_normal(samples, 2)
    dens = stats.gaussian_kde(samples)
    dist = pm.Normal.dist(*approximation)
    tight_dist = pm.Normal.dist(*tight_approximation)
    loose_dist = pm.Normal.dist(*loose_approximation)
    xspace = np.linspace(-10, 10, 200)
    plt.plot(xspace, dens.pdf(xspace), label="observed")
    plt.plot(xspace, np.exp(pm.logp(dist, xspace).eval()), label="approsximation")
    plt.plot(
        xspace, np.exp(pm.logp(tight_dist, xspace).eval()), label="tight approximation"
    )
    plt.plot(
        xspace, np.exp(pm.logp(loose_dist, xspace).eval()), label="loose approximation"
    )
    plt.legend()
    plt.savefig(os.path.join("test", "output", f"normal_samples_{mu}_{sigma}.png"))
    plt.close()
    assert np.allclose(approximation, (mu, sigma), atol=0.01)


@pytest.mark.parametrize("sigma", [(1), (2), (3)])
def test_approximate_halfnormal(sigma):
    # samples = np.random.normal(0, sigma, 1000000)
    samples = stats.halfnorm.rvs(loc=0, scale=sigma, size=1000000)
    approximation = factorize_halfnormal(samples, 1)
    tight_approximation = factorize_halfnormal(samples, 0.5)
    loose_approximation = factorize_halfnormal(samples, 2)
    dens = stats.gaussian_kde(samples)
    dist = pm.HalfNormal.dist(*approximation)
    tight_dist = pm.HalfNormal.dist(*tight_approximation)
    loose_dist = pm.HalfNormal.dist(*loose_approximation)
    xspace = np.linspace(0, 10, 200)
    plt.plot(xspace, dens.pdf(xspace), label="observed")
    plt.plot(xspace, np.exp(pm.logp(dist, xspace).eval()), label="approximation")
    plt.plot(
        xspace, np.exp(pm.logp(tight_dist, xspace).eval()), label="tight approximation"
    )
    plt.plot(
        xspace, np.exp(pm.logp(loose_dist, xspace).eval()), label="loose approximation"
    )
    plt.legend()
    plt.savefig(os.path.join("test", "output", f"halfnormal_samples_{sigma}.png"))
    plt.close()
    assert np.allclose(approximation, (sigma,), atol=0.01)


@pytest.mark.parametrize("s, sigma", [(1, 1), (1, 2), (1, 0.5)])
def test_approximate_lognormal(s, sigma):
    samples = stats.lognorm.rvs(s=s, loc=0, scale=sigma, size=1000000)
    dens = stats.gaussian_kde(samples)
    approximation = factorize_lognormal(samples, 1)
    tight_approximation = factorize_lognormal(samples, 0.5)
    loose_approximation = factorize_lognormal(samples, 2)
    dist = pm.LogNormal.dist(*approximation)
    tight_dist = pm.LogNormal.dist(*tight_approximation)
    loose_dist = pm.LogNormal.dist(*loose_approximation)
    xspace = np.linspace(0, 10, 1000)
    plt.plot(xspace, dens.pdf(xspace), label="observed")
    plt.plot(xspace, np.exp(pm.logp(dist, xspace).eval()), label="approximation")
    plt.plot(
        xspace, np.exp(pm.logp(tight_dist, xspace).eval()), label="tight approximation"
    )
    plt.plot(
        xspace, np.exp(pm.logp(loose_dist, xspace).eval()), label="loose approximation"
    )
    plt.legend()
    plt.savefig(os.path.join("test", "output", f"lognormal_samples_{s}_{sigma}.png"))
    plt.close()
    assert np.allclose(approximation, (np.log(sigma), s), atol=0.01)


# @pytest.mark.parametrize("beta", [( 0.7), ( 2), ( 0.5)])
# def test_approximate_cauchy(beta):
#     samples = stats.cauchy.rvs(0, beta, size=100000)
#     samples2 = pm.draw(pm.Cauchy.dist(0, beta), draws=100000)
#     dens = stats.gaussian_kde(samples)
#     dens2 = stats.gaussian_kde(samples2)
#     approximation = factorize_cauchy(samples, 1)
#     tight_approximation = factorize_cauchy(samples, 0.5)
#     loose_approximation = factorize_cauchy(samples, 2)
#     dist = pm.Cauchy.dist(*approximation)
#     tight_dist = pm.Cauchy.dist(*tight_approximation)
#     loose_dist = pm.Cauchy.dist(*loose_approximation)
#     xspace = np.linspace(-10, 10, 1000)
#     plt.plot(xspace, dens.pdf(xspace), label="observed (scipy)")
#     plt.plot(xspace, dens2.pdf(xspace), label="observed (pm)")
#     # plt.plot(xspace, np.exp(pm.logp(dist, xspace).eval()), label="approximation")
#     # plt.plot(xspace, np.exp(pm.logp(tight_dist, xspace).eval()), label="tight approximation")
#     # plt.plot(xspace, np.exp(pm.logp(loose_dist, xspace).eval()), label="loose approximation")
#     plt.legend()
#     plt.savefig(os.path.join("test", "output", f"cauchy_samples_{beta}.png"))
#     plt.close()
#     assert np.allclose(approximation, (0, beta), atol=0.01)

# @pytest.mark.parametrize("beta, freedom", [(1, 1), (2, 2), (3, 3)])
# def test_approximate_halfcauchy(beta, freedom):
#     samples = stats.halfcauchy.rvs(beta, size=1000000)
#     approximation = factorize_halfcauchy(samples, freedom)
#     assert np.allclose(approximation, (beta * freedom,), atol=0.01)


@pytest.mark.parametrize("shape, scale", [(1, 1), (2, 2), (1, 0.5)])
def test_approximate_gamma(shape, scale):
    samples = stats.gamma.rvs(a=shape, scale=scale, size=100000)
    dens = stats.gaussian_kde(samples)
    approximation = factorize_gamma(samples, 1)
    loose_approximation = factorize_gamma(samples, 2)
    tight_approximation = factorize_gamma(samples, 0.1)
    dist = pm.Gamma.dist(*approximation)
    loose_dist = pm.Gamma.dist(*loose_approximation)
    tight_dist = pm.Gamma.dist(*tight_approximation)
    xspace = np.linspace(0, 10, 1000)
    plt.plot(xspace, dens.pdf(xspace), label="observed")
    plt.plot(xspace, np.exp(pm.logp(dist, xspace).eval()), label="approximation")
    plt.plot(
        xspace, np.exp(pm.logp(loose_dist, xspace).eval()), label="loose approximation"
    )
    plt.plot(
        xspace, np.exp(pm.logp(tight_dist, xspace).eval()), label="tight approximation"
    )
    plt.legend()
    plt.savefig(os.path.join("test", "output", f"gamma_samples_{shape}_{scale}.png"))
    plt.close()
    assert np.allclose(approximation, (shape, 1 / scale), atol=0.05)


@pytest.mark.parametrize("lower, upper", [(0, 1), (1, 2), (-1, 0.5)])
def test_approximate_uniform(lower, upper):
    samples = stats.uniform.rvs(lower, upper, size=1000000)
    dens = stats.gaussian_kde(samples)
    approximation = factorize_uniform(samples, 1)
    loose_approximation = factorize_uniform(samples, 2)
    tight_approximation = factorize_uniform(samples, 0.5)
    dist = pm.Uniform.dist(*approximation)
    loose_dist = pm.Uniform.dist(*loose_approximation)
    tight_dist = pm.Uniform.dist(*tight_approximation)
    xspace = np.linspace(0, 10, 1000)
    plt.plot(xspace, dens.pdf(xspace), label="observed")
    plt.plot(xspace, np.exp(pm.logp(dist, xspace).eval()), label="approximation")
    plt.plot(
        xspace, np.exp(pm.logp(loose_dist, xspace).eval()), label="loose approximation"
    )
    plt.plot(
        xspace, np.exp(pm.logp(tight_dist, xspace).eval()), label="tight approximation"
    )
    plt.legend()
    plt.savefig(os.path.join("test", "output", f"uniform_samples_{lower}_{upper}.png"))
    plt.close()
    assert np.allclose(approximation, (lower, upper), atol=0.01)


@pytest.mark.parametrize("lmb", [(1), (2), (0.5)])
def test_approximate_exponential(lmb):
    samples = stats.expon.rvs(scale=1 / lmb, size=1000000)
    dens = stats.gaussian_kde(samples)
    approximation = factorize_exponential(samples, 1)
    loose_approximation = factorize_exponential(samples, 2)
    tight_approximation = factorize_exponential(samples, 0.5)
    dist = pm.Exponential.dist(*approximation)
    loose_dist = pm.Exponential.dist(*loose_approximation)
    tight_dist = pm.Exponential.dist(*tight_approximation)
    xspace = np.linspace(0, 10, 1000)
    plt.plot(xspace, dens.pdf(xspace), label="observed")
    plt.plot(xspace, np.exp(pm.logp(dist, xspace).eval()), label="approximation")
    plt.plot(
        xspace, np.exp(pm.logp(loose_dist, xspace).eval()), label="loose approximation"
    )
    plt.plot(
        xspace, np.exp(pm.logp(tight_dist, xspace).eval()), label="tight approximation"
    )
    plt.legend()
    plt.savefig(os.path.join("test", "output", f"exponential_samples_{lmb}.png"))
    plt.close()
    assert np.allclose(approximation, (lmb,), atol=0.01)
