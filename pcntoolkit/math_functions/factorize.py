
import numpy as np
import scipy.stats as stats
from numpy.typing import ArrayLike


def factorize_normal(samples, freedom=1) -> tuple[float, float]:
    mu, sigma = stats.norm.fit(samples)
    return (mu, sigma * freedom)


def factorize_halfnormal(samples: ArrayLike, freedom: float = 1) -> tuple[float]:
    _, scale = stats.halfnorm.fit(samples, floc=0)
    return (scale * freedom,)


def factorize_lognormal(samples, freedom=1) -> tuple[float, float]:
    s, _, scale = stats.lognorm.fit(samples, floc=0)
    return (np.log(scale), s * freedom)


# def factorize_cauchy(samples, freedom=1) -> tuple[float, float]:
#     alpha, beta = stats.cauchy.fit(samples)
#     return (alpha,  freedom/beta)


# def factorize_halfcauchy(samples, freedom=1) -> tuple[float]:
#     a, beta = stats.halfcauchy.fit(samples)
#     return (beta * freedom,)


def factorize_gamma(samples, freedom=1) -> tuple[float, float]:
    a, loc, beta = stats.gamma.fit(samples, floc=0)
    return (a / freedom, 1 / (beta * freedom))


# def factorize_invgamma(samples, freedom=1) -> tuple[float, float]:
#     a, loc, beta = stats.invgamma.fit(samples, floc=0)
#     return (a, beta*freedom)


def factorize_uniform(samples: ArrayLike, freedom: float = 1) -> tuple[float, float]:
    # scipy returns (loc, scale), PyMC expects (lower, upper)
    loc, scale = stats.uniform.fit(samples)
    return (loc, loc + scale)


def factorize_exponential(samples, freedom=1) -> tuple[float]:
    _, scale = stats.expon.fit(samples, floc=0)
    return (freedom / scale,)
