"""Sinh-Arcsinh (SHASH) Distribution Implementation Module.

This module implements the Sinh-Arcsinh (SHASH) distribution and its variants as described in
Jones and Pewsey (2009) [1]_. The SHASH distribution is a flexible distribution family that can
model skewness and kurtosis through separate parameters.

The module provides:

1. Basic SHASH transformations (S, S_inv, C)

2. SHASH distribution (base implementation)

3. SHASHo distribution (location-scale variant)

4. SHASHo2 distribution (alternative parameterization)

5. SHASHb distribution (standardized variant)


References
----------
.. [1] Jones, M. C., & Pewsey, A. (2009). Sinh-arcsinh distributions. Biometrika, 96(4), 761-780.
       https://doi.org/10.1093/biomet/asp053

Notes
-----
The implementation uses PyMC and PyTensor for probabilistic programming capabilities.
All distributions support random sampling and log-probability calculations.
"""

# Third-party imports
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import scipy.special as spp  # type: ignore
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from pymc.distributions import Continuous  # type: ignore
from pymc.pytensorf import floatX  # type: ignore
from pytensor import tensor as pt
from pytensor.gradient import grad_not_implemented
from pytensor.graph.basic import Variable
from pytensor.scalar.basic import BinaryScalarOp, upgrade_to_float
from pytensor.tensor import as_tensor_variable  # type: ignore
from pytensor.tensor.elemwise import Elemwise, scalar_elemwise
from pytensor.tensor.random.op import RandomVariable  # type: ignore


# pylint: disable=arguments-differ

# Constant that controls the accuracy of the finite difference approximation of dkv/dp
KV_GRADIENT_DP = 1e-8

# Basic shash operations
def S(x: NDArray[np.float64], e: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sinh arcsinh transformation."""
    return np.sinh(np.arcsinh(x) * d - e)


def S_inv(x: NDArray[np.float64], e: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse sinh arcsinh transformation."""
    return np.sinh((np.arcsinh(x) + e) / d)


# def K(p: NDArray[np.float64], x: float) -> NDArray[np.float64]:
#     """Bessel function of the second kind for unique values.
#     """
#     ps, idxs = np.unique(p, return_inverse=True)
#     return spp.kv(ps, x)[idxs].reshape(p.shape)


# def K(p: NDArray[np.float64], x: float) -> NDArray[np.float64]:
#     """Bessel function of the second kind for unique values."""
#     return spp.kv(p, x)


# Fast version 3: Dask parallelized (best for large arrays, multi-core)
def K(p, x, chunks=None):
    if isinstance(p, float):
        return spp.kv(p, x)
    return da.map_blocks(lambda c: spp.kv(c, x), da.from_array(p, chunks=chunks or "auto"), dtype=np.float64)


def P(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The P function as given in Jones et al."""
    frac = np.exp(1 / 4) / np.sqrt(8 * np.pi)
    K1 = K((q + 1) / 2, 1 / 4, chunks=(1000, 1000))
    K2 = K((q - 1) / 2, 1 / 4, chunks=(1000, 1000))
    a = (K1 + K2) * frac
    return a


def m(epsilon: NDArray[np.float64], delta: NDArray[np.float64], r: int) -> NDArray[np.float64]:
    """The r'th uncentered moment as given in Jones et al."""
    frac1 = 1 / np.power(2, r)
    acc = 0
    for i in range(r + 1):
        combs = spp.comb(r, i)
        flip = np.power(-1, i)
        ex = np.exp((r - 2 * i) * epsilon / delta)
        p = P((r - 2 * i) / delta)
        acc += combs * flip * ex * p
    return frac1 * acc


def m1m2(epsilon: float, delta: float) -> Tuple[float, float]:
    inv_delta = 1.0 / delta
    two_inv_delta = 2.0 * inv_delta
    p1 = P(inv_delta)
    p2 = P(two_inv_delta)
    eps_delta = epsilon / delta
    sinh_eps_delta = np.sinh(eps_delta)
    cosh_2eps_delta = np.cosh(2 * eps_delta)
    mean = sinh_eps_delta * p1
    raw_second = (cosh_2eps_delta * p2 - 1) / 2
    return mean, raw_second


class Kv(BinaryScalarOp):
    nfunc_spec = ("scipy.special.kv", 2, 1)

    @staticmethod
    def st_impl(p: Union[float, int], x: Union[float, int]) -> float:
        return spp.kve(p, x) * np.exp(-x)

    def impl(self, p: Union[float, int], x: Union[float, int]) -> float:
        return self.st_impl(p, x)

    def grad(
        self,
        inputs: Sequence[Variable[Any, Any]],
        output_gradients: Sequence[Variable[Any, Any]],
    ) -> List[Variable]:
        dp = KV_GRADIENT_DP
        (p, x) = inputs
        (gz,) = output_gradients
        # Use finite differences for derivative with respect to p
        dfdp = (kv(p + dp, x) - kv(p - dp, x)) / (2 * dp)  # type: ignore
        return [gz * dfdp, grad_not_implemented(self, 1, "x")]  # type: ignore


# Create operation instances
kv = Kv(upgrade_to_float, name="kv")  # type:ignore

##### Constants #####

CONST1 = np.exp(0.25) / np.power(8.0 * np.pi, 0.5)

CONST2 = -np.log(2 * np.pi) / 2


##### SHASH Distributions #####


class SHASHrv(RandomVariable):
    name = "shash"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("SHASH", "\\operatorname{SHASH}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        epsilon: float,
        delta: float,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        return np.sinh((np.arcsinh(rng.normal(loc=0, scale=1, size=size)) + epsilon) / delta)


shash = SHASHrv()


class SHASH(Continuous):
    rv_op = shash
    my_K = Elemwise(kv)

    @staticmethod
    @lru_cache(maxsize=128)
    def P(q: float) -> float:
        K1 = SHASH.my_K((q + 1) / 2, 0.25)
        K2 = SHASH.my_K((q - 1) / 2, 0.25)
        a: Variable[Any, Any] = (K1 + K2) * CONST1  # type: ignore
        return a  # type: ignore

    @staticmethod
    def m1(epsilon: float, delta: float) -> float:
        return np.sinh(epsilon / delta) * SHASH.P(1 / delta)

    @staticmethod
    def m2(epsilon: float, delta: float) -> float:
        return (np.cosh(2 * epsilon / delta) * SHASH.P(2 / delta) - 1) / 2

    @staticmethod
    def m1m2(epsilon: float, delta: float) -> Tuple[float, float]:
        inv_delta = 1.0 / delta
        two_inv_delta = 2.0 * inv_delta
        p1 = SHASH.P(inv_delta)
        p2 = SHASH.P(two_inv_delta)
        eps_delta = epsilon / delta
        sinh_eps_delta = np.sinh(eps_delta)
        cosh_2eps_delta = np.cosh(2 * eps_delta)
        mean = sinh_eps_delta * p1
        raw_second = (cosh_2eps_delta * p2 - 1) / 2
        return mean, raw_second

    @classmethod
    def dist(cls, epsilon: pt.TensorLike, delta: pt.TensorLike, **kwargs: Any) -> Any:
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([epsilon, delta], **kwargs)

    def logp(value: ArrayLike, epsilon: float, delta: float) -> float:  # type: ignore
        this_S = S(value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = np.log(delta) + np.log(this_C_sqr) / 2 - np.log(1 + np.square(value)) / 2
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp


class SHASHoRV(RandomVariable):
    name = "shasho"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo", "\\operatorname{SHASHo}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        s = rng.normal(size=size)
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma + mu  # type: ignore


shasho = SHASHoRV()


class SHASHo(Continuous):
    rv_op = shasho

    @classmethod
    def dist(
        cls,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        **kwargs: Any,
    ) -> Any:
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(
        value: ArrayLike,
        mu: float,
        sigma: float,
        epsilon: float,
        delta: float,  # type: ignore
    ) -> float:
        remapped_value = (value - mu) / sigma  # type: ignore
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = np.log(delta) + np.log(this_C_sqr) / 2 - np.log(1 + np.square(remapped_value)) / 2
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - np.log(sigma)


class SHASHo2RV(RandomVariable):
    name = "shasho2"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHo2", "\\operatorname{SHASHo2}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        s = rng.normal(size=size)
        sigma_d = sigma / delta  # type: ignore
        return np.sinh((np.arcsinh(s) + epsilon) / delta) * sigma_d + mu  # type: ignore


shasho2 = SHASHo2RV()


class SHASHo2(Continuous):
    rv_op = shasho2

    @classmethod
    def dist(
        cls,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        **kwargs: Any,
    ) -> Any:
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(
        value: ArrayLike,
        mu: float,
        sigma: float,
        epsilon: float,
        delta: float,  # type: ignore
    ) -> float:
        sigma_d = sigma / delta
        remapped_value = (value - mu) / sigma_d  # type: ignore
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = np.log(delta) + np.log(this_C_sqr) / 2 - np.log(1 + np.square(remapped_value)) / 2
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp - np.log(sigma_d)


class SHASHbRV(RandomVariable):
    name = "shashb"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SHASHb", "\\operatorname{SHASHb}")

    @classmethod
    def rng_fn(
        cls,
        rng: Generator,
        mu: float,
        sigma: float,
        epsilon: float,
        delta: float,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> NDArray[np.float64]:
        s = rng.normal(size=size)


        mean, raw_second = m1m2(epsilon, delta)
        out = ((np.sinh((np.arcsinh(s) + epsilon) / delta) - mean) / np.sqrt(raw_second - mean**2)) * sigma + mu  # type: ignore
        return out


shashb = SHASHbRV()


class SHASHb(Continuous):
    rv_op = shashb

    @classmethod
    def dist(
        cls,
        mu: pt.TensorLike,
        sigma: pt.TensorLike,
        epsilon: pt.TensorLike,
        delta: pt.TensorLike,
        **kwargs: Any,
    ) -> Any:
        mu = as_tensor_variable(floatX(mu))
        sigma = as_tensor_variable(floatX(sigma))
        epsilon = as_tensor_variable(floatX(epsilon))
        delta = as_tensor_variable(floatX(delta))
        return super().dist([mu, sigma, epsilon, delta], **kwargs)

    def logp(
        value: ArrayLike,
        mu: float,
        sigma: float,
        epsilon: float,
        delta: float,  # type: ignore
    ) -> float:
        mean, raw_second = SHASH.m1m2(epsilon, delta)
        var = raw_second - mean**2
        remapped_value = ((value - mu) / sigma) * np.sqrt(var) + mean  # type: ignore
        this_S = S(remapped_value, epsilon, delta)
        this_S_sqr = np.square(this_S)
        this_C_sqr = 1 + this_S_sqr
        frac2 = np.log(delta) + np.log(this_C_sqr) / 2 - np.log(1 + np.square(remapped_value)) / 2
        exp = -this_S_sqr / 2
        return CONST2 + frac2 + exp + np.log(var) / 2 - np.log(sigma)
