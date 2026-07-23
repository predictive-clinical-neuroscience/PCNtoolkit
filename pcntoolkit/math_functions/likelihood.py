from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import arviz as az  # type: ignore
import numpy as np
import pymc as pm  # type: ignore
import scipy.stats as stats
import xarray as xr
from scipy.stats import norm, nbinom # for ZINB

from pcntoolkit.math_functions.basis_function import BsplineBasisFunction
from pcntoolkit.math_functions.factorize import *
from pcntoolkit.math_functions.prior import BasePrior, make_prior, prior_from_args
from pcntoolkit.math_functions.shash import S, S_inv, SHASHb, SHASHo, SHASHo2, m1m2
from pcntoolkit.util.migration import registry


class Likelihood(ABC):
    def __init__(self, name: str):
        self.name = name

    def compile(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> pm.Model:
        model = self.create_model_with_data(X, be, be_maps, Y)
        self._compile(model, X, be, be_maps, Y)
        return model

    def create_model_with_data(self, X, be, be_maps, Y) -> pm.Model:
        coords = {"batch_effect_dims": be.coords["batch_effect_dims"].values, "observations": X.coords["observations"].values}
        for _be, _map in be_maps.items():
            coords[_be] = [k for k in sorted(_map.keys(), key=(lambda v: _map[v]))]

        model = pm.Model(coords=coords)
        with model:
            for be_name in be.coords["batch_effect_dims"].values:
                pm.Data(
                    f"{be_name}_data",
                    be.sel(batch_effect_dims=be_name).values,
                    dims=("observations",),
                )
            pm.Data("Y", Y.values, dims=("observations",))
        return model

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        with model:
            model.set_data(name="Y", values=Y.values, coords={"observations": Y.coords["observations"].values})
            for be_name in be.coords["batch_effect_dims"].values:
                model.set_data(
                    name=f"{be_name}_data",
                    values=be.sel(batch_effect_dims=be_name).values,
                )
        self._update_data(model, X, be, be_maps, Y)

    @abstractmethod
    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        pass

    @abstractmethod
    def transfer(self, idata: az.InferenceData, **kwargs) -> "Likelihood":
        pass

    @abstractmethod
    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        pass

    @abstractmethod
    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def yhat(self, *args, **kwargs):
        pass

    @staticmethod
    def from_dict(
        dct: Dict[str, Any], version: str | None = None
    ) -> "Likelihood":
        # Apply any registered Likelihood migrations for this version.
        dct = registry.migrate("Likelihood", dct, version=version)
        likelihood = dct.pop("name", "Normal")
        match likelihood:
            case "Normal":
                return NormalLikelihood._from_dict(dct, version=version)
            case "SHASHb":
                return SHASHbLikelihood._from_dict(dct, version=version)
            # case "SHASHo":
            #     return SHASHoLikelihood._from_dict(dct, version=version)
            # case "SHASHo2":
            #     return SHASHo2Likelihood._from_dict(dct, version=version)
            case "beta":
                return BetaLikelihood._from_dict(dct, version=version)
            case"ZINB":
                return ZeroInflatedNegativeBinomialLikelihood._from_dict(dct, version=version)
            case _:
                raise ValueError(f"Unknown likelihood: {likelihood}")

    @staticmethod
    def from_args(args: Dict[str, Any]) -> "Likelihood":
        likelihood = args.pop("likelihood", "Normal")
        match likelihood:
            case "Normal":
                return NormalLikelihood._from_args(args)
            case "SHASHb":
                return SHASHbLikelihood._from_args(args)
            # case "SHASHo":
            #     return SHASHoLikelihood._from_args(args)
            # case "SHASHo2":
            #     return SHASHo2Likelihood._from_args(args)
            case "beta":
                return BetaLikelihood._from_args(args)
            case "ZINB":
                return ZeroInflatedNegativeBinomialLikelihood._from_args(args)
            case _:
                raise ValueError(f"Unknown likelihood: {likelihood}")

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def _from_dict(
        cls,
        dct: Dict[str, Any],
        version: str | None = None,
    ) -> "Likelihood":
        pass

    @classmethod
    @abstractmethod
    def _from_args(cls, args: Dict[str, Any]) -> "Likelihood":
        pass

    @abstractmethod
    def has_random_effect(self) -> bool:
        pass


class NormalLikelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior):
        super().__init__(name="Normal")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        compiled_params = self.compile_params(model, X, be, be_maps, Y)
        compiled_params = {k: v[0] for k, v in compiled_params.items()}
        with model:
            pm.Normal("Yhat", **compiled_params, observed=model["Y"], dims="observations")
        return model

    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        return {
            "mu": (self.mu.compile(model, X, be, be_maps, Y), self.mu.sample_dims),
            "sigma": (self.sigma.compile(model, X, be, be_maps, Y), self.sigma.sample_dims),
        }

    def transfer(self, idata: az.InferenceData, **kwargs) -> "Likelihood":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_sigma = self.sigma.transfer(idata, **kwargs)
        return NormalLikelihood(new_mu, new_sigma)

    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        self.mu.update_data(model, X, be, be_maps, Y)
        self.sigma.update_data(model, X, be, be_maps, Y)

    def forward(self, *args, **kwargs):
        mu, sigma = args
        Y = kwargs.get("Y", None)
        return (Y - mu) / sigma

    def backward(self, *args, **kwargs):
        mu, sigma = args
        Z = kwargs.get("Z")
        return Z * sigma + mu

    def yhat(self, *args, **kwargs):
        mu, _ = args
        return mu

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mu": self.mu.to_dict(), "sigma": self.sigma.to_dict()}

    @classmethod
    def _from_dict(
        cls,
        dct: Dict[str, Any],
        version: str | None = None,
    ) -> "NormalLikelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"], version=version),
            sigma=BasePrior.from_dict(dct["sigma"], version=version),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "NormalLikelihood":
        return cls(mu=prior_from_args("mu", args), sigma=prior_from_args("sigma", args))

    def has_random_effect(self) -> bool:
        return self.mu.has_random_effect or self.sigma.has_random_effect


class SHASHbLikelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior, epsilon: BasePrior, delta: BasePrior):
        super().__init__(name="SHASHb")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")
        self.epsilon = epsilon
        self.epsilon.set_name("epsilon")
        self.delta = delta
        self.delta.set_name("delta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        compiled_params = self.compile_params(model, X, be, be_maps, Y)
        compiled_params = {k: v[0] for k, v in compiled_params.items()}
        with model:
            SHASHb("Yhat", **compiled_params, observed=model["Y"], dims="observations")
        return model

    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        return {
            "mu": (self.mu.compile(model, X, be, be_maps, Y), self.mu.sample_dims),
            "sigma": (self.sigma.compile(model, X, be, be_maps, Y), self.sigma.sample_dims),
            "epsilon": (self.epsilon.compile(model, X, be, be_maps, Y), self.epsilon.sample_dims),
            "delta": (self.delta.compile(model, X, be, be_maps, Y), self.delta.sample_dims),
        }

    def transfer(self, idata: az.InferenceData, **kwargs) -> "SHASHbLikelihood":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_sigma = self.sigma.transfer(idata, **kwargs)
        new_epsilon = self.epsilon.transfer(idata, **kwargs)
        new_delta = self.delta.transfer(idata, **kwargs)
        return SHASHbLikelihood(new_mu, new_sigma, new_epsilon, new_delta)

    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        self.mu.update_data(model, X, be, be_maps, Y)
        self.sigma.update_data(model, X, be, be_maps, Y)
        self.epsilon.update_data(model, X, be, be_maps, Y)
        self.delta.update_data(model, X, be, be_maps, Y)

    def has_random_effect(self) -> bool:
        return (
            self.mu.has_random_effect
            or self.sigma.has_random_effect
            or self.epsilon.has_random_effect
            or self.delta.has_random_effect
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mu": self.mu.to_dict(),
            "sigma": self.sigma.to_dict(),
            "epsilon": self.epsilon.to_dict(),
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def _from_dict(
        cls,
        dct: Dict[str, Any],
        version: str | None = None,
    ) -> "SHASHbLikelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"], version=version),
            sigma=BasePrior.from_dict(dct["sigma"], version=version),
            epsilon=BasePrior.from_dict(dct["epsilon"], version=version),
            delta=BasePrior.from_dict(dct["delta"], version=version),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "SHASHbLikelihood":
        return cls(
            mu=prior_from_args("mu", args),
            sigma=prior_from_args("sigma", args),
            epsilon=prior_from_args("epsilon", args),
            delta=prior_from_args("delta", args),
        )

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples", "epsilon_samples", "delta_samples"]

    def forward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        Y = kwargs.get("Y", None)
        m1, m2 = m1m2(epsilon, delta)
        true_mu = m1
        true_sigma = np.sqrt(m2 - true_mu**2)
        SHASH_centered = (Y - mu) / sigma
        SHASH_uncentered = SHASH_centered * true_sigma + true_mu
        Z = S(SHASH_uncentered, epsilon, delta)
        return Z

    def backward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        Z = kwargs.get("Z", None)
        m1, m2 = m1m2(epsilon, delta)
        true_mu = m1
        true_sigma = np.sqrt(m2 - true_mu**2)
        SHASH_uncentered = S_inv(Z, epsilon, delta)
        SHASH_centered = (SHASH_uncentered - true_mu) / true_sigma
        Y = SHASH_centered * sigma + mu
        return Y

    def yhat(self, *args, **kwargs):
        mu, _, _, _ = args
        return mu


class SHASHoLikelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior, epsilon: BasePrior, delta: BasePrior):
        super().__init__(name="SHASHo")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")
        self.epsilon = epsilon
        self.epsilon.set_name("epsilon")
        self.delta = delta
        self.delta.set_name("delta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        with model:
            mu_samples = self.mu.compile(model, X, be, be_maps, Y)
            sigma_samples = self.sigma.compile(model, X, be, be_maps, Y)
            epsilon_samples = self.epsilon.compile(model, X, be, be_maps, Y)
            delta_samples = self.delta.compile(model, X, be, be_maps, Y)
            mu_samples = pm.Deterministic("mu_samples", mu_samples, dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", sigma_samples, dims=self.sigma.sample_dims)
            epsilon_samples = pm.Deterministic("epsilon_samples", epsilon_samples, dims=self.epsilon.sample_dims)
            delta_samples = pm.Deterministic("delta_samples", delta_samples, dims=self.delta.sample_dims)
            SHASHo(
                "Yhat",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=model["Y"],
                dims="observations",
            )
        return model

    def has_random_effect(self) -> bool:
        return (
            self.mu.has_random_effect
            or self.sigma.has_random_effect
            or self.epsilon.has_random_effect
            or self.delta.has_random_effect
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mu": self.mu.to_dict(),
            "sigma": self.sigma.to_dict(),
            "epsilon": self.epsilon.to_dict(),
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def _from_dict(
        cls,
        dct: Dict[str, Any],
        version: str | None = None,
    ) -> "SHASHoLikelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"], version=version),
            sigma=BasePrior.from_dict(dct["sigma"], version=version),
            epsilon=BasePrior.from_dict(dct["epsilon"], version=version),
            delta=BasePrior.from_dict(dct["delta"], version=version),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "SHASHoLikelihood":
        return cls(
            mu=prior_from_args("mu", args),
            sigma=prior_from_args("sigma", args),
            epsilon=prior_from_args("epsilon", args),
            delta=prior_from_args("delta", args),
        )

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples", "epsilon_samples", "delta_samples"]

    def forward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        y = kwargs.get("Y", None)
        SHASH = (y - mu) / sigma
        Z = np.sinh(np.arcsinh(SHASH) * delta - epsilon)
        return Z

    def backward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        Z = kwargs.get("Z", None)
        SHASH = S_inv(Z, epsilon, delta)
        Y = SHASH * sigma + mu
        return Y


class SHASHo2Likelihood(Likelihood):
    def __init__(self, mu: BasePrior, sigma: BasePrior, epsilon: BasePrior, delta: BasePrior):
        super().__init__(name="SHASHo2")
        self.mu = mu
        self.mu.set_name("mu")
        self.sigma = sigma
        self.sigma.set_name("sigma")
        self.epsilon = epsilon
        self.epsilon.set_name("epsilon")
        self.delta = delta
        self.delta.set_name("delta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        with model:
            mu_samples = self.mu.compile(model, X, be, be_maps, Y)
            sigma_samples = self.sigma.compile(model, X, be, be_maps, Y)
            epsilon_samples = self.epsilon.compile(model, X, be, be_maps, Y)
            delta_samples = self.delta.compile(model, X, be, be_maps, Y)
            mu_samples = pm.Deterministic("mu_samples", mu_samples, dims=self.mu.sample_dims)
            sigma_samples = pm.Deterministic("sigma_samples", sigma_samples, dims=self.sigma.sample_dims)
            epsilon_samples = pm.Deterministic("epsilon_samples", epsilon_samples, dims=self.epsilon.sample_dims)
            delta_samples = pm.Deterministic("delta_samples", delta_samples, dims=self.delta.sample_dims)
            SHASHo2(
                "Yhat",
                mu=mu_samples,
                sigma=sigma_samples,
                epsilon=epsilon_samples,
                delta=delta_samples,
                observed=model["Y"],
                dims="observations",
            )
        return model

    def has_random_effect(self) -> bool:
        return (
            self.mu.has_random_effect
            or self.sigma.has_random_effect
            or self.epsilon.has_random_effect
            or self.delta.has_random_effect
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mu": self.mu.to_dict(),
            "sigma": self.sigma.to_dict(),
            "epsilon": self.epsilon.to_dict(),
            "delta": self.delta.to_dict(),
        }

    @classmethod
    def _from_dict(
        cls,
        dct: Dict[str, Any],
        version: str | None = None,
    ) -> "SHASHo2Likelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"], version=version),
            sigma=BasePrior.from_dict(dct["sigma"], version=version),
            epsilon=BasePrior.from_dict(dct["epsilon"], version=version),
            delta=BasePrior.from_dict(dct["delta"], version=version),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "SHASHo2Likelihood":
        return cls(
            mu=prior_from_args("mu", args),
            sigma=prior_from_args("sigma", args),
            epsilon=prior_from_args("epsilon", args),
            delta=prior_from_args("delta", args),
        )

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "sigma_samples", "epsilon_samples", "delta_samples"]

    def forward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        sigma_d = sigma / delta
        Y = kwargs.get("Y", None)
        SHASH = (Y - mu) / sigma_d
        Z = S(SHASH, epsilon, delta)
        return Z

    def backward(self, *args, **kwargs):
        mu, sigma, epsilon, delta = args
        sigma_d = sigma / delta
        Z = kwargs.get("Z", None)
        SHASH = S_inv(Z, epsilon, delta)
        Y = SHASH * sigma_d + mu
        return Y


class BetaLikelihood(Likelihood):
    def __init__(self, alpha: BasePrior, beta: BasePrior):
        super().__init__(name="beta")
        self.alpha = alpha
        self.alpha.set_name("alpha")
        self.beta = beta
        self.beta.set_name("beta")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        with model:
            compiled_params = self.compile_params(model, X, be, be_maps, Y)
            compiled_params = {k: v[0] for k, v in compiled_params.items()}
            pm.Beta(
                "Yhat",
                **compiled_params,
                observed=model["Y"],
                dims="observations",
            )
        return model

    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        return {
            "alpha": (self.alpha.compile(model, X, be, be_maps, Y), self.alpha.sample_dims),
            "beta": (self.beta.compile(model, X, be, be_maps, Y), self.beta.sample_dims),
        }

    def transfer(self, idata: az.InferenceData, **kwargs) -> "BetaLikelihood":
        new_alpha = self.alpha.transfer(idata, **kwargs)
        new_beta = self.beta.transfer(idata, **kwargs)
        return BetaLikelihood(new_alpha, new_beta)

    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        self.alpha.update_data(model, X, be, be_maps, Y)
        self.beta.update_data(model, X, be, be_maps, Y)

    def forward(self, *args, **kwargs):
        alpha, beta = args
        Y = kwargs.get("Y", None)
        cdf = stats.beta.cdf(Y, alpha, beta)
        Z = stats.norm.ppf(cdf)
        return Z

    def backward(self, *args, **kwargs):
        alpha, beta = args
        Z = kwargs.get("Z", None)
        cdf_norm = stats.norm.cdf(Z)
        quantiles = stats.beta.ppf(cdf_norm, alpha, beta)
        return quantiles

    def yhat(self, *args, **kwargs):
        alpha, beta = args
        return alpha / (alpha + beta)

    def has_random_effect(self) -> bool:
        return self.alpha.has_random_effect or self.beta.has_random_effect

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "alpha": self.alpha.to_dict(), "beta": self.beta.to_dict()}

    @classmethod
    def _from_dict(
        cls,
        dct: Dict[str, Any],
        version: str | None = None,
    ) -> "BetaLikelihood":
        return cls(
            alpha=BasePrior.from_dict(dct["alpha"], version=version),
            beta=BasePrior.from_dict(dct["beta"], version=version),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "BetaLikelihood":
        return cls(alpha=prior_from_args("alpha", args), beta=prior_from_args("beta", args))

    def get_var_names(self) -> List[str]:
        return ["alpha_samples", "beta_samples"]

class ZeroInflatedNegativeBinomialLikelihood(Likelihood):
    def __init__(self, mu: BasePrior, alpha: BasePrior, psi: BasePrior):
        super().__init__(name="ZINB")
        self.mu = mu
        self.mu.set_name("mu")
        self.alpha = alpha
        self.alpha.set_name("alpha")
        self.psi = psi
        self.psi.set_name("psi")

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> pm.Model:
        compiled_params = self.compile_params(model, X, be, be_maps, Y)
        compiled_params = {k: v[0] for k, v in compiled_params.items()}
        with model:
            pm.ZeroInflatedNegativeBinomial("Yhat", **compiled_params, observed=model["Y"], dims="observations")
        return model

    def compile_params(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> dict[str, Any]:
        return {
            "mu": (self.mu.compile(model, X, be, be_maps, Y), self.mu.sample_dims),
            "alpha": (self.alpha.compile(model, X, be, be_maps, Y), self.alpha.sample_dims),
            "psi": (self.psi.compile(model, X, be, be_maps, Y), self.psi.sample_dims),
        }

    def transfer(self, idata: az.InferenceData, **kwargs) -> "Likelihood":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_alpha = self.alpha.transfer(idata, **kwargs)
        new_psi = self.psi.transfer(idata, **kwargs)
        return ZeroInflatedNegativeBinomialLikelihood(new_mu, new_alpha, new_psi)

    def _update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        self.mu.update_data(model, X, be, be_maps, Y)
        self.alpha.update_data(model, X, be, be_maps, Y)
        self.psi.update_data(model, X, be, be_maps, Y)
    
    def forward(self, *args, **kwargs):
        
        """
        ZINB is discrete and mixed, so mapping to Gaussian Z space could work like this

        1. a point mass at zero with probability psi
        2. a negative binomial count distribution with mean mu and dispersion alpha with probability 1-psi for the non-zero part
        
        So we cannot do one-to-one mapping to Z space, we use randomized quantile residuals to map to Z-space
        1. compute CDF interval for the observed Y value, which is [F(Y-1), F(Y)]
        2. sample a uniform random value in that interval
        3. apply the inverse standard normal CDF
        
        This should yield Z that is approximately N(0,1) but is semi-randomized
        """
        mu, alpha, psi = args
        Y = kwargs.get("Y")
        
        Y = np.asarray(Y)
        
        # NB parameterization
        # If Var(Y_NB) = mu + alpha * mu^2, then the number of failures r = 1 / alpha and the success probability p = r / (r + mu)
        r = 1 / alpha
        p = r / (r + mu)
        
        # randomized CDF just below Y
        Fm1 = np.where (Y > 0, psi + (1 - psi) * nbinom.cdf(Y - 1, r, p), 0.0)
        
        # CDF at Y
        Fy = psi + (1 - psi) * nbinom.cdf(Y, r, p)
        
        # Randomized uniform sample in [Fm1, Fy]
        U = np.random.uniform(Fm1, Fy)
        
        # Ran into nan/inf during testing: avoid infinities in the CDF by clipping U to (0,1)
        eps = np.finfo(float).eps
        U = np.clip(U, eps, 1 - eps)
        
        # Map to Z space
        Z = norm.ppf(U)
        
        return Z

    def backward(self, *args, **kwargs):
        """
        Map Z-space back to Y-space
        
        1. convert Gaussian Z to uniform U using standard normal CDF
        2. invert the ZINB CDF using the quantile function to get Y
        
        Since ZINB is discrete the inverse could be obtained by finding the smallest integer Y such that F(Y) >= U
        """
        
        mu, alpha, psi = args
        Z = kwargs.get("Z")
        
        print("mu:", type(mu), np.shape(mu))
        print("alpha:", type(alpha), np.shape(alpha))
        print("psi:", type(psi), np.shape(psi))
        print("Z:", type(Z), np.shape(Z))
        
        Z = np.asarray(Z)
        U = norm.cdf(Z)
        
        r = 1 / alpha
        p = r / (r + mu)
        
        # Probability of 0 under the ZINB mixture
        p0 = psi + (1 - psi) * nbinom.pmf(0, r, p)
        
        # Z turns out to have shape (9600,1) which centile calc cannot deal with. So we initialize Y with the same shape as mu and not U (which I did previously)
        Y = np.zeros_like(mu, dtype=int)
        
        # For probabilities beyond the point zero mass we invert the NB component
        mask = U > p0
        if np.any(mask):
            # Remove the point mass at zero and scale U to the NB CDF
            U_nb = (U[mask] - psi) / (1 - psi)
            U_nb = np.clip(U_nb, 0, 1)  # Ensure U_nb is in [0,1]
            
            # Quantile from the NB component
            y_nb = nbinom.ppf(U_nb, r, p).astype(int)
            # Print shape for debugging
            print("y_nb:", y_nb.shape)
            
            # Ensure positive counts and assign to Y
            Y[mask] = np.maximum(y_nb, 1)
        
        return Y

    def yhat(self, *args, **kwargs):
        mu, alpha, psi = args
        return (1-psi) * mu

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mu": self.mu.to_dict(), "alpha": self.alpha.to_dict(), "psi": self.psi.to_dict()}

    @classmethod
    def _from_dict(
        cls,
        dct: Dict[str, Any],
        version: str | None = None,
    ) -> "ZeroInflatedNegativeBinomialLikelihood":
        return cls(
            mu=BasePrior.from_dict(dct["mu"], version=version),
            alpha=BasePrior.from_dict(dct["alpha"], version=version),
            psi=BasePrior.from_dict(dct["psi"], version=version),
        )

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "ZeroInflatedNegativeBinomialLikelihood":
        return cls(
            mu=prior_from_args("mu", args),
            alpha=prior_from_args("alpha", args),
            psi=prior_from_args("psi", args),
        )

    def has_random_effect(self) -> bool:
        return self.mu.has_random_effect or self.alpha.has_random_effect or self.psi.has_random_effect


def get_default_normal_likelihood() -> NormalLikelihood:
    # Random effect in mu, and also bsplines for mu and sigma
    likelihood = NormalLikelihood(
            mu=make_prior(
                # Mu is linear because we want to allow the mean to vary as a function of the covariates.
                linear=True,
                # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
                slope=make_prior(dist_params=(0.0, 3.0)),
                # The intercept is random, because we expect the intercept to vary between sites and sexes.
                intercept=make_prior(
                    random=True,
                    # Mu is the mean of the intercept, which is  distributed with a mean of 0 and a standard deviation of 1.
                    mu=make_prior(dist_params=(0, 1)),
                    # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we sample from a Gamma distribution
                    sigma=make_prior(
                        dist_name="Gamma",
                        dist_params=(1, 0.5),
                    ),
                ),
                basis_function=BsplineBasisFunction(),
            ),
            sigma=make_prior(
                # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
                linear=True,
                # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
                slope=make_prior(dist_params=(0.0, 2.0)),
                # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
                intercept=make_prior(dist_params=(0.0, 1.0)),
                # We use a softplus mapping to ensure that sigma is strictly positive.
                mapping="softplus",
                # We scale the softplus mapping by a factor of 2, to avoid spikes in the resulting density.
                # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
                # This basically provides an affine transformation of the softplus function.
                # a -> horizontal shift
                # b -> scaling
                # c -> vertical shift
                # You can leave c out, and it will default to 0.
                mapping_params=(0, 2),
                # We use a B-spline basis function to allow for non-linearity in the standard deviation.
                basis_function=BsplineBasisFunction(),
            ),
        )

    # mu = make_prior(
    #     linear=True,
    #     slope=make_prior(dist_params=(0, 2)),
    #     intercept=make_prior(
    #         random=True,
    #         mu=make_prior(dist_params=(0.0, 1.0)),
    #         sigma=make_prior(
    #             dist_params=(1.0, 1.0),
    #             mapping="softplus",
    #             mapping_params=(0.0, 2.0),
    #         ),
    #     ),
    #     # We use a B-spline basis function to allow for non-linearity in the mean.
    #     basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    # )
    # sigma = make_prior(
    #     linear=True,
    #     slope=make_prior(dist_params=(0.0, 2.0)),
    #     intercept=make_prior(dist_params=(1.0, 2.0)),
    #     basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    #     mapping="softplus",
    #     mapping_params=(0.0, 2.0),
    # )

    # # Set the likelihood with the priors we just created.
    # likelihood = NormalLikelihood(mu, sigma)

    return likelihood
