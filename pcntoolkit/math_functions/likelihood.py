from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import arviz as az  # type: ignore
import numpy as np
import pymc as pm  # type: ignore
import scipy.stats as stats
import xarray as xr
from scipy.stats import norm, nbinom  # for ZINB

from pcntoolkit.math_functions.basis_function import BsplineBasisFunction
from pcntoolkit.math_functions.factorize import *
from pcntoolkit.math_functions.prior import BasePrior, make_prior, prior_from_args
from pcntoolkit.math_functions.shash import S, S_inv, SHASHb, SHASHo, SHASHo2, m1m2
from pcntoolkit.util.migration import registry
from pcntoolkit.util.output import Errors, Output, Warnings


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
        Map counts to Z-space using randomized quantile residuals.

        The ZINB distribution is discrete, so each count y is actually an interval of
        probability rather than a single value: everything between F(y-1) and F(y), where
        F is the CDF -- the fraction of people scoring at or below y. 

        Because we want Z to be a distribution rather than a handful of single points --
        and y being an interval is what allows that -- we draw uniformly from the interval (=randomized quantile residuals)
        and map the result to Z through the inverse normal CDF.

        This makes ``forward`` stochastic: the same count maps to a slightly
        different Z on each call. ``backward`` remains deterministic, so centiles
        are unaffected.

        Parameters
        ----------
        Y : array_like
            Observed counts, non-negative integers.

        Returns
        -------
        ndarray
            Z-scores
        """
        mu, alpha, psi = args

        Y = kwargs.get("Y")
        Y = np.asarray(Y)

        # Y must be counts so it must be a finite and non-negative integer
        if not np.all(np.isfinite(Y)):
            raise ValueError(Output.error(Errors.ERROR_ZINB_Y_NOT_FINITE))
        if np.any(Y < 0) or np.any(Y != np.floor(Y)):
            raise ValueError(Output.error(Errors.ERROR_ZINB_Y_NOT_COUNTS))

        # Randomized uniform sample in [F(y-1), F(y)]
        Fm1 = self._cdf(Y - 1, mu, alpha, psi)  # F(y-1)
        Fy = self._cdf(Y, mu, alpha, psi)  # F(y)

        # Allow user to pass a specific number generator for reproducibility
        rng = kwargs.get("rng")
        # if user doesn't pass a generator, use the default one that is
        # random and non-reproducible
        rng = np.random.default_rng() if rng is None else rng

        U = rng.uniform(Fm1, Fy)  # randomized quantile residuals

        # Keep U strictly inside (0, 1) so norm.ppf stays finite.
        eps = np.finfo(float).eps
        U = np.clip(U, eps, 1 - eps)
        Z = norm.ppf(U)

        return Z

    def backward(self, *args, **kwargs):
        """
        Map Z-space back to counts.

        Inverts the mapping applied by ``forward``: the Z-score is turned into a
        probability between 0 and 1, and the smallest count whose CDF reaches that
        probability is returned.

        Probabilities at or below ``F(0)`` map to zero;
        above it the structural zero mass is removed and the remainder rescaled
        onto the negative binomial component before inverting it.

        Unlike ``forward`` this mapping is deterministic, so centiles derived
        from it are reproducible.

        Parameters
        ----------
        Z : array_like
            Z-scores to map back to count space.

        Returns
        -------
        Y : ndarray
            Non-negative integer-valued counts, as floats.
        """
        mu, alpha, psi = args
        Z = kwargs.get("Z")

        Z = np.asarray(Z)
        U = norm.cdf(Z)
        # broadcast U to the shape of mu to avoid shape mismatch issues
        U = np.broadcast_to(U, mu.shape)

        n, p = self._nb_params(mu, alpha)

        # Probability of 0 under the ZINB mixture
        p0 = self._cdf(0, mu, alpha, psi)

        # These are centiles rather than observed counts, so they can be floats
        Y = np.zeros_like(mu, dtype=float)

        # For probabilities beyond the point zero mass we invert the NB
        # component
        mask = U > p0
        if np.any(mask):
            # Remove the structural-zero mass (1 - psi) and rescale onto the NB
            # component, whose weight is psi: F(y) = (1 - psi) + psi * F_NB(y).
            U_nb = (U[mask] - (1 - psi[mask])) / psi[mask]
            U_nb = np.clip(U_nb, 0, 1)  # Ensure U_nb is in [0,1]

            # Centiles from the NB component
            y_nb = nbinom.ppf(U_nb, n[mask], p[mask])

            if np.isinf(y_nb).any():
                Output.warning(Warnings.ZINB_SATURATED_QUANTILE)

            Y[mask] = y_nb

        return Y

    def yhat(self, *args, **kwargs):
        mu, alpha, psi = args
        return psi * mu

    @staticmethod
    def _nb_params(mu, alpha):
        """
        Convert the ZINB parameters to the (n, p) parameterization used by scipy.

        PyMC parameterizes the negative binomial component by its mean ``mu`` and
        shape ``alpha``.
        Scipy's ``nbinom`` takes the number of successes ``n`` and the success 
        probability ``p``.

        Returns
        -------
        tuple of ndarray
            ``(n, p)`` suitable for passing to ``scipy.stats.nbinom``.
        """
        n = alpha  # the PyMC docs state ``alpha = n``
        p = alpha / (alpha + mu)
        return n, p

    @classmethod
    def _cdf(cls, y, mu, alpha, psi):
        """
        Evaluate the ZINB cumulative distribution function.

        The distribution mixes a point mass at zero with a negative binomial
        component. The structural-zero component is a point mass at 0, so its 
        CDF is 0 below zero and 1 from zero onward

            F(y) = (1 - psi) + psi * F_NB(y)   for y >= 0
            F(y) = 0                           for y < 0

        Parameters
        ----------
        y : array_like
            Count values at which to evaluate the CDF. Values below zero return 0.
        mu : array_like
            Mean of the negative binomial component, strictly positive.
        alpha : array_like
            Shape of the negative binomial component, strictly positive.
        psi : array_like
            Expected proportion of negative binomial draws, in (0, 1).

        Returns
        -------
        ndarray
            Cumulative probability at ``y``, in [0, 1], broadcast over the inputs.
        """
        n, p = cls._nb_params(mu, alpha)
        return np.where(y < 0, 0.0, (1 - psi) + psi * nbinom.cdf(y, n, p))

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

    def get_var_names(self) -> List[str]:
        return ["mu_samples", "alpha_samples", "psi_samples"]


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
