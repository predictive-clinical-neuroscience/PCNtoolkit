from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az  # type: ignore
import numpy as np
import pymc as pm  # type: ignore
import xarray as xr
from pymc import math

from pcntoolkit.math_functions.basis_function import BasisFunction, LinearBasisFunction
from pcntoolkit.math_functions.factorize import *
from pcntoolkit.util.output import Errors, Output
from pcntoolkit.util.migration import registry

PM_DISTMAP = {
    "Normal": pm.Normal,
    # "Cauchy": pm.Cauchy,
    "HalfNormal": pm.HalfNormal,
    # "HalfCauchy": pm.HalfCauchy,
    "Uniform": pm.Uniform,
    "Gamma": pm.Gamma,
    # "InvGamma": pm.InverseGamma,
    "LogNormal": pm.LogNormal,
}

DEFAULT_PRIOR_ARGS = {
    # For all models
    "linear_mu": True,
    "random_intercept_mu": True,
    "slope_mu_dist": "Normal",
    "slope_mu_params": (0, 10.0),
    "mu_intercept_mu_dist": "Normal",
    "mu_intercept_mu_params": (0, 10.0),
    "sigma_intercept_mu_dist": "HalfNormal",
    "sigma_intercept_mu_params": (2.0,),
    "linear_sigma": True,
    "mapping_sigma": "softplus",
    "mapping_params_sigma": (0, 2.0),
    "slope_sigma_dist": "Normal",
    "slope_sigma_params": (0, 10.0),
    "intercept_sigma_dist": "HalfNormal",
    "intercept_sigma_params": (2.0,),
    # For SHASH models
    "linear_epsilon": False,
    "random_epsilon": False,
    "epsilon_dist": "Normal",
    "epsilon_params": (0, 1.0),
    "linear_delta": False,
    "random_delta": False,
    "delta_dist": "Normal",
    "delta_params": (1.0, 2.0),
    "mapping_delta": "softplus",
    "mapping_params_delta": (0, 3.0, 0.3),
}


def make_prior(name: str = "theta", **kwargs) -> BasePrior:
    kwargs["name"] = name
    if kwargs.pop("linear", False):
        return LinearPrior(**kwargs)
    elif kwargs.pop("random", False):
        return RandomPrior(**kwargs)
    else:
        return Prior(**kwargs)


def prior_from_args(name: str, args: Dict[str, Any], dims: Optional[Union[Tuple[str, ...], str]] = None) -> BasePrior:
    my_args = DEFAULT_PRIOR_ARGS | args
    mapping = my_args.get(f"mapping_{name}", "identity")
    mapping_params = my_args.get(f"mapping_params_{name}", (0, 1))
    if name.split("_")[0] in ["sigma", "delta"]:
        dist_name = my_args.get(f"dist_name_{name}", "HalfNormal")
        dist_params = my_args.get(f"dist_params_{name}", (1.0,))
        if mapping == "identity":
            if dist_name in ["Normal", "Cauchy"] or (dist_name == "Uniform" and dist_params[0] <= 0):
                raise ValueError(Output.error(Errors.ENSURE_POSITIVE_DISTRIBUTION, name=name))
    else:
        dist_name = my_args.get(f"dist_name_{name}", "Normal")
        dist_params = my_args.get(f"dist_params_{name}", (0, 1))

    dims = my_args.get(f"dims_{name}", dims)
    if my_args.get(f"linear_{name}", False):
        slope = prior_from_args(f"slope_{name}", my_args, dims=dims)
        intercept = prior_from_args(f"intercept_{name}", my_args, dims=dims)
        basis_function = BasisFunction.from_args(f"basis_function_{name}", my_args)
        return LinearPrior(
            slope=slope,
            intercept=intercept,
            name=name,
            dims=dims,
            mapping=mapping,
            mapping_params=mapping_params,
            basis_function=basis_function,
        )
    elif my_args.get(f"random_{name}", False):
        mu = prior_from_args(f"mu_{name}", my_args, dims=dims)
        sigma = prior_from_args(f"sigma_{name}", my_args, dims=dims)
        return RandomPrior(mu=mu, sigma=sigma, name=name, dims=dims, mapping=mapping, mapping_params=mapping_params)
    else:
        return Prior(
            name=name, dims=dims, mapping=mapping, mapping_params=mapping_params, dist_name=dist_name, dist_params=dist_params
        )


class BasePrior(ABC):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple = None,  # type: ignore
        **kwargs,
    ):
        self.name = name
        self._dims = dims
        has_covariate_dim = False if not self.dims else "covariates" in self.dims
        if self.name.startswith("slope") and not has_covariate_dim:
            if self.dims is None:
                self.dims = ("covariates",)
            else:
                self.dims = (dims, "covariates")
        self.mapping = mapping
        self.mapping_params = mapping_params or (0, 1)
        self.sample_dims = ()

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        self._dims = value

    def apply_mapping(self, x: Any) -> Any:
        a, b = self.mapping_params[0], self.mapping_params[1]
        if self.mapping == "identity":
            toreturn = x
        elif self.mapping == "exp":
            toreturn = math.exp((x - a) / b) * b
        elif self.mapping == "softplus":
            toreturn = math.log(1 + math.exp((x - a) / b)) * b  # type: ignore
        else:
            raise ValueError(Output.error(Errors.ERROR_UNKNOWN_MAPPING, mapping=self.mapping))
        if len(self.mapping_params) > 2:
            toreturn = toreturn + self.mapping_params[2]
        return toreturn

    def compile(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ) -> Any:
        samples = self._compile(model, X, be, be_maps, Y)
        return self.apply_mapping(samples)

    @abstractmethod
    def transfer(self, idata: az.InferenceData, **kwargs) -> "BasePrior":
        pass

    @abstractmethod
    def _compile(self, model, X, be, be_maps, Y) -> Any:
        pass

    @abstractmethod
    def update_data(self, model, X, be, be_maps, Y):
        pass

    @property
    @abstractmethod
    def has_random_effect(self) -> bool:
        pass

    @abstractmethod
    def set_name(self, name: str) -> None:
        pass

    def to_dict(self):
        dct = copy.deepcopy(self.__dict__)
        dct["dims"] = self.dims
        del dct["sample_dims"]
        return dct | {"type": self.__class__.__name__}

    @classmethod
    def from_dict(
        cls, dict: dict, version: str | None = None
    ) -> "BasePrior":
        # Apply any registered Prior migrations for this version
        dict = registry.migrate("Prior", dict, version=version)
        return globals()[dict.pop("type")].from_dict(
            dict, version=version
        )

    def __eq__(self, other: BasePrior):
        return self.to_dict() == other.to_dict()


class Prior(BasePrior):
    def __init__(
        self,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = None,  # type: ignore
        dist_name: str = "Normal",
        dist_params: Tuple[float | int | list[float | int], ...] = None,  # type: ignore
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.dist_name = dist_name
        self.dist_params = dist_params or (0, 10.0)
        self.sample_dims = ()

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ):
        with model:
            self.dist = PM_DISTMAP[self.dist_name](self.name, *self.dist_params, dims=self.dims)
        return self.dist

    def transfer(self, idata: az.InferenceData, **kwargs) -> "Prior":
        new_prior = Prior(self.name, self.dims, self.mapping, self.mapping_params, self.dist_name, self.dist_params)
        freedom = kwargs.get("freedom", 1)

        def infer_params(s):
            if self.dist_name == "Normal":
                return factorize_normal(s, freedom)
            elif self.dist_name == "HalfNormal":
                return factorize_halfnormal(s, freedom)
            elif self.dist_name == "LogNormal":
                return factorize_lognormal(s, freedom)
            elif self.dist_name == "Uniform":
                return factorize_uniform(s, freedom)
            elif self.dist_name == "Gamma":
                return factorize_gamma(s, freedom)
            else:
                raise ValueError(Output.error(Errors.ERROR_UNKNOWN_DISTRIBUTION, dist_name=self.dist_name))

        samples = az.extract(idata, var_names=self.name)
        covariate_dims = [i for i in samples.dims if i.endswith("covariates")]
        if len(covariate_dims) == 1:
            params = [infer_params(samples.sel(**{covariate_dims[0]: i})) for i in samples.coords[covariate_dims[0]]]
            new_prior.dist_params = [i.tolist() for i in np.array(params).T]
        elif len(covariate_dims) == 0:
            new_prior.dist_params = infer_params(samples)
        else:
            raise ValueError(Output.error(Errors.ERROR_MULTIPLE_COVARIATE_DIMS, covariate_dims=covariate_dims))
        return new_prior

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        pass

    @property
    def has_random_effect(self):
        return False

    def to_dict(self):
        dct = super().to_dict()
        dct.pop("dist", None)
        dct["type"] = self.__class__.__name__
        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(
            **{
                k: v
                for k, v in dct.items()
                if k
                in [
                    "name",
                    "dims",
                    "mapping",
                    "mapping_params",
                    "dist_name",
                    "dist_params",
                ]
            }
        )

    def set_name(self, name: str) -> None:
        self.name = name


class RandomPrior(BasePrior):
    def __init__(
        self,
        mu: Optional[BasePrior] = None,
        sigma: Optional[BasePrior] = None,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = None,  # type: ignore
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.mu = mu or make_prior(dist_name="Normal", dist_params=(0, 2.0))
        self.sigma = sigma or make_prior(
            dist_name="Normal", dist_params=(1.0, 1.0), mapping="softplus", mapping_params=(0.0, 1.0)
        )
        self.sigmas = {}
        self.offsets = {}
        self.scaled_offsets = {}
        self.sample_dims = ("observations",)
        self.set_name(self.name)

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        if hasattr(self, "mu"):
            self.mu.dims = value
        if hasattr(self, "sigma"):
            self.sigma.dims = value
        self._dims = value

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ):
        # outdims = "observations" if not self.dims else ("observations", *self.dims)
        with model:
            mu_samples = self.mu.compile(model, X, be, be_maps, Y)
            if self.dims:
                acc = mu_samples[None]  # type: ignore
            else:
                acc = mu_samples  # type: ignore
            for be_i in model.coords["batch_effect_dims"]:  # type:ignore
                be_dims = (be_i,) if not self.dims else ( *self.dims, be_i)
                if be_i not in self.sigmas:
                    self.sigmas[be_i] = copy.deepcopy(self.sigma)
                    self.sigmas[be_i].set_name(f"{be_i}_sigma_{self.name}")
                sig = self.sigmas[be_i].compile(model, X, be, be_maps, Y)
                # norm = pm.Normal(f"normalized_{be_i}_offset_{self.name}", dims=be_dims,mu=0,sigma=1)
                norm = pm.ZeroSumNormal(f"normalized_{be_i}_offset_{self.name}", dims=be_dims, sigma=1).T
                self.scaled_offsets[be_i] = pm.Deterministic(f"{be_i}_offset_{self.name}", sig * norm)
                indices = model[f"{be_i}_data"]
                accumulant = self.scaled_offsets[be_i][indices]
                acc += accumulant
            self.dist = acc
        return self.dist

    def transfer(self, idata: az.InferenceData, **kwargs) -> "RandomPrior":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_sigma = copy.deepcopy(self.sigma)
        new_prior = RandomPrior(
            name=self.name, dims=self.dims, mapping=self.mapping, mapping_params=self.mapping_params, mu=new_mu, sigma=new_sigma
        )
        for be_i in self.sigmas.keys():
            new_prior.sigmas[be_i] = self.sigmas[be_i].transfer(idata, **kwargs)
        return new_prior

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        pass

    def set_name(self, name: str):
        self.name = name
        self.mu.set_name(f"mu_{self.name}")
        self.sigma.set_name(f"sigma_{self.name}")

    @property
    def has_random_effect(self):
        return True

    def to_dict(self):
        dct = super().to_dict()
        dct["mu"] = self.mu.to_dict()
        dct["sigma"] = self.sigma.to_dict()
        if hasattr(self, "sigmas"):
            for k, v in self.sigmas.items():
                dct[f"{k}_sigma"] = v.to_dict()

        for thing in ["sigmas", "offsets", "scaled_offsets", "dist"]:
            if hasattr(self, thing):
                del dct[thing]
        return dct

    @classmethod
    def from_dict(
        cls, dict: dict, version: str | None = None
    ) -> "RandomPrior":
        mu = BasePrior.from_dict(dict["mu"], version=version)
        sigma = BasePrior.from_dict(dict["sigma"], version=version)
        instance = cls(
            mu=mu,
            sigma=sigma,
            **{k: v for k, v in dict.items() if k in ["name", "dims", "mapping", "mapping_params"]},
        )
        instance.sigmas = {
            k.split("_")[0]: BasePrior.from_dict(v, version=version)
            for k, v in dict.items()
            if k.endswith("_sigma")
        }
        # instance.scaled_offsets = {k: Param.from_dict(v) for k, v in dict.items() if k.endswith("_offset")}
        return instance


class CenteredRandomPrior(BasePrior):
    def __init__(
        self,
        mu: Optional[BasePrior] = None,
        sigma: Optional[BasePrior] = None,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = None,  # type: ignore
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.mu = mu or make_prior(dist_name="Normal", dist_params=(0, 2.0))
        self.sigma = sigma or make_prior(
            dist_name="Normal", dist_params=(1.0, 1.0), mapping="softplus", mapping_params=(0.0, 1.0)
        )
        self.sigmas = {}
        self.offsets = {}
        self.scaled_offsets = {}
        self.sample_dims = ("observations",)
        self.set_name(self.name)

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        if hasattr(self, "mu"):
            self.mu.dims = value
        if hasattr(self, "sigma"):
            self.sigma.dims = value
        self._dims = value

    def _compile(
        self,
        model: pm.Model,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ):
        # outdims = "observations" if not self.dims else ("observations", *self.dims)
        with model:
            mu_samples = self.mu.compile(model, X, be, be_maps, Y)
            if self.dims:
                acc = mu_samples[None]  # type: ignore
            else:
                acc = mu_samples  # type: ignore
            for be_i in model.coords["batch_effect_dims"]:  # type:ignore
                be_dims = (be_i,) if not self.dims else ( *self.dims,be_i)
                if be_i not in self.sigmas:
                    self.sigmas[be_i] = copy.deepcopy(self.sigma)
                    self.sigmas[be_i].set_name(f"{be_i}_sigma_{self.name}")
                self.scaled_offsets[be_i] = pm.ZeroSumNormal(
                    f"{be_i}_offset_{self.name}", sigma=self.sigmas[be_i].compile(model, X, be, be_maps, Y), dims=be_dims
                )
                acc += self.scaled_offsets[be_i][...,model[f"{be_i}_data"]]
            self.dist = acc
        return self.dist

    def transfer(self, idata: az.InferenceData, **kwargs) -> "RandomPrior":
        new_mu = self.mu.transfer(idata, **kwargs)
        new_sigma = copy.deepcopy(self.sigma)
        new_prior = RandomPrior(
            name=self.name, dims=self.dims, mapping=self.mapping, mapping_params=self.mapping_params, mu=new_mu, sigma=new_sigma
        )
        for be_i in self.sigmas.keys():
            new_prior.sigmas[be_i] = self.sigmas[be_i].transfer(idata, **kwargs)
        return new_prior

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        pass

    def set_name(self, name: str):
        self.name = name
        self.mu.set_name(f"mu_{self.name}")
        self.sigma.set_name(f"sigma_{self.name}")

    @property
    def has_random_effect(self):
        return True

    def to_dict(self):
        dct = super().to_dict()
        dct["mu"] = self.mu.to_dict()
        dct["sigma"] = self.sigma.to_dict()
        if hasattr(self, "sigmas"):
            for k, v in self.sigmas.items():
                dct[f"{k}_sigma"] = v.to_dict()

        for thing in ["sigmas", "offsets", "scaled_offsets", "dist"]:
            if hasattr(self, thing):
                del dct[thing]
        return dct

    @classmethod
    def from_dict(
        cls, dict: dict, version: str | None = None
    ) -> "CenteredRandomPrior":
        mu = BasePrior.from_dict(dict["mu"], version=version)
        sigma = BasePrior.from_dict(dict["sigma"], version=version)
        instance = cls(
            mu=mu,
            sigma=sigma,
            **{k: v for k, v in dict.items() if k in ["name", "dims", "mapping", "mapping_params"]},
        )
        instance.sigmas = {
            k.split("_")[0]: BasePrior.from_dict(v, version=version)
            for k, v in dict.items()
            if k.endswith("_sigma")
        }
        # instance.scaled_offsets = {k: Param.from_dict(v) for k, v in dict.items() if k.endswith("_offset")}
        return instance


class LinearPrior(BasePrior):
    def __init__(
        self,
        slope: Optional[BasePrior] = None,
        intercept: Optional[BasePrior] = None,
        name: str = "theta",
        dims: Optional[Union[Tuple[str, ...], str]] = None,
        mapping: str = "identity",
        mapping_params: tuple[float, ...] = None,  # type: ignore
        basis_function: BasisFunction = LinearBasisFunction(),
        **kwargs,
    ):
        super().__init__(name, dims, mapping, mapping_params, **kwargs)
        self.slope = slope or make_prior(dist_name="Normal", dist_params=(0, 10.0))
        self.slope.dims = ("covariates",) if not self.dims else ("covariates", *self.dims)
        self.intercept = intercept or make_prior(dist_name="Normal", dist_params=(0, 5.0))
        self.intercept.dims = self.dims
        self.sample_dims = ("observations",)
        self.set_name(self.name)
        self.basis_function = basis_function

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        if hasattr(self, "slope"):
            self.slope.dims = value
        if hasattr(self, "intercept"):
            self.intercept.dims = value
        self._dims = value

    def _compile(self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray):
        if not self.basis_function.is_fitted:
            self.basis_function.fit(X.values)  # Do this indexing to avoid ordering issues
        mapped_X = self.basis_function.transform(X.values)
        covs = f"{self.name}_covariates"
        self.covariate_dims = [f"{covs}_{i}" for i in range(mapped_X.shape[1])]

        if not model.coords.get(covs):
            model.add_coords({covs: self.covariate_dims})
        with model:
            pm_X = pm.Data(f"{self.name}_X", mapped_X, dims=("observations", covs))

        self.slope.dims = (covs,) if not self.dims else (covs, *self.dims)
        self.intercept.dims = self.dims
        self.one_dimensional = len(self.covariate_dims) == 1  # type: ignore
        slope_samples = self.slope.compile(model, X, be, be_maps, Y)
        intercept_samples = self.intercept.compile(model, X, be, be_maps, Y)
        if self.one_dimensional:
            return (slope_samples * pm_X)[:, 0] + intercept_samples
        else:
            return math.sum(slope_samples * pm_X, axis=1, keepdims=False) + intercept_samples

    def transfer(self, idata: az.InferenceData, **kwargs) -> "LinearPrior":
        new_slope = self.slope.transfer(idata, **kwargs)
        new_intercept = self.intercept.transfer(idata, **kwargs)
        new_basis_function = copy.deepcopy(self.basis_function)
        new_basis_function.compute_min = False
        new_basis_function.compute_max = False
        new_prior = LinearPrior(
            name=self.name,
            dims=self.dims,
            mapping=self.mapping,
            mapping_params=self.mapping_params,
            slope=new_slope,
            intercept=new_intercept,
            basis_function=new_basis_function,
        )
        return new_prior

    def update_data(
        self, model: pm.Model, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray
    ):
        mapped_X = self.basis_function.transform(X.values)
        model.set_data(f"{self.name}_X", mapped_X, coords={"observations": X.coords["observations"].values})

    def to_dict(self):
        dct = super().to_dict()
        dct["slope"] = self.slope.to_dict()
        dct["intercept"] = self.intercept.to_dict()
        dct["basis_function"] = self.basis_function.to_dict()
        return dct

    @classmethod
    def from_dict(
        cls, dict: dict, version: str | None = None
    ) -> "LinearPrior":
        slope = BasePrior.from_dict(dict["slope"], version=version)
        intercept = BasePrior.from_dict(
            dict["intercept"], version=version
        )
        basis_function = BasisFunction.from_dict(
            dict["basis_function"], version=version
        )
        return cls(
            slope=slope,
            intercept=intercept,
            basis_function=basis_function,
            **{
                k: v
                for k, v in dict.items()
                if k in ["name", "dims", "mapping", "mapping_params"]
            },
        )

    @property
    def has_random_effect(self):
        return self.slope.has_random_effect or self.intercept.has_random_effect

    def set_name(self, name):
        self.name = name
        self.slope.set_name(f"slope_{self.name}")
        self.intercept.set_name(f"intercept_{self.name}")
