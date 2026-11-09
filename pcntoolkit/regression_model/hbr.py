from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import arviz as az  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm  # type: ignore
import xarray as xr

from pcntoolkit.math_functions.factorize import *
from pcntoolkit.math_functions.likelihood import (
    Likelihood,
    get_default_normal_likelihood,
)
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.migration import registry
from pcntoolkit.util.output import Errors, Output, Warnings


class HBR(RegressionModel):
    """
    Hierarchical Bayesian Regression model implementation.

    This class implements a Bayesian hierarchical regression model using PyMC for
    posterior sampling. It supports multiple likelihood functions and provides
    methods for model fitting, prediction, and analysis.
    """

    def __init__(
        self,
        name: str = "template",
        is_fitted: bool = False,
        is_from_dict: bool = False,
        likelihood: Optional[Likelihood] = None,
        inference_method: Optional[InferenceMethod] = None,
        # TODO: Deprecation kwargs, drop them in a future release. Here for backward 
        # compatibility.
        draws: Optional[int] = None,
        tune: Optional[int] = None,
        cores: Optional[int] = None,
        chains: Optional[int] = None,
        init: Optional[str] = None,
        nuts_sampler: Optional[str] = None,
        progressbar: Optional[bool] = None,
    ) -> None:
        """
        Hierarchical Bayesian regression model fitted with PyMC.

        Select an ``inference_method`` to approximate the posterior distribution.

        Parameters
        ----------
        name : str, optional
            Unique identifier for the model instance, by default "template".
        is_fitted : bool, optional
            Whether the model has been fitted, by default False.
        is_from_dict : bool, optional
            Whether the model was created from a dictionary, by default False.
        likelihood : Likelihood, optional
            Distribution of the response variable. By default a Normal likelihood.
        inference_method : InferenceMethod, optional
            How to approximate the posterior, by default ``MCMC()``.
            Current supported methods include ``MCMC`` and ``Laplace``.
        draws, tune, cores, chains, init, nuts_sampler, progressbar : optional
            Deprecated MCMC arguments; pass them to ``MCMC(...)`` instead.

        Raises
        ------
        ValueError
            If a deprecated argument is given together with
            ``inference_method``.

        Examples
        --------
        >>> mcmc = MCMC(draws=1000, progressbar=False)
        >>> model = HBR(likelihood=likelihood, inference_method=mcmc)
        """
        super().__init__(name, is_fitted, is_from_dict)

        legacy = {
            "draws": draws,
            "tune": tune,
            "cores": cores,
            "chains": chains,
            "init": init,
            "nuts_sampler": nuts_sampler,
            "progressbar": progressbar,
        }
        provided = {k: v for k, v in legacy.items() if v is not None}
        if len(provided) > 0:
            if inference_method is not None:
                raise ValueError(
                    Output.error(
                        Errors.ERROR_HBR_MCMC_KWARGS_AND_INFERENCE_METHOD,
                        kwargs=list(provided.keys()),
                    )
                )
            else:
                Output.warning(
                    Warnings.HBR_MCMC_KWARGS_DEPRECATED, kwargs=list(provided.keys())
                )
                inference_method = MCMC()
                for k, v in provided.items():
                    setattr(inference_method, k, v)

        if inference_method is None:
            Output.print("No inference_method given; using MCMC()")
            inference_method = MCMC()

        self.likelihood = likelihood or get_default_normal_likelihood()
        self.inference_method = inference_method
        self.pymc_model: pm.Model = None  # type: ignore
        self.be_maps: dict = None  # type:ignore

    @property
    def idata(self) -> xr.DataTree:
        """Posterior samples stored on the inference method."""
        return self.inference_method.data_tree

    def fit(
        self,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
    ) -> None:
        """
        Fit the model to training data using MCMC sampling.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        be_maps : dict[str, dict[str, int]]
            Batch effect maps
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        None
        """
        self.be_maps = copy.deepcopy(be_maps)
        self.pymc_model: pm.Model = self.likelihood.compile(X, be, self.be_maps, Y)
        self.inference_method.fit(self.pymc_model)
        self.is_fitted = True

    def forward(
        self, X: xr.DataArray, be: xr.DataArray, Y: xr.DataArray
    ) -> xr.DataArray:
        """
        Map Y values to Z space using MCMC samples

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.DataArray
            Batch effect data
        Y : xr.DataArray
            Response variable data

        Returns
        -------
        xr.DataArray
            Z-values mapped to Y space
        """
        fn = self.likelihood.forward
        kwargs = {"Y": np.squeeze(Y.values)[:, None]}
        model = self.likelihood.create_model_with_data(X, be, self.be_maps, Y)
        params = self.likelihood.compile_params(model, X, be, self.be_maps, Y)
        return self.inference_method.apply(fn, model, params, kwargs)

    def backward(self, X, be, Z) -> xr.DataArray:  # type: ignore
        """
        Map Z values to Y space using MCMC samples

        Parameters
        ----------
        X : xr.DataArray
            Covariate data
        be : xr.
            Batch effect data
        Z : xr.DataArray
            Z-score data

        Returns
        -------
        xr.DataArray
            Z-values mapped to Y space
        """
        Y = xr.DataArray(np.zeros_like(Z.values), dims=Z.dims)
        fn = self.likelihood.backward
        kwargs = kwargs = {"Z": np.squeeze(Z.values)[:, None]}
        model = self.likelihood.create_model_with_data(X, be, self.be_maps, Y)
        params = self.likelihood.compile_params(model, X, be, self.be_maps, Y)
        return self.inference_method.apply(fn, model, params, kwargs)

    def elemwise_logp(self, X, be, Y):
        if not self.pymc_model:
            self.pymc_model = self.likelihood.compile(X, be, self.be_maps, Y)
        else:
            self.likelihood.update_data(self.pymc_model, X, be, self.be_maps, Y)
        return self.inference_method.elemwise_logp(self.pymc_model)

    # def model_specific_evaluation(self, path: str) -> None:
    #     """
    #     Save model-specific evaluation metrics.
    #     """
    #     plotdir = os.path.join(path, "plots")
    #     os.makedirs(plotdir, exist_ok=True)
    #     resultsdir = os.path.join(path, "results")
    #     os.makedirs(resultsdir, exist_ok=True)
    #     if self.is_fitted:
    #         if self.idata is not None:
    #             az.summary(
    #                 self.idata,
    #                 fmt="wide",
    #                 var_names=["~_per_subject"],
    #                 filter_vars="like",
    #             ).to_csv(os.path.join(resultsdir, self.name + "_summary.csv"))
    #             # Trace and autocorrelation plots only mean something for MCMC:
    #             # variational draws are independent by construction.
    #             if self.inference_method == "mcmc":
    #                 self._save_plot(
    #                     az.plot_trace_dist(
    #                         self.idata, var_names="~_per_subject", filter_vars="like"
    #                     ),
    #                     os.path.join(plotdir, self.name + "_trace.png"),
    #                 )
    #                 self._save_plot(
    #                     az.plot_autocorr(
    #                         self.idata, var_names="~_per_subject", filter_vars="like"
    #                     ),
    #                     os.path.join(plotdir, self.name + "_autocorr.png"),
    #                 )
    #             elif self.vi_loss is not None:
    #                 # For ADVI the ELBO trace is the convergence diagnostic.
    #                 plt.plot(self.vi_loss)
    #                 plt.xlabel("iteration")
    #                 plt.ylabel("ELBO loss")
    #                 plt.yscale("log")
    #                 plt.tight_layout()
    #                 plt.savefig(os.path.join(plotdir, self.name + "_elbo.png"))
    #                 plt.close()
    #             if "posterior_predictive" in self.idata.children:
    #                 self._save_plot(
    #                     az.plot_ppc_dist(self.idata),
    #                     os.path.join(plotdir, self.name + "_ppc.png"),
    #                 )
    #         if self.pymc_model is not None:
    #             self.pymc_model.to_graphviz(
    #                 save=os.path.join(plotdir, self.name + "_model.png")
    #             )
    #     else:
    #         raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))

    def transfer(
        self,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
        **kwargs: Any,
    ) -> HBR:
        """
        Perform transfer learning using existing model as prior.

        Parameters
        ----------
        X : xr.DataArray
            Covariate data of the new site(s).
        be : xr.DataArray
            Batch effect data of the new site(s).
        be_maps : dict[str, dict[str, int]]
            Maps each batch effect level to an integer index.
        Y : xr.DataArray
            Response variable data of the new site(s).
        **kwargs : Any
            ``freedom`` (float, default 1) sets how far the new model may move
            away from what the old model learned: 1 keeps the old uncertainty,
            2 allows twice as much, 0.5 half as much.

        Returns
        -------
        HBR
            New model instance with transferred knowledge
        """
        new_likelihood = self.inference_method.use_posterior_as_new_prior(
            self.likelihood, **kwargs
        )
        new_hbr_model = HBR(
            name=self.name,
            likelihood=new_likelihood,
            inference_method=self.inference_method.clone(),
        )
        new_hbr_model.fit(X, be, be_maps, Y)
        return new_hbr_model

    def has_batch_effect(self) -> bool:
        return False

    def to_dict(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Serialize model to dictionary format.

        Parameters
        ----------
        path : Optional[str], optional
            Path to save inference data, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing serialized model
        """
        my_dict = self.regmodel_dict
        my_dict["likelihood"] = self.likelihood.to_dict()
        my_dict["inference_method"] = self.inference_method.to_dict(path)
        for key, value in self.__dict__.items():
            if key not in [
                "likelihood",
                "pymc_model",
                "ptk_version",
                "inference_method",
            ]:
                my_dict[key] = value
        if self.is_fitted:
            my_dict["be_maps"] = copy.deepcopy(self.be_maps)
            my_dict["is_fitted"] = self.is_fitted
        else:
            my_dict["be_maps"] = None
        return my_dict

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Any], path: Optional[str] = None) -> "HBR":
        """
        Create model instance from serialized dictionary.

        Parameters
        ----------
        dict : Dict[str, Any]
            Dictionary containing serialized model
        path : Optional[str], optional
            Path to load inference data from, by default None

        Returns
        -------
        HBR
            New model instance
        """
        # Extract the saved version; default to "0.0.0" for old models.
        version: str = my_dict.get("ptk_version", "0.0.0")
        # Apply any registered HBR migrations for this version.
        my_dict = registry.migrate("HBR", my_dict, version=version)
        name: str = my_dict["name"]
        # Pass version down so likelihood/prior migrations are applied.
        likelihood: Likelihood = Likelihood.from_dict(
            my_dict["likelihood"], version=version
        )
        inference_method_type: InferenceMethod = globals()[
            my_dict["inference_method"]["type"]
        ]
        inference_method: str = inference_method_type.from_dict(
            my_dict["inference_method"], path
        )
        self = cls(
            name=name,
            is_from_dict=True,
            likelihood=likelihood,
            inference_method=inference_method,
        )
        self.is_fitted = my_dict["is_fitted"]
        self.be_maps = my_dict["be_maps"]
        return self

    @classmethod
    def from_args(cls, name: str, args: Dict[str, Any]) -> "HBR":
        """
        Create model instance from command line arguments.

        Parameters
        ----------
        name : str
            Name for new model instance
        args : Dict[str, Any]
            Dictionary of command line arguments

        Returns
        -------
        HBR
            New model instance
        """
        if args.get("inference_method", "mcmc") != "mcmc":
            raise ValueError(Output.error(Errors.ERROR_HBR_FROM_ARGS_ONLY_MCMC))
        return cls(
            name,
            is_fitted=args.get("is_fitted", False),
            is_from_dict=True,
            likelihood=Likelihood.from_args(args),
            inference_method=MCMC(**{k: args[k] for k in MCMC_INIT_KEYS if k in args}),
        )

    def compute_yhat(self, data, responsevar, X, be):
        fn = self.likelihood.yhat
        Y = xr.DataArray(np.squeeze(data.Y.values), dims=("observations",))
        model = self.likelihood.create_model_with_data(X, be, self.be_maps, Y)
        params = self.likelihood.compile_params(model, X, be, self.be_maps, Y)
        yhat = self.inference_method.apply(fn, model, params, kwargs={})
        return yhat

    @staticmethod
    def _save_plot(plot_collection: Any, path: str) -> None:
        """
        Save an ArviZ plot to disk.

        ArviZ 1.0 returns a PlotCollection instead of matplotlib axes, so we
        save its figure directly rather than relying on the current figure.

        Parameters
        ----------
        plot_collection : Any
            PlotCollection returned by an ArviZ plotting function
        path : str
            Path to save the figure to

        Returns
        -------
        None
        """
        figure = plot_collection.viz["figure"].item()
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)


class InferenceMethod(ABC):
    # Results of fitting; clone() does not copy these.
    _FITTED_ATTRS: tuple[str, ...] = ("data_tree", "is_fitted", "is_from_dict")

    @abstractmethod
    def fit(self, model: pm.Model):
        pass

    def clone(self) -> InferenceMethod:
        """
        Return a copy with the same settings (e.g. draws, chains) but no fitted results
        (e.g. ```data_tree``` which are the posterior samples).

        For example, a fitted ``MCMC(draws=1000)`` gives back a new, unfitted
        ``MCMC(draws=1000)``. ``HBR.transfer`` uses this to refit on new data.

        Returns
        -------
        InferenceMethod
            Unfitted copy of this inference method.
        """
        settings = {
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
            if k not in self._FITTED_ATTRS
        }
        return type(self)(**settings)

    @classmethod
    @abstractmethod
    def from_dict(cls, my_dict: dict, path: Optional[str]) -> InferenceMethod:
        pass

    def to_dict(self, path):
        my_dict = {"type": self.__class__.__name__}
        if self.is_fitted and (path is not None):
            data_tree_path = os.path.join(path, "data_tree.nc")
            self.save_data_tree(data_tree_path)
            my_dict["data_tree_path"] = data_tree_path
        for key, value in self.__dict__.items():
            if key not in ["data_tree"]:
                my_dict[key] = value
        return my_dict

    @abstractmethod
    def use_posterior_as_new_prior(
        self, likelihood: Likelihood, **kwargs: Any
    ) -> Likelihood:
        pass

    def apply(self, fn: Callable, model: pm.Model, params: dict[str, any], kwargs):
        """
        Apply a generic function to likelihood parameters
        """
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))
        var_names = [f"{k}_per_subject" for k, _ in params.items()]
        with model:
            for param_name, (value, dims) in params.items():
                pm.Deterministic(f"{param_name}_per_subject", value, dims=dims)
            data_tree = pm.sample_posterior_predictive(
                self.data_tree,
                extend_inferencedata=False,
                var_names=var_names,
                progressbar=False,
            )

        post_pred = az.extract(
            data_tree,
            "posterior_predictive",
            var_names=var_names,
        )

        n_observations = model.dim_lengths["observations"].eval().item()
        array_of_vars = list(
            map(
                lambda x: self.extract_and_reshape(post_pred, n_observations, x),
                var_names,
            )
        )
        result = xr.apply_ufunc(fn, *array_of_vars, kwargs=kwargs).mean(dim="sample")
        return result

    def elemwise_logp(self, model) -> xr.DataArray:  # type: ignore
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))
        with model:
            logp = pm.compute_log_likelihood(
                self.data_tree,
                var_names=["Yhat"],
                extend_inferencedata=False,
                progressbar=False,
            )
        return az.extract(logp, "log_likelihood", var_names=["Yhat"]).mean("sample")

    def extract_and_reshape(
        self, post_pred, observations, var_name: str
    ) -> xr.DataArray:
        preds = post_pred[var_name].values
        if len(preds.shape) == 1:
            preds = np.repeat(preds[None, :], observations, axis=0)
        return xr.DataArray(np.squeeze(preds), dims=["observations", "sample"])

    def save_data_tree(self, path: str) -> None:
        if self.is_fitted:
            if hasattr(self, "data_tree"):
                groups = {"posterior": self.data_tree["posterior"].dataset}
                if "fit" in self.data_tree.children:
                    groups["fit"] = self.data_tree["fit"].dataset
                xr.DataTree.from_dict(groups).to_netcdf(path)
            else:
                raise ValueError(Output.error(Errors.ERROR_HBR_FITTED_BUT_NO_IDATA))

    def load_data_tree(self, path: str) -> None:
        if self.is_fitted:
            try:
                self.data_tree = az.from_netcdf(path)
            except Exception as exc:
                raise ValueError(
                    Output.error(Errors.ERROR_HBR_COULD_NOT_LOAD_IDATA, path=path)
                ) from exc


MCMC_INIT_KEYS = [
    "draws",
    "tune",
    "cores",
    "chains",
    "init",
    "nuts_sampler",
    "progressbar",
]


class MCMC(InferenceMethod):
    def __init__(
        self,
        draws: int = 1500,
        tune: int = 500,
        cores: int = 4,
        chains: int = 4,
        init: str = "auto",
        nuts_sampler: str = "nutpie",
        progressbar: bool = True,
    ):
        self.data_tree: xr.DataTree = None  # type: ignore
        self.draws = draws
        self.tune = tune
        self.cores = cores
        self.chains = chains
        self.init = init
        self.nuts_sampler = nuts_sampler
        self.progressbar = progressbar
        self.is_fitted = False
        self.is_from_dict = False

    def fit(self, model: pm.Model):
        with model:
            self.data_tree = pm.sample(
                draws=self.draws,
                tune=self.tune,
                cores=self.cores,
                chains=self.chains,
                nuts_sampler=self.nuts_sampler,
                init=self.init,
                progressbar=self.progressbar,
            )
        self.is_fitted = True

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Any], path: Optional[str] = None) -> MCMC:
        mcmc = cls(**{k: my_dict[k] for k in MCMC_INIT_KEYS if k in my_dict})
        if my_dict["is_fitted"]:
            if path is not None:
                # Models saved before v1.4.0 store their samples in idata.nc.
                data_tree_path = os.path.join(
                    path,
                    os.path.basename(my_dict.get("data_tree_path", "data_tree.nc")),
                )
                mcmc.is_fitted = True
                mcmc.load_data_tree(data_tree_path)
            else:
                raise AttributeError(
                    Output.error(Errors.ERROR_HBR_FITTED_BUT_NO_DATA_TREE_PATH)
                )
        mcmc.is_from_dict = True
        return mcmc

    def use_posterior_as_new_prior(
        self, likelihood: Likelihood, **kwargs: Any
    ) -> Likelihood:
        return likelihood.transfer(self.data_tree, **kwargs)


class LaPlace(InferenceMethod):
    def __init__(self, draws=4000, progressbar=True, kwargs=None):
        self.draws = draws
        self.progressbar = progressbar
        self.kwargs = kwargs or {}
        self.data_tree = None
        self.is_fitted = False
        self.is_from_dict = False

    def fit(self, model: pm.Model):
        try:
            import pymc_extras as pmx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                Output.error(Errors.ERROR_HBR_LAPLACE_REQUIRES_PYMC_EXTRAS)
            ) from exc
        with model:
            self.data_tree = pmx.fit_laplace(
                draws=self.draws,
                progressbar=self.progressbar,
                **self.kwargs,
            )
        self.is_fitted = True

    @classmethod
    def from_dict(cls, my_dict: dict, path: Optional[str]) -> InferenceMethod:
        draws = my_dict["draws"]
        progressbar = my_dict["progressbar"]
        kwargs = my_dict["kwargs"] or {}
        is_fitted = my_dict["is_fitted"]
        laplace = cls(draws=draws, progressbar=progressbar, kwargs=kwargs)
        if is_fitted:
            if path is not None:
                data_tree_path = os.path.join(path, "data_tree.nc")
                laplace.is_fitted = True
                laplace.load_data_tree(data_tree_path)
            else:
                raise AttributeError(
                    Output.error(Errors.ERROR_HBR_FITTED_BUT_NO_DATA_TREE_PATH)
                )
        laplace.is_from_dict = True
        return laplace

    def use_posterior_as_new_prior(
        self, likelihood: Likelihood, **kwargs: Any
    ) -> Likelihood:
        return likelihood.transfer(self.data_tree, **kwargs)


class ADVI(InferenceMethod):
    def __init__(
        self,
        iterations: int,
        progressbar: bool = True,
        method: str = "advi",
        kwargs=None,
    ):
        self.iterations = iterations
        self.progressbar = progressbar
        # "advi" is mean-field; "fullrank_advi" models correlations.
        self.method = method
        self.approximation = None
        self.kwargs = kwargs or {}

    def fit(self, model: pm.Model):
        self.approx = pm.fit(
            n=self.iterations,
            method=self.method,
            progressbar=self.progressbar**self.kwargs,
        )
