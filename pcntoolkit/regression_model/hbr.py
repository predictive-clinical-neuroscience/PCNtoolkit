from __future__ import annotations

import os
from typing import Any, Dict, Optional

import arviz as az  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm  # type: ignore
import xarray as xr
import copy

from pcntoolkit.math_functions.factorize import *
from pcntoolkit.math_functions.likelihood import Likelihood, get_default_normal_likelihood
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.util.migration import registry
from pcntoolkit.util.output import Errors, Output


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
        likelihood: Likelihood = get_default_normal_likelihood(),  # type:ignore
        draws: int = 1500,
        tune: int = 500,
        cores: int = 4,
        chains: int = 4,
        nuts_sampler: str = "nutpie",
        init: str = "jitter+adapt_diag",
        progressbar: bool = True,
        is_fitted: bool = False,
        is_from_dict: bool = False,
    ):
        """
        This class implements a Bayesian hierarchical regression model using PyMC for
        posterior sampling.

        Parameters
        ----------
        name : str
            Unique identifier for the model instance
        likelihood : Likelihood
            Likelihood function to use for the model
        draws : int, optional
            Number of samples to draw from the posterior distribution per chain, by default 1000
        tune : int, optional
            Number of tuning samples to draw from the posterior distribution per chain, by default 500
        cores : int, optional
            Number of cores to use for parallel sampling, by default 4
        chains : int, optional
            Number of chains to use for parallel sampling, by default 4
        nuts_sampler : str, optional
            NUTS sampler to use for parallel sampling, by default "nutpie"
        init : str, optional
            Initialization method for the model, by default "jitter+adapt_diag"
        progressbar : bool, optional
            Whether to display a progress bar during sampling, by default True
        is_fitted : bool, optional
            Whether the model has been fitted, by default False
        is_from_dict : bool, optional
            Whether the model was created from a dictionary, by default False

        """
        super().__init__(name, is_fitted, is_from_dict)
        self.likelihood = likelihood or get_default_normal_likelihood()
        self.draws = draws
        self.tune = tune
        self.cores = cores
        self.chains = chains
        self.nuts_sampler = nuts_sampler
        self.init = init
        self.progressbar = progressbar
        self.idata: az.InferenceData = None  # type: ignore
        self.pymc_model: pm.Model = None  # type: ignore
        self.be_maps: dict = None  # type:ignore

    def fit(self, X: xr.DataArray, be: xr.DataArray, be_maps: dict[str, dict[str, int]], Y: xr.DataArray) -> None:
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
        with self.pymc_model:
            self.idata = pm.sample(
                self.draws,
                tune=self.tune,
                cores=self.cores,
                chains=self.chains,
                nuts_sampler=self.nuts_sampler,  # type: ignore
                init=self.init,
                progressbar=self.progressbar,
            )
        self.is_fitted = True

    def forward(self, X: xr.DataArray, be: xr.DataArray, Y: xr.DataArray) -> xr.DataArray:
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
        return self.generic_MCMC_apply(X, be, Y, fn, kwargs)

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
        return self.generic_MCMC_apply(X, be, Y, fn, kwargs)

    def generic_MCMC_apply(self, X, be, Y, fn, kwargs):
        """
        Apply a generic function to likelihood parameters
        """
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))

        model = self.likelihood.create_model_with_data(X, be, self.be_maps, Y)
        params = self.likelihood.compile_params(model, X, be, self.be_maps, Y)
        var_names = [f"{k}_per_subject" for k, _ in params.items()]
        with model:
            for param_name, (value, dims) in params.items():
                pm.Deterministic(f"{param_name}_per_subject", value, dims=dims)
            idata = pm.sample_posterior_predictive(
                self.idata,
                extend_inferencedata=False,
                var_names=var_names,
                progressbar=False,
            )

        post_pred = az.extract(
            idata,
            "posterior_predictive",
            var_names=var_names,
        )

        n_observations = model.dim_lengths["observations"].eval().item()
        array_of_vars = list(map(lambda x: self.extract_and_reshape(post_pred, n_observations, x), var_names))
        result = xr.apply_ufunc(fn, *array_of_vars, kwargs=kwargs).mean(dim="sample")
        return result

    def elemwise_logp(self, X, be, Y) -> xr.DataArray:  # type: ignore
        """
        Compute log-probabilities for each observation in the data.

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
        xr.DataArray
            Log-probabilities of the data
        """

        if not self.is_fitted:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))
        if not self.pymc_model:
            self.pymc_model = self.likelihood.compile(X, be, self.be_maps, Y)
        else:
            self.likelihood.update_data(self.pymc_model, X, be, self.be_maps, Y)
        with self.pymc_model:
            logp = pm.compute_log_likelihood(
                self.idata,
                var_names=["Yhat"],
                extend_inferencedata=False,
                progressbar=False,
            )
        return az.extract(logp, "log_likelihood", var_names=["Yhat"]).mean("sample")

    def model_specific_evaluation(self, path: str) -> None:
        """
        Save model-specific evaluation metrics.
        """
        plotdir = os.path.join(path, "plots")
        os.makedirs(plotdir, exist_ok=True)
        resultsdir = os.path.join(path, "results")
        os.makedirs(resultsdir, exist_ok=True)
        if self.is_fitted:
            if self.idata is not None:
                az.summary(self.idata, fmt="wide", var_names=["~_per_subject"], filter_vars="like").to_csv(
                    os.path.join(resultsdir, self.name + "_summary.csv")
                )
                az.plot_trace(self.idata, var_names="~_per_subject", filter_vars="like")
                plt.tight_layout()
                plt.savefig(os.path.join(plotdir, self.name + "_trace.png"))
                plt.close()
                az.plot_autocorr(self.idata, var_names="~_per_subject", filter_vars="like")
                plt.tight_layout()
                plt.savefig(os.path.join(plotdir, self.name + "_autocorr.png"))
                plt.close()
                if hasattr(self.idata, "posterior_predictive"):
                    az.plot_ppc(self.idata)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plotdir, self.name + "_ppc.png"))
                    plt.close()
            if self.pymc_model is not None:
                self.pymc_model.to_graphviz(save=os.path.join(plotdir, self.name + "_model.png"))
        else:
            raise ValueError(Output.error(Errors.HBR_MODEL_NOT_FITTED))

    def transfer(
        self,
        X: xr.DataArray,
        be: xr.DataArray,
        be_maps: dict[str, dict[str, int]],
        Y: xr.DataArray,
        **kwargs,
    ) -> HBR:
        """
                Perform transfer learning using existing model as prior.

                Parameters
                ----------
                hbrconf : HBRConf
                    Configuration for new model
                transferdata : HBRData
                    Data for transfer learning
                freedom : float
                    Parameter controlling influence of prior model (0-1)
        x
                Returns
                -------
                HBR
                    New model instance with transferred knowledge
        """
        new_likelihood = self.likelihood.transfer(self.idata, **kwargs)
        new_hbr_model = HBR(
            self.name,
            new_likelihood,
            self.draws,
            self.tune,
            self.cores,
            self.chains,
            self.nuts_sampler,
            self.init,
            self.progressbar,
            self.is_fitted,
            self.is_from_dict,
        )
        new_hbr_model_model = new_hbr_model.likelihood.compile(X, be, be_maps, Y)
        with new_hbr_model_model:
            new_hbr_model.idata = pm.sample(
                kwargs.get("draws", self.draws),
                tune=kwargs.get("tune", self.tune),
                cores=kwargs.get("cores", self.cores),
                chains=kwargs.get("chains", self.chains),
                nuts_sampler=kwargs.get("nuts_sampler", self.nuts_sampler),  # type: ignore
                progressbar=kwargs.get("progressbar", self.progressbar),
            )
            new_hbr_model.is_fitted = True
        new_hbr_model.pymc_model = new_hbr_model_model
        new_hbr_model.be_maps = be_maps
        return new_hbr_model

    def has_batch_effect(self) -> bool:
        return False

    def extract_and_reshape(self, post_pred, observations, var_name: str) -> xr.DataArray:
        preds = post_pred[var_name].values
        if len(preds.shape) == 1:
            preds = np.repeat(preds[None, :], observations, axis=0)
        return xr.DataArray(np.squeeze(preds), dims=["observations", "sample"])

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
        for key, value in self.__dict__.items():
            if key not in ["likelihood", "pymc_model", "idata"]:
                my_dict[key] = value
        if self.is_fitted and (path is not None):
            idata_path = os.path.join(path, "idata.nc")
            self.save_idata(idata_path)
            my_dict["idata_path"] = idata_path
        if self.is_fitted:
            my_dict["be_maps"] = copy.deepcopy(self.be_maps)
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
        draws: int = my_dict["draws"]
        tune: int = my_dict["tune"]
        cores: int = my_dict["cores"]
        chains: int = my_dict["chains"]
        nuts_sampler: str = my_dict["nuts_sampler"]
        init: str = my_dict["init"]
        progressbar: bool = my_dict["progressbar"]
        is_fitted: bool = my_dict["is_fitted"]
        is_from_dict: bool = True
        self = cls(name, likelihood, draws, tune, cores, chains, nuts_sampler, init, progressbar, is_fitted, is_from_dict)
        if is_fitted and (path is not None):
            idata_path = os.path.join(path, "idata.nc")
            self.load_idata(idata_path)
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
        likelihood = Likelihood.from_args(args)
        draws = args.get("draws", 1000)
        tune = args.get("tune", 1000)
        cores = args.get("cores", 1)
        chains = args.get("chains", 1)
        nuts_sampler = args.get("nuts_sampler", "pymc")
        init = args.get("init", "auto")
        progressbar = args.get("progressbar", True)
        is_fitted = args.get("is_fitted", False)
        is_from_dict = True
        self = cls(name, likelihood, draws, tune, cores, chains, nuts_sampler, init, progressbar, is_fitted, is_from_dict)
        return self

    def save_idata(self, path: str) -> None:
        """
        Save inference data to NetCDF file.

        Parameters
        ----------
        path : str
            Path to save inference data to. Should end in '.nc'

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If model is fitted but does not have inference data
        """
        if self.is_fitted:
            if hasattr(self, "idata"):
                self.idata.to_netcdf(path, groups=["posterior"])
            else:
                raise ValueError(Output.error(Errors.ERROR_HBR_FITTED_BUT_NO_IDATA))

    def load_idata(self, path: str) -> None:
        """
        Load inference data from NetCDF file.

        Parameters
        ----------
        path : str
            Path to load inference data from. Should end in '.nc'

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If model is fitted but inference data cannot be loaded from path
        """
        if self.is_fitted:
            try:
                self.idata = az.from_netcdf(path)
            except Exception as exc:
                raise ValueError(Output.error(Errors.ERROR_HBR_COULD_NOT_LOAD_IDATA, path=path)) from exc

    def compute_yhat(self, data, responsevar, X, be):
        fn = self.likelihood.yhat
        Y = xr.DataArray(np.squeeze(data.Y.values), dims=("observations",))
        yhat = self.generic_MCMC_apply(X, be, Y, fn, kwargs={})
        return yhat
