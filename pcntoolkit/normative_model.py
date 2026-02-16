"""
Module providing the NormativeModel class, which is the main class for building and using normative models.
"""

from __future__ import annotations

import copy
import glob
import importlib.metadata
import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.scaler import Scaler
from pcntoolkit.math_functions.thrive import get_correlation_matrix, get_thrive_Z_X

# pylint: disable=unused-import
from pcntoolkit.regression_model.blr import BLR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.hbr import HBR  # noqa: F401 # type: ignore
from pcntoolkit.regression_model.regression_model import RegressionModel
from pcntoolkit.regression_model.test_model import TestModel  # noqa: F401 # type: ignore
from pcntoolkit.util.evaluator import Evaluator
from pcntoolkit.util.output import Errors, Messages, Output, Warnings
from pcntoolkit.util.paths import ensure_dir_exists, get_default_save_dir, get_save_subdirs
from pcntoolkit.util.plotter import plot_centiles, plot_qq


class NormativeModel:
    """
    This class provides the foundation for building normative models, handling multiple
    response variables through separate regression models. It manages data preprocessing,
    model fitting, prediction, and evaluation.

    Parameters
    ----------
    template_reg_model : RegressionModel
        Regression model used as a template to create all regression models.
    savemodel : bool
        Whether to save the model.
    evaluate_model : bool
        Whether to evaluate the model.
    saveresults : bool
        Whether to save the results.
    saveplots : bool
        Whether to save the plots.
    save_dir : str
        Directory to save the model, results, and plots.
    inscaler : str
        Input (X/covariates) scaler to use.
    outscaler: str
        Output (Y/response_vars) scaler to use.
    name: str
        Name of the model
    """

    def __init__(
        self,
        template_regression_model: RegressionModel,
        savemodel: bool = True,
        evaluate_model: bool = True,
        saveresults: bool = True,
        saveplots: bool = True,
        save_dir: Optional[str] = None,
        inscaler: str = "standardize",
        outscaler: str = "standardize",
        name: Optional[str] = None,
    ):
        self.savemodel: bool = savemodel
        self.evaluate_model: bool = evaluate_model
        self.saveresults: bool = saveresults
        self.saveplots: bool = saveplots
        self._save_dir = save_dir if save_dir is not None else get_default_save_dir()
        self.inscaler: str = inscaler
        self.outscaler: str = outscaler
        self.name: Optional[str] = name
        self.response_vars: list[str] = None  # type: ignore
        self.template_regression_model: RegressionModel = template_regression_model
        self.regression_models: dict[str, RegressionModel] = {}
        self.evaluator = Evaluator()
        self.inscalers: dict = {}
        self.outscalers: dict = {}
        self.is_fitted: bool = False
        self.covariates = None
        self.response_vars = None

        self.unique_batch_effects = None
        self.unique_batch_effects = None
        self.batch_effects_maps = None
        self.batch_effect_counts = None
        self.batch_effect_covariate_ranges = None
        self.covariate_ranges = None

        self.thrive_covariate = None
        self.correlation_matrix = None

    """
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################

        main functions

        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
    """

    def fit(self, data: NormData) -> None:
        """
        Fits a regression model for each response variable in the data.

        Parameters
        ----------
        data : NormData
            Training data containing covariates (X), batch effects (batch_effects), and response variables (Y).
            Must be a valid NormData object with properly formatted dimensions:
            - X: (n_samples, n_covariates)
            - batch_effects: (n_samples, n_batch_effects)
            - Y: (n_samples, n_response_vars)
        """
        self.register_data_info(data)
        self.preprocess(data)
        Output.print(Messages.FITTING_MODELS, n_models=len(self.response_vars))
        for responsevar in self.response_vars:
            Output.print(Messages.FITTING_MODEL, model_name=responsevar)
            resp_fit_data = data.sel({"response_vars": responsevar})
            X, be, be_maps, Y, _ = self.extract_data(resp_fit_data)
            self[responsevar].fit(X, be, be_maps, Y)
        self.is_fitted = True
        self.postprocess(data)
        if self.savemodel:  # Make sure model is saved
            self.save()
        self.predict(data)  # Make sure everything is evaluated and saved

    def predict(self, data: NormData) -> NormData:
        """Computes Z-scores, centiles, logp, yhat for each observation using fitted regression models."""
        self.set_ensure_save_dirs()
        self.compute_zscores(data)
        self.compute_centiles(data, recompute=True)
        self.compute_baseline_logp(data)
        self.compute_logp(data)
        self.compute_yhat(data)
        if self.evaluate_model:
            self.evaluate(data)
        if self.saveresults:
            resultsdir = os.path.join(self.save_dir, "results")
            data.save_results(resultsdir)
        if self.saveplots:
            plotdir = os.path.join(self.save_dir, "plots")
            plot_qq(data, plot_id_line=True, save_dir=plotdir)
            plot_centiles(
                self,
                data,
                save_dir=plotdir,
            )
        return data

    def fit_predict(self, fit_data: NormData, predict_data: NormData) -> NormData:
        """
        Combines model.fit and model.predict in a single operation.
        """
        self.fit(fit_data)
        self.predict(predict_data)
        return predict_data

    def transfer(self, transfer_data: NormData, save_dir: str | None = None, **kwargs) -> NormativeModel:
        """
        Transfers the model to a new dataset.
        """
        new_model = NormativeModel(
            copy.deepcopy(self.template_regression_model),
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            inscaler=self.inscaler,
            outscaler=self.outscaler,
            save_dir=self.save_dir,
        )
        if save_dir is not None:
            new_model.save_dir = save_dir
        else:
            new_model.save_dir = self.save_dir + "_transfer"
        new_model.covariates = copy.deepcopy(self.covariates)
        new_model.inscalers = copy.deepcopy(self.inscalers)
        new_model.outscalers = copy.deepcopy(self.outscalers)

        respvar_intersection = list(set(self.response_vars).intersection(transfer_data.response_vars.values))
        new_model.response_vars = respvar_intersection

        new_model.preprocess(transfer_data)
        new_model.register_batch_effects(transfer_data)

        Output.print(Messages.TRANSFERRING_MODELS, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            Output.print(Messages.TRANSFERRING_MODEL, model_name=responsevar)
            resp_transfer_data = transfer_data.sel({"response_vars": responsevar})
            X, be, be_maps, Y, _ = new_model.extract_data(resp_transfer_data)
            new_model[responsevar] = self[responsevar].transfer(X, be, be_maps, Y, **kwargs)
            #new_model[responsevar].be_maps = copy.deepcopy(be_maps)
        new_model.is_fitted = True
        new_model.postprocess(transfer_data)
        if new_model.savemodel:
            new_model.save()
        new_model.predict(transfer_data)  # Make sure everything is evaluated and saved
        return new_model

    def transfer_predict(
        self, transfer_data: NormData, predict_data: NormData, save_dir: str | None = None, **kwargs
    ) -> NormativeModel:
        """
        Transfers the model to a new dataset and predicts the data.
        """
        new_model = self.transfer(transfer_data, save_dir, **kwargs)
        new_model.predict(predict_data)
        return new_model

    def extend(self, data: NormData, save_dir: str | None = None, n_synth_samples: int | None = None) -> NormativeModel:
        """
        Extends the model to a new dataset.
        """
        synth = self.synthesize(n_samples=n_synth_samples, covariate_range_per_batch_effect=True)
        self.postprocess(synth)
        self.postprocess(data)
        merged_data = data.merge(synth)
        if save_dir is None:
            save_dir = self.save_dir + "_extend"

        new_model = NormativeModel(
            copy.deepcopy(self.template_regression_model),
            savemodel=True,
            evaluate_model=True,
            saveresults=True,
            saveplots=True,
            inscaler=self.inscaler,
            outscaler=self.outscaler,
            save_dir=save_dir,
        )

        new_model.fit(merged_data)
        return new_model

    def extend_predict(
        self, extend_data: NormData, predict_data: NormData, save_dir: str | None = None, n_synth_samples: int | None = None
    ) -> NormativeModel:
        """
        Extends the model to a new dataset and predicts the data.
        """
        new_model = self.extend(extend_data, save_dir, n_synth_samples)
        new_model.predict(predict_data)
        return new_model

    @classmethod
    def merge(cls, save_dir: str, models: list[Union[NormativeModel, str]]) -> NormativeModel:
        """
        Merges multiple models into a single model.
        """
        assert len(models) > 1, "At least two models are required to merge"
        if isinstance(models[0], NormativeModel):
            merged_model = copy.deepcopy(models[0])
        else:
            merged_model = cls.load(models[0])
        if save_dir is not None:
            merged_model.save_dir = save_dir
        else:
            merged_model.save_dir = merged_model.save_dir + "_merged"
        acc = merged_model.synthesize()
        for model in models[1:]:
            if isinstance(model, NormativeModel):
                acc = acc.merge(model.synthesize())
            else:
                this_model = cls.load(model)
                acc = acc.merge(this_model.synthesize())
                del this_model
        merged_model.fit(acc)
        return merged_model

    def synthesize(
        self, data: NormData | None = None, n_samples: int | None = None, covariate_range_per_batch_effect=False
    ) -> NormData:  # type: ignore
        """Synthesize data from the model

        Parameters
        ----------
        data : NormData, optional
            A NormData object with X and batch_effects. If provided, used to generate the synthetic data.
            If the data has no batch_effects, batch_effects are sampled from the model.
            If the data has no X, X is sampled from the model, using the provided or sampled batch_effects.
            If neither X nor batch_effects are provided, the model is used to generate the synthetic data.
        n_samples : int, optional
            Number of samples to synthesize. If this is None, the number of samples that were in the train data is used.
        covariate_range_per_batch_effect : bool, optional
            If True, the covariate range is different for each batch effect.
        """
        assert self.is_fitted
        if data:
            self.check_compatibility(data)
            data = copy.deepcopy(data)
            if n_samples is not None:
                Output.warning(Warnings.SYNTHESIZE_N_SAMPLES_IGNORED, n_samples=n_samples)
            n_samples = data.X.shape[0] if data.X is not None else data.batch_effects.shape[0]
            if not hasattr(data, "batch_effects") or data.batch_effects is None:
                data["batch_effects"] = self.sample_batch_effects(n_samples)  # type: ignore
            if not hasattr(data, "X") or data.X is None:
                data["X"] = self.sample_covariates(data.batch_effects, covariate_range_per_batch_effect)
        else:
            if n_samples is None:
                n_samples = self.n_fit_observations
            bes = self.sample_batch_effects(n_samples)
            X = self.sample_covariates(bes, covariate_range_per_batch_effect)
            subjects = xr.DataArray(np.arange(n_samples), dims=("observations",))
            data = NormData(
                name="synthesized",
                data_vars={"X": X, "batch_effects": bes, "subject_ids": subjects},
                coords={
                    "observations": np.arange(n_samples),
                    "response_vars": self.response_vars,
                },
                attrs={"real_ids": False},
            )

        data["Z"] = xr.DataArray(
            np.random.randn(n_samples, len(self.response_vars)),  # type: ignore
            dims=("observations", "response_vars"),
            coords=(data.coords["observations"], data.coords["response_vars"]),
        )

        data["Y"] = xr.DataArray(
            np.zeros((n_samples, len(self.response_vars))),  # type: ignore
            dims=("observations", "response_vars"),
            coords=(data.coords["observations"], data.coords["response_vars"]),
        )
        self.preprocess(data)
        Output.print(Messages.SYNTHESIZING_DATA, n_models=len(self.response_vars))
        for responsevar in self.response_vars:
            Output.print(Messages.SYNTHESIZING_DATA_MODEL, model_name=responsevar)
            resp_fit_data = data.sel({"response_vars": responsevar})
            resp_Z_data = data.Z.sel({"response_vars": responsevar})
            X, be, _, _, _ = self.extract_data(resp_fit_data)
            Z_pred = self[responsevar].backward(X, be, resp_Z_data)
            data["Y"].loc[{"response_vars": responsevar}] = Z_pred
        self.postprocess(data)
        return data

    def harmonize(self, data: NormData, reference_batch_effect: dict[str, str] | None = None) -> NormData:
        """Harmonizes the data to a reference batch effect. Harmonizes to the provided reference batch effect if provided,
        otherwise, harmonizes to the first batch effect alphabetically.

        Parameters
        ----------
        data : NormData
            Data to harmonize.
        reference_batch_effect : dict[str, str]
            Reference batch effect.
        """
        self.preprocess(data)
        _, be, _, _, _ = self.extract_data(data)
        ref_be_array = be.astype(str)
        if not reference_batch_effect:
            ref_be = {k: v[0] for k, v in data.get_single_batch_effect().items()}
        else:
            for k, v in data.get_single_batch_effect().items():
                if k not in reference_batch_effect:
                    reference_batch_effect[k] = v[0]
            ref_be = reference_batch_effect
        for k, v in ref_be.items():
            ref_be_array.loc[{"batch_effect_dims": k}] = v
        ref_be_array = self.map_batch_effects(ref_be_array)

        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        n_vars = len(respvar_intersection)
        Output.print(Messages.HARMONIZING_DATA, n_models=n_vars)

        data["Y_harmonized"] = xr.DataArray(
            np.zeros((data.X.shape[0], n_vars)),
            dims=("observations", "response_vars"),
            coords={"observations": data.observations, "response_vars": data.response_vars},
        )
        if hasattr(data, "thrive_Y"):
            data["thrive_Y_harmonized"] = xr.DataArray(
                np.zeros(data.thrive_Y.shape),
                dims=("observations", "response_vars", "offset"),
                coords={"observations": data.observations, "response_vars": data.response_vars},
            )
        for responsevar in respvar_intersection:
            Output.print(Messages.HARMONIZING_DATA_MODEL, model_name=responsevar)
            resp_fit_data = data.sel({"response_vars": responsevar})
            X, be, _, Y, _ = self.extract_data(resp_fit_data)
            Z_pred = self[responsevar].forward(X, be, Y)
            Y_harmonized = self[responsevar].backward(X, ref_be_array, Z_pred)
            data["Y_harmonized"].loc[{"response_vars": responsevar}] = Y_harmonized
            if hasattr(data, "thrive_Y"):
                for o in data.offset:
                    offset_X = X.copy()
                    # ! here
                    offset_X.loc[{"covariates": self.thrive_covariate}] = resp_fit_data.thrive_X.sel({"offset": o})
                    thrive_Y_harmonized = self[responsevar].backward(
                        offset_X, ref_be_array, resp_fit_data.thrive_Z.sel({"offset": o})
                    )
                    data["thrive_Y_harmonized"].loc[{"response_vars": responsevar, "offset": o}] = thrive_Y_harmonized
        self.is_fitted = True

        self.postprocess(data)
        return data

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model to a file.

        Args:
            path (str, optional): The path to save the model to. If None, the model is saved to the save_dir provided in the norm_conf.
        """
        savepath = path if path is not None else self.save_dir
        modelpath = os.path.join(savepath, "model")
        os.makedirs(modelpath, exist_ok=True)
        my_dict = self.to_dict()
        self.set_ensure_save_dirs()
        Output.print(Messages.SAVING_MODEL, save_dir=savepath)
        with open(os.path.join(modelpath, "normative_model.json"), "w", encoding="utf-8") as f:
            json.dump(my_dict, f, indent=4)

        for responsevar, model in self.regression_models.items():
            regmodel_path = os.path.join(modelpath, responsevar)
            os.makedirs(regmodel_path, exist_ok=True)
            reg_model_dict = {}
            reg_model_dict["model"] = model.to_dict(regmodel_path)
            reg_model_dict["outscaler"] = self.outscalers[responsevar].to_dict()
            with open(os.path.join(regmodel_path, "regression_model.json"), "w", encoding="utf-8") as f:
                json.dump(reg_model_dict, f, indent=4)

    @classmethod
    def load(cls, path: str, into: NormativeModel | None = None) -> NormativeModel:
        """
        Load a normative model from a path.

        Parameters
        ----------
        path : str
            The path to the normative model.
        into : NormBase, optional
            The normative model to load the data into. If None, a new normative model is created.
            This is useful if you want to load a normative model into an existing normative model, for example in the runner.
        """
        assert isinstance(path, str), f"Path must be a string, got {type(path)}"
        assert os.path.exists(path), f"Path {path} does not exist"
        model_path = os.path.join(path, "model", "normative_model.json")
        with open(model_path, mode="r", encoding="utf-8") as f:
            metadata = json.load(f)

        savemodel = metadata["savemodel"]
        saveresults = metadata["saveresults"]
        save_dir = metadata["save_dir"]
        inscaler = metadata["inscaler"]
        outscaler = metadata["outscaler"]
        saveplots = metadata["saveplots"]
        evaluate_model = metadata["evaluate_model"]
        name = metadata["name"]

        response_vars = []
        outscalers = {}
        regression_models = {}
        reg_models_path = os.path.join(path, "model", "*")
        for path in glob.glob(reg_models_path):
            if os.path.isdir(path):
                with open(
                    os.path.join(path, "regression_model.json"),
                    mode="r",
                    encoding="utf-8",
                ) as f:
                    reg_model_dict = json.load(f)
                    responsevar = reg_model_dict["model"]["name"]
                    response_vars.append(responsevar)
                    regression_model_type = globals()[reg_model_dict["model"]["type"]]
                    regression_models[responsevar] = regression_model_type.from_dict(reg_model_dict["model"], path)
                    outscalers[responsevar] = Scaler.from_dict(reg_model_dict["outscaler"])
        template_regression_model = type(regression_models[response_vars[0]]).from_dict(
            metadata["template_regression_model"], None
        )
        if into is None:
            self = cls(
                template_regression_model=template_regression_model,
                savemodel=savemodel,
                evaluate_model=evaluate_model,
                saveresults=saveresults,
                saveplots=saveplots,
                save_dir=save_dir,
                inscaler=inscaler,
                outscaler=outscaler,
                name=name,
            )
        else:
            self = into

        self.regression_models = regression_models
        self.outscalers = outscalers
        self.response_vars = response_vars
        self.inscalers = {k: Scaler.from_dict(v) for k, v in metadata["inscalers"].items()}

        if "batch_effects_maps" in metadata:
            self.batch_effects_maps = {
                be: {v: int(k.split("_")[1]) for k, v in mp.items()} for be, mp in metadata["batch_effects_maps"].items()
            }
        if "inverse_batch_effect_counts" in metadata:
            self.batch_effect_counts = {
                be: {v: int(k.split("_")[1]) for k, v in mp.items()} for be, mp in metadata["inverse_batch_effect_counts"].items()
            }
        if "batch_effect_covariate_ranges" in metadata:
            self.batch_effect_covariate_ranges = metadata["batch_effect_covariate_ranges"]

        if "unique_batch_effects" in metadata:
            self.unique_batch_effects = metadata["unique_batch_effects"]

        if "covariates" in metadata:
            self.covariates = metadata["covariates"]

        if "covariate_ranges" in metadata:
            self.covariate_ranges = metadata["covariate_ranges"]

        self.is_fitted = metadata["is_fitted"]

        return self

    def model_specific_evaluation(self) -> None:
        """
        Save model-specific evaluation metrics.
        """
        for responsevar in self.response_vars:
            self[responsevar].model_specific_evaluation(self.save_dir)

    """
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################

        helper functions

        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
    """

    def preprocess(self, data: NormData) -> None:
        """
        Applies preprocessing transformations to the input data.

        Args:
            data (NormData): Data to preprocess.
        """
        self.scale_forward(data)

    def scale_forward(self, data: NormData, overwrite: bool = False) -> None:
        """
        Scales input data to standardized form using configured scalers.

        Parameters
        ----------
        data : NormData
            Data object containing arrays to be scaled:
            - X : array-like, shape (n_samples, n_covariates)
                Covariate data to be scaled
            - y : array-like, shape (n_samples, n_response_vars), optional
                Response variable data to be scaled

        overwrite : bool, default=False
            If True, creates new scalers even if they already exist.
            If False, uses existing scalers when available.
        """
        for covariate in data.covariates.to_numpy():
            if (covariate not in self.inscalers) or overwrite:
                self.inscalers[covariate] = Scaler.from_string(self.inscaler)
                self.inscalers[covariate].fit(data.X.sel(covariates=covariate).data)

        for responsevar in data.response_vars.to_numpy():
            if (responsevar not in self.outscalers) or overwrite:
                self.outscalers[responsevar] = Scaler.from_string(self.outscaler)
                self.outscalers[responsevar].fit(data.Y.sel(response_vars=responsevar).data)

        data.scale_forward(self.inscalers, self.outscalers)

    def postprocess(self, data: NormData) -> None:
        """Apply postprocessing to the data.

        Args:
            data (NormData): Data to postprocess.
        """
        self.scale_backward(data)

    def scale_backward(self, data: NormData) -> None:
        """
        Scales data back to its original scale using stored scalers.

        Parameters
        ----------
        data : NormData
            Data object containing arrays to be scaled back:
            - X : array-like, shape (n_samples, n_covariates)
                Covariate data to be scaled back
            - y : array-like, shape (n_samples, n_response_vars), optional
                Response variable data to be scaled back
        """
        data.scale_backward(self.inscalers, self.outscalers)

    def evaluate(self, data: NormData) -> None:
        """
        Evaluates the model performance on the data.
        This method performs the following steps:
        1. Preprocesses the data

        5. Evaluates the model performance
        6. Postprocesses the data
        """
        # self.preprocess(data)
        self.evaluator.evaluate(data)
        # self.postprocess(data)

    def compute_zscores(self, data: NormData) -> NormData:
        """
        Computes Z-scores for each response variable using fitted regression models.

        Parameters
        ----------
        data : NormData
            Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).

        Returns
        -------
        NormData
            Prediction results containing:
            - Zscores: z-scores of the response variables
        """
        assert self.is_fitted, "Model is not fitted!"
        assert self.check_compatibility(data), "Data is not compatible with the model!"

        self.preprocess(data)
        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        Output.print(Messages.PREDICTING_MODELS, n_models=len(respvar_intersection))

        data["Z"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("observations", "response_vars"),
            coords={
                "observations": data.observations,
                "response_vars": list(respvar_intersection),
            },
        )
        Output.print(Messages.COMPUTING_ZSCORES, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            Output.print(Messages.COMPUTING_ZSCORES_MODEL, model_name=responsevar)
            resp_predict_data = data.sel({"response_vars": responsevar})
            X, be, _, Y, _ = self.extract_data(resp_predict_data)
            data["Z"].loc[{"response_vars": responsevar}] = self[responsevar].forward(X, be, Y)

        self.postprocess(data)
        return data

    def compute_centiles(self, data: NormData, centiles: Optional[List[float] | np.ndarray] = None, **kwargs) -> NormData:
        """
        Computes the centiles for each response variable in the data.

        Parameters
        ----------
        data : NormData
            Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).
        centiles : np.ndarray, optional
            The centiles to compute. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].

        Returns
        -------
        NormData
            Prediction results containing:
            - Centiles: centiles of the response variables
        """
        self.preprocess(data)

        if centiles is None:
            centiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        if isinstance(centiles, list):
            centiles = np.array(centiles)

        ppf = stats.norm.ppf(centiles)

        # Drop the centiles and dimensions if they already exist
        centiles_already_computed = "centiles" in data or "centile" in data.coords
        if centiles_already_computed:
            if not kwargs.get("recompute", False):
                if all([c in data.centile.values for c in centiles]):
                    Output.warning(
                        Warnings.CENTILES_ALREADY_COMPUTED_FOR_CENTILES, dataset_name=data.attrs["name"], centiles=centiles
                    )
                    return data
            del data.coords['centile']
            del data.dims.mapping['centile']
            del data['centiles']

        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["centiles"] = xr.DataArray(
            np.zeros((centiles.shape[0], data.X.shape[0], len(respvar_intersection))),
            dims=("centile", "observations", "response_vars"),
            coords={"centile": centiles},
        )

        Output.print(Messages.COMPUTING_CENTILES, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            Output.print(Messages.COMPUTING_CENTILES_MODEL, model_name=responsevar)
            X, be, _, _, _ = self.extract_data(resp_predict_data)
            for p, c in zip(ppf, centiles):
                Z = xr.DataArray(np.full(resp_predict_data.X.shape[0], p), dims=("observations",))
                data["centiles"].loc[{"response_vars": responsevar, "centile": c}] = self[responsevar].backward(X, be, Z)

        self.postprocess(data)
        return data

    def compute_baseline_logp(self, data: NormData) -> NormData:
        """
        Computes the log-probability of the data under a simple Gaussian model.
        
        The baseline model is a Gaussian with mean and standard deviation
        computed from the scaled Y data. This serves as a null model reference
        to evaluate for example the MSLL (Mean Standardized Log Loss) of our
        fitted model.

        Parameters
        ----------
        data : NormData
            Test data containing response variables (Y).

        Returns
        -------
        NormData
            Data with baseline_logp computed for each response variable.
        """
        self.preprocess(data)

        # Initialize logp array for a baseline Gaussian model with mean/std of the data
        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["baseline_logp"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("observations", "response_vars"),
            coords={"observations": data.observations},
        )

        # Compute the baseline Gaussian's model logp on scaled Y data
        Output.print(Messages.COMPUTING_LOGP, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            _, _, _, Y, _ = self.extract_data(resp_predict_data)
            y_scaled = Y.values
            baseline_logp = self.elemwise_logp_baseline_model(y_scaled)
            data["baseline_logp"].loc[{"response_vars": responsevar}] = baseline_logp

        self.postprocess(data)
        return data
    
    @staticmethod
    def elemwise_logp_baseline_model(y_scaled: np.ndarray) -> np.ndarray:
        """
        Compute log-probability for each observation under a baseline 
        Gaussian model.
        
        Parameters
        ----------
        y_scaled : np.ndarray
            Scaled response variable values.
            
        Returns
        -------
        np.ndarray
            Log-probability
        """
        baseline_mu = np.mean(y_scaled)
        baseline_sigma = np.std(y_scaled)
        return -0.5 * np.log(2 * np.pi * baseline_sigma**2) - ((y_scaled - baseline_mu) ** 2) / (2 * baseline_sigma**2)

    def compute_logp(self, data: NormData) -> NormData:
        """
        Computes the log-probability of the data under the fitted model.

        Parameters
        ----------
        data : NormData
            Test data containing covariates (X) for which to generate predictions, batch effects (batch_effects), and response variables (Y).

        Returns
        -------
        NormData
            Prediction results containing:
            - logp: log-probability of the response variables per datapoint under the fitted model
        """
        self.preprocess(data)

        # Initialise logp array with the correct dimensions and coordinates
        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["logp"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("observations", "response_vars"),
            coords={"observations": data.observations},
        )

        # Compute the fitted model's logp on scaled Y data
        Output.print(Messages.COMPUTING_LOGP, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            X, be, _, Y, _ = self.extract_data(resp_predict_data)
            Output.print(Messages.COMPUTING_LOGP_MODEL, model_name=responsevar)
            data["logp"].loc[{"response_vars": responsevar}] = self[responsevar].elemwise_logp(X, be, Y)

        self.postprocess(data)
        return data

    def compute_yhat(self, data: NormData) -> NormData:
        """
        Computes the predicted values for each response variable in the data.
        """
        self.preprocess(data)
        respvar_intersection = set(self.response_vars).intersection(data.response_vars.values)
        data["Yhat"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection))),
            dims=("observations", "response_vars"),
            coords={"observations": data.observations, "response_vars": list(respvar_intersection)},
        )
        Output.print(Messages.COMPUTING_YHAT, n_models=len(respvar_intersection))
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            X, be, _, _, _ = self.extract_data(resp_predict_data)
            data["Yhat"].loc[{"response_vars": responsevar}] = self[responsevar].compute_yhat(
                resp_predict_data, responsevar, X, be
            )
        self.postprocess(data)
        return data

    def compute_correlation_matrix(self, data, bandwidth=5, covariate="age"):
        self.thrive_covariate = covariate
        self.correlation_matrix = get_correlation_matrix(data, bandwidth, covariate)

    def compute_thrivelines(
        self: NormativeModel, data: NormData, span: int = 5, step: int = 1, z_thrive: float = 0.0, covariate="age", **kwargs
    ) -> NormData:
        """
        Computes the thrivelines for each responsevar in the data
        """
        data.attrs["thrive_covariate"] = self.thrive_covariate
        self.preprocess(data)
        # TODO: Write utility function to create a normdata object for easy thriveline creation (with appropriate Z scores)
        offsets = np.arange(0, span + 1, step=step)
        # Compute the thrivelines
        # Add them to the dataset, label them correctly

        # Drop the thrivelines and dimensions if they already exist
        thrivelines_already_computed = ("thrive_Z" in data or "thrive_Y" in data or "offset" in data.coords) and (
            data.attrs["z_thrive"] == z_thrive
        )
        if thrivelines_already_computed:
            if not kwargs.get("recompute", False):
                if all([c in data.offset.values for c in offsets]):
                    Output.warning(Warnings.THRIVELINES_ALREADY_COMPUTED_FOR, dataset_name=data.attrs["name"], offsets=offsets)
                    return data
            del data["thrive_Z"]
            del data["thrive_Y"]
            del data.coords["offset"]
            del data.dims.mapping["offset"]
        data.attrs["z_thrive"] = z_thrive

        # Make Z-score predictions if needed
        if not hasattr(data, "Z"):
            self.predict(data)

        respvar_intersection = list(set(self.response_vars).intersection(data.response_vars.values))

        # Get the covariate matrix that was derived during fit
        cormat = self.correlation_matrix

        # Create X, Y, and Z for thrivelines data
        data["thrive_Z"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection), offsets.shape[0])),
            dims=("observations", "response_vars", "offset"),
            coords={"offset": offsets},
        )
        data["thrive_Y"] = xr.DataArray(
            np.zeros((data.X.shape[0], len(respvar_intersection), offsets.shape[0])),
            dims=("observations", "response_vars", "offset"),
            coords={"offset": offsets},
        )
        for responsevar in respvar_intersection:
            resp_predict_data = data.sel({"response_vars": responsevar})
            X, be, _, m, Z = self.extract_data(resp_predict_data)
            X_cov = self.inscalers[covariate].inverse_transform(X.sel({"covariates": covariate}, drop=False))
            thrive_Z, thrive_X = get_thrive_Z_X(cormat.sel({"response_vars": responsevar}), X_cov, Z, span, z_thrive=z_thrive)
            data["thrive_X"] = xr.DataArray(
                self.inscalers[covariate].transform(thrive_X),
                dims=("observations", "offset"),
                coords={"offset": offsets},
            )
            data["thrive_Z"].loc[{"response_vars": responsevar}] = thrive_Z
            for io, o in enumerate(offsets):
                this_Z = thrive_Z[:, io]
                offset_X = X.copy()
                offset_X.loc[{"covariates": self.thrive_covariate}] = data.thrive_X.sel({"offset": o})
                scaled_thrive_Y = self[responsevar].backward(X, be, this_Z)
                data["thrive_Y"].loc[{"response_vars": responsevar, "offset": o}] = scaled_thrive_Y
        # self.postprocess(data)
        return data

    def register_data_info(self, data: NormData) -> None:
        self.covariates = data.covariates.to_numpy().copy().tolist()
        self.response_vars = data.response_vars.to_numpy().copy().tolist()
        self.register_batch_effects(data)

    def register_batch_effects(self, data: NormData) -> None:
        self.unique_batch_effects = copy.deepcopy(data.unique_batch_effects)
        self.unique_batch_effects = {k: list(v) for k, v in self.unique_batch_effects.items()}
        self.batch_effects_maps = {
            be: {k: i for i, k in enumerate(self.unique_batch_effects[be])} for be in self.unique_batch_effects.keys()
        }
        self.batch_effect_counts = copy.deepcopy(data.batch_effect_counts)
        self.batch_effect_covariate_ranges = copy.deepcopy(data.batch_effect_covariate_ranges)
        self.covariate_ranges = copy.deepcopy(data.covariate_ranges)

    def sample_batch_effects(self, n_samples: int) -> xr.DataArray:
        """
        Sample the batch effects from the estimated distribution.
        """
        max_batch_effect_count = max([len(v) for v in self.unique_batch_effects.values()])
        if n_samples < max_batch_effect_count:
            raise ValueError(
                Output.error(
                    Errors.SAMPLE_BATCH_EFFECTS,
                    n_samples=n_samples,
                    max_batch_effect_count=max_batch_effect_count,
                )
            )

        bes = xr.DataArray(
            np.zeros((n_samples, len(self.batch_effect_counts.keys()))).astype(str),
            dims=("observations", "batch_effect_dims"),
            coords={"observations": np.arange(n_samples), "batch_effect_dims": self.batch_effect_dims},
        )
        for be in self.batch_effect_dims:
            countsum = np.sum(list(self.batch_effect_counts[be].values()))
            bes.loc[{"batch_effect_dims": be}] = np.random.choice(
                list(self.batch_effect_counts[be].keys()),
                size=n_samples,
                p=[c / countsum for c in list(self.batch_effect_counts[be].values())],
            )
        return bes

    def sample_covariates(self, bes: xr.DataArray, covariate_range_per_batch_effect: bool = False) -> xr.DataArray:
        """
        Sample the covariates from the estimated distribution.

        Uses ranges of observed covariates matched with batch effects to create a representative sample
        """
        X = xr.DataArray(
            np.zeros((bes.shape[0], len(self.covariates))),
            dims=("observations", "covariates"),
            coords={"observations": np.arange(bes.shape[0]), "covariates": self.covariates},
        )
        if covariate_range_per_batch_effect:
            for c in self.covariates:
                for i in range(X.shape[0]):
                    running_min, running_max = -np.inf, np.inf
                    for k in self.batch_effect_dims:
                        my_be = bes.sel({"observations": i, "batch_effect_dims": k}).values.item()
                        running_min = max(running_min, self.batch_effect_covariate_ranges[k][my_be][c]["min"])
                        running_max = min(running_max, self.batch_effect_covariate_ranges[k][my_be][c]["max"])

                    X.loc[{"observations": i, "covariates": c}] = np.random.uniform(running_min, running_max, size=1).item()
        else:
            for c in self.covariates:
                X.loc[{"observations": np.arange(bes.shape[0]), "covariates": c}] = np.random.uniform(
                    self.covariate_ranges[c]["min"], self.covariate_ranges[c]["max"], size=(bes.shape[0])
                )
        return X

    def to_dict(self):
        my_dict = {
            "name": self.name,
            "save_dir": self.save_dir,
            "savemodel": self.savemodel,
            "saveresults": self.saveresults,
            "saveplots": self.saveplots,
            "evaluate_model": self.evaluate_model,
            "template_regression_model": self.template_regression_model.to_dict(),
            "inscalers": {k: v.to_dict() for k, v in self.inscalers.items()},
            "is_fitted": self.is_fitted,
            "inscaler": self.inscaler,
            "outscaler": self.outscaler,
            "ptk_version": importlib.metadata.version("pcntoolkit"),
        }

        if hasattr(self, "covariates"):
            my_dict["covariates"] = self.covariates

        if hasattr(self, "unique_batch_effects"):
            my_dict["unique_batch_effects"] = self.unique_batch_effects

        # JSON keys are always string, so we have use a trick to save the batch effects distributions, which may have string or int keys.
        # We invert the map, so that the original keys are stored as the values
        # The original integer values are then converted to strings by json, but we can safely convert them back to ints when loading
        # We also add an index to the keys to make sure they are unique
        if hasattr(self, "batch_effect_counts") and self.batch_effect_counts is not None:
            my_dict["inverse_batch_effect_counts"] = {
                be: {f"{i}_{k}": v for i, (v, k) in enumerate(self.batch_effect_counts[be].items())}
                for be, mp in self.batch_effect_counts.items()
            }
        if hasattr(self, "batch_effects_maps") and self.batch_effects_maps is not None:
            my_dict["batch_effects_maps"] = {
                be: {f"{i}_{k}": v for i, (v, k) in enumerate(self.batch_effects_maps[be].items())}
                for be in self.batch_effects_maps.keys()
            }
        if hasattr(self, "batch_effect_covariate_ranges") and self.batch_effect_covariate_ranges is not None:
            my_dict["batch_effect_covariate_ranges"] = copy.deepcopy(self.batch_effect_covariate_ranges)

        if hasattr(self, "covariate_ranges") and self.covariate_ranges is not None:
            my_dict["covariate_ranges"] = copy.deepcopy(self.covariate_ranges)

        return my_dict

    @classmethod
    def from_args(cls, **kwargs) -> NormativeModel:
        """
        Create a new normative model from command line arguments.

        Parameters
        ----------
        args : dict[str, str]
            A dictionary of command line arguments.

        Returns
        -------
        NormBase
            An instance of a normative model.

        Raises
        ------
        ValueError
            If the regression model specified in the arguments is unknown.
        """
        savemodel = kwargs.get("savemodel", True) in ["True", True]
        saveresults = kwargs.get("saveresults", True) in ["True", True]
        saveplots = kwargs.get("saveplots", True) in ["True", True]
        evaluate_model = kwargs.get("evaluate_model", True) in ["True", True]
        save_dir = kwargs.get("save_dir", "./saves")
        inscaler = kwargs.get("inscaler", "none")
        outscaler = kwargs.get("outscaler", "none")
        name = kwargs.get("name", None)
        assert "alg" in kwargs, "Algorithm must be specified"
        if kwargs["alg"] == "blr":
            template_regression_model = BLR.from_args("template", kwargs)
        elif kwargs["alg"] == "hbr":
            template_regression_model = HBR.from_args("template", kwargs)
        elif kwargs["alg"] == "test_model":
            template_regression_model = TestModel.from_args("template", kwargs)
        else:
            raise ValueError(Output.error(Errors.ERROR_UNKNOWN_CLASS, class_name=kwargs["alg"]))
        return cls(
            template_regression_model=template_regression_model,
            savemodel=savemodel,
            saveresults=saveresults,
            saveplots=saveplots,
            evaluate_model=evaluate_model,
            save_dir=save_dir,
            inscaler=inscaler,
            outscaler=outscaler,
            name=name,
        )

    def set_save_dir(self, save_dir: str) -> None:
        """Override the save_dir in the norm_conf.

        Args:
            save_dir (str): New save directory.
        """
        self.save_dir = save_dir

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value: str) -> None:
        self._save_dir = value
        self.set_ensure_save_dirs()

    def set_ensure_save_dirs(self):
        """
        Ensures that the save directories for results and plots are created when they are not there yet (otherwise resulted in an error)
        """
        model_dir, results_dir, plots_dir = get_save_subdirs(self.save_dir)
        ensure_dir_exists(results_dir)
        ensure_dir_exists(model_dir)
        if self.saveplots:
            ensure_dir_exists(plots_dir)


    def extract_data(
        self, data: NormData
    ) -> Tuple[xr.DataArray, xr.DataArray, dict[str, dict[str, int]], xr.DataArray, xr.DataArray]:
        """Returns a 5-tuple of covariates, batch effects, batch effect maps, response vars, Z-scores.
        If the variable is not available, returns None instead of the variable.
        """
        if hasattr(data, "X"):
            X = data.X
        else:
            X = None
        if hasattr(data, "batch_effects"):
            batch_effects = self.map_batch_effects(data.batch_effects)
            batch_effects_maps = self.batch_effects_maps
        else:
            batch_effects = None
            batch_effects_maps = None
        if hasattr(data, "Y"):
            Y = data.Y
        else:
            Y = None

        if hasattr(data, "Z"):
            Z = data.Z
        else:
            Z = None
        return X, batch_effects, batch_effects_maps, Y, Z  # type: ignore


    def check_compatibility(self, data: NormData) -> bool:
        """
        Check if the data is compatible with the model.

        Parameters
        ----------
        data : NormData
            Data to check compatibility with.

        Returns
        -------
        bool
            True if compatible, False otherwise
        """
        missing_covariates = [i for i in self.covariates if i not in data.covariates.values]
        if len(missing_covariates) > 0:
            Output.warning(
                Warnings.MISSING_COVARIATES,
                covariates=missing_covariates,
                dataset_name=data.name,
            )

        extra_covariates = [i for i in data.covariates.values if i not in self.covariates]
        if len(extra_covariates) > 0:
            Output.warning(
                Warnings.EXTRA_COVARIATES,
                covariates=extra_covariates,
                dataset_name=data.name,
            )

        extra_response_vars = [i for i in data.response_vars.values if i not in self.response_vars]
        if len(extra_response_vars) > 0:
            Output.warning(
                Warnings.EXTRA_RESPONSE_VARS,
                response_vars=extra_response_vars,
                dataset_name=data.name,
            )

        compatible = True
        unknown_batch_effects = {
            be: [u for u in unique if u not in self.unique_batch_effects[be]] for be, unique in data.unique_batch_effects.items()
        }
        compatible = sum([len(unknown) for unknown in unknown_batch_effects.values()]) == 0
        if not compatible:
            Output.warning(
                Warnings.UNKNOWN_BATCH_EFFECTS,
                batch_effects=unknown_batch_effects,
                dataset_name=data.name,
            )
        return (
            (len(missing_covariates) == 0) and (len(extra_covariates) == 0) and (len(extra_response_vars) == 0) and (compatible)
        )

    def map_batch_effects(self, batch_effects: xr.DataArray) -> xr.DataArray:
        mapped_batch_effects = xr.DataArray(
            np.zeros(batch_effects.values.shape).astype(int),
            dims=("observations", "batch_effect_dims"),
            coords={"batch_effect_dims": list(self.unique_batch_effects.keys())},
        )
        for i, be in enumerate(self.unique_batch_effects.keys()):
            vals = batch_effects.sel(batch_effect_dims=be).values
            unique_vals, inverses = np.unique(vals, return_inverse=True)
            unique_vals_mapped = np.array([self.batch_effects_maps[be][un] for un in unique_vals])
            mapped_batch_effects.loc[{"batch_effect_dims": be}] = unique_vals_mapped[list(inverses)]

        return mapped_batch_effects


    def __getitem__(self, key: str) -> RegressionModel:
        if key not in self.regression_models:
            self.regression_models[key] = copy.deepcopy(self.template_regression_model)
            self.regression_models[key]._name = key
        return self.regression_models[key]

    def __setitem__(self, key: str, value: RegressionModel) -> None:
        self.regression_models[key] = value

    @property
    def has_batch_effect(self) -> bool:
        """Returns whether the model has a batch effect.
        Returns:
            bool: True if the model has a batch effect, False otherwise. This currently looks at the template reg conf
        """
        return self.template_regression_model.has_batch_effect

    @property
    def batch_effect_dims(self) -> list[str]:
        """Returns the batch effect dimensions.
        Returns:
            list[str]: The batch effect dimensions.
        """
        return list(self.unique_batch_effects.keys())

    @property
    def n_fit_observations(self) -> int:
        """Returns the number of batch effects.
        Returns:
            int: The number of batch effects.
        """
        return sum(self.batch_effect_counts[self.batch_effect_dims[0]].values())
