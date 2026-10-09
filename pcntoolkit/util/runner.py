import copy
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
from copy import deepcopy
from typing import Callable, Dict, Literal, Optional

import cloudpickle as pickle

from pcntoolkit.dataio.fileio import create_incremental_backup
from pcntoolkit.dataio.norm_data import NormData, merge_result_parts, result_parts
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.util.job_observer import JobObserver
from pcntoolkit.util.output import Errors, Messages, Output, Warnings
from pcntoolkit.util.paths import get_default_log_dir, get_default_temp_dir


class Runner:
    """Run NormativeModel tasks in this process or as scheduler jobs (SLURM, Torque).

    On a scheduler, each job writes its results as part files in
    ``<save_dir>/results/parts/``, so jobs do not read or lock shared files.
    With ``observe=True`` the Runner merges the parts into the usual result
    CSV files when the jobs end. With ``observe=False``, the parts are merged
    when ``NormData.load_results`` reads the results, or by
    ``pcntoolkit.dataio.norm_data.merge_result_parts(<save_dir>/results)``.
    """

    cross_validate: bool = False
    cv_folds: int = 5
    parallelize: bool = True
    job_type: str = "local"
    n_batches: int | None = None
    batch_size: int | None = 2
    n_cores: int = 1
    time_limit_str: str | int = "00:05:00"
    time_limit_seconds: int = 300
    environment: str = None  # type: ignore
    preamble: str = "module load anaconda3"
    memory: str = "5gb"
    max_retries: int = 3
    log_dir: str = ""
    temp_dir: str = ""

    def __init__(
        self,
        parallelize: bool = False,
        job_type: Literal["torque", "slurm"] = "slurm",
        n_batches: int | None = None,
        batch_size: int | None = None,
        n_cores: int = 1,
        time_limit: str | int = "00:05:00",
        memory: str = "5GB",
        max_retries: int = 3,
        environment: Optional[str] = None,
        cross_validate: bool = False,
        cv_folds: int = 5,
        preamble: str = "module load anaconda3",
        log_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize the runner.

        Parameters
        ----------
        parallelize : bool, optional
            Whether to parallelize the jobs.
        job_type : Literal["torque", "slurm"], optional
            The type of job to use.
        n_batches : int, optional
            The number of jobs to run in parallel.
        n_cores : int, optional
            The number of cores to use for each job.
        time_limit : str | int, optional
            The time limit for each job.
        memory : str, optional
            The memory to use for each job.
        max_retries : int, optional
            The maximum number of retries for each job.
        environment : str, optional
            The environment to use for each job.
        cross_validate : bool, optional
            Whether to cross-validate the model.
        cv_folds : int, optional
            The number of folds to use for cross-validation.
        preamble : str, optional
            The preamble to use for each job.
        log_dir : str, optional
            The directory to save the logs to.
        temp_dir : str, optional
            The directory to save the temporary files to.
        """
        self.parallelize = parallelize
        self.job_type = job_type
        self.n_batches = n_batches
        self.batch_size = batch_size
        if (self.n_batches is not None) and (self.batch_size is not None):
            Output.warning(
                Warnings.BATCH_SIZE_AND_N_BATCHES_SPECIFIED,
                n_batches=self.n_batches,
                batch_size=self.batch_size,
            )
            self.batch_size = None
        self.n_cores = n_cores
        try:
            if isinstance(time_limit, str):
                self.time_limit_str = time_limit
                self.time_limit_seconds = sum(
                    [
                        int(v) * 60**i
                        for i, v in enumerate(reversed(self.time_limit_str.split(":")))
                    ]
                )
            elif isinstance(time_limit, int):
                self.time_limit_seconds = time_limit
                s = self.time_limit_seconds
                self.time_limit_str = f"{str(s // 3600)}:{str((s // 60) % 60).rjust(2, '0')}:{str(s % 60).rjust(2, '0')}"
        except Exception:
            raise ValueError(
                Output.error(Errors.ERROR_PARSING_TIME_LIMIT, time_limit_str=time_limit)
            )
        self.memory = memory
        self.max_retries = max_retries
        if parallelize:
            if not environment:
                raise ValueError(Output.error(Errors.ERROR_NO_ENVIRONMENT_SPECIFIED))
            else:
                # Check if the environment is valid
                if not os.path.exists(os.path.join(environment, "bin", "python")):
                    raise ValueError(
                        Output.error(
                            Errors.INVALID_ENVIRONMENT, environment=environment
                        )
                    )
                else:
                    self.environment = environment
                    self.preamble = preamble

        self.cross_validate = cross_validate
        self.cv_folds = cv_folds
        if self.cross_validate and self.cv_folds <= 1:
            raise ValueError(
                Output.error(
                    Errors.ERROR_CROSS_VALIDATION_FOLDS, cv_folds=self.cv_folds
                )
            )
        if log_dir is None:
            self.log_dir = get_default_log_dir()
            Output.print(Messages.NO_LOG_DIR_SPECIFIED, log_dir=self.log_dir)
        else:
            self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        if temp_dir is None:
            self.temp_dir = get_default_temp_dir()
            Output.print(Messages.NO_TEMP_DIR_SPECIFIED, temp_dir=self.temp_dir)
        else:
            self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.task_id = ""
        self.unique_temp_dir = ""
        self.unique_log_dir = ""
        self.job_observer = None
        self.save_dir = ""
        self.job_commands: Dict[str, list[str]] = {}
        self.active_jobs: Dict[str, str] = {}
        self.failed_jobs: Dict[str, str] = {}

    def wait_or_finish(
        self, observe: bool, into: NormativeModel | None = None, *data_sources
    ) -> NormativeModel | None:
        if self.parallelize:
            if observe:
                self.job_observer = JobObserver(
                    self.active_jobs, self.job_type, self.unique_log_dir, self.task_id
                )
                self.job_observer.wait_for_jobs()
                _, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
                self.merge_result_parts()
                if data_sources:
                    for data_source in data_sources:
                        self.load_data(data_source)
                return self.load_model(into=into)
            else:
                self.save()
                return None
        else:
            return self.load_model(into=into)

    def merge_result_parts(self) -> None:
        """Merge the result part files of the jobs into the result CSV files.

        This covers ``<save_dir>/results`` and, with cross-validation, the
        results folder of each fold.
        """
        result_dirs = [os.path.join(self.save_dir, "results")]
        folds = os.path.join(glob.escape(self.save_dir), "folds", "fold_*", "results")
        result_dirs += glob.glob(folds)
        for result_dir in result_dirs:
            merge_result_parts(result_dir)

    def set_task_id(self, task_name: str, model: NormativeModel, data: NormData):
        unique_id = ""
        if task_name is not None:
            unique_id = task_name + "_"
        if model.name is not None:
            unique_id = unique_id + model.name + "_"
        if data.name is not None:
            unique_id = unique_id + data.name + "_"
        milliseconds = time.time() * 1000 % 1000
        milliseconds = f"{milliseconds:03f}"
        # No ":" in the time: task_id is a folder name, and Windows does not allow ":".
        self.task_id = (
            unique_id + "_" + time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + milliseconds
        )
        Output.print(Messages.TASK_ID_CREATED, task_id=self.task_id)

    def create_temp_and_log_dir(self):
        self.unique_temp_dir = os.path.join(self.temp_dir, self.task_id)
        self.unique_log_dir = os.path.join(self.log_dir, self.task_id)
        os.makedirs(self.unique_temp_dir, exist_ok=True)
        with open(os.path.join(self.unique_temp_dir, "README.txt"), "wt") as file:
            file.write(
                "The contents of the folder containing this README file are temporary files used by the Runner. Once the Runner has successfully finished all its jobs, this folder and all its contents can be safely deleted."
            )
        os.makedirs(self.unique_log_dir, exist_ok=True)
        Output.print(Messages.TEMP_DIR_CREATED, temp_dir=self.unique_temp_dir)
        Output.print(Messages.LOG_DIR_CREATED, log_dir=self.unique_log_dir)

    def fit(
        self,
        model: NormativeModel,
        data: NormData,
        save_dir: Optional[str] = None,
        observe: bool = True,
    ) -> NormativeModel | None:
        """
        Fit a normative model on a dataset.

        Parameters
        ----------
        model : NormBase
            The normative model to fit.
        data : NormData
            The data to fit the model on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
            If false, the function will dispatch the jobs and return. In that case, the model will not be loaded into the model object, it will have to be loaded manually using the load function when the jobs are done.

        Returns
        -------
        NormativeModel | None
            The fitted model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        save_dir = save_dir if save_dir is not None else model.save_dir
        self.save_dir = save_dir
        self.set_task_id("fit", model, data)
        self.create_temp_and_log_dir()
        fn = self.get_fit_chunk_fn(model, save_dir)
        self.submit_jobs(fn, first_data_source=data, mode="unary")
        return self.wait_or_finish(observe, model, data)

    def fit_predict(
        self,
        model: NormativeModel,
        fit_data: NormData,
        predict_data: Optional[NormData] = None,
        save_dir: Optional[str] = None,
        observe: bool = True,
    ) -> NormativeModel | None:
        """
        Fit a normative model on a dataset and predict on another dataset.

        Parameters
        ----------
        model : NormativeModel
            The normative model to fit.
        fit_data : NormData
            The data to fit the model on.
        predict_data : Optional[NormData], optional
            The data to predict on. Can be None if cross-validation is used.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish, then load the model into the model object
            If false, the function will dispatch the jobs and return. In that case, the model will not be loaded into the model object, it will have to be loaded manually using the load function when the jobs are done.

        Returns
        -------
        NormativeModel | None
            The fitted and model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        save_dir = save_dir if save_dir is not None else model.save_dir
        self.save_dir = save_dir
        self.set_task_id("fit_predict", model, fit_data)
        self.create_temp_and_log_dir()
        fn = self.get_fit_predict_chunk_fn(model, save_dir)
        self.submit_jobs(
            fn,
            first_data_source=fit_data,
            second_data_source=predict_data,
            mode="binary",
        )
        return self.wait_or_finish(observe, model, fit_data, predict_data)

    def predict(
        self,
        model: NormativeModel,
        data: NormData,
        save_dir: Optional[str] = None,
        observe: bool = True,
    ) -> NormativeModel | None:
        """
        Predict on a dataset.

        Parameters
        ----------
        model : NormativeModel
            The normative model to predict on.
        data : NormData
            The data to predict on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish.

        Returns
        -------
        None. If you want to load the model, use the runner.load_model function.
        """
        save_dir = save_dir if save_dir is not None else model.save_dir
        assert save_dir is not None
        self.save_dir = save_dir
        self.set_task_id("predict", model, data)
        self.create_temp_and_log_dir()
        fn = self.get_predict_chunk_fn(model, save_dir)
        self.submit_jobs(fn, first_data_source=data, mode="unary")
        return self.wait_or_finish(observe, None, data)

    def transfer(
        self,
        model: NormativeModel,
        data: NormData,
        save_dir: Optional[str] = None,
        observe: bool = True,
        **kwargs,
    ) -> NormativeModel | None:
        """
        Transfer a normative model to a new dataset.

        Parameters
        ----------
        model : NormativeModel
            The normative model to transfer.
        data : NormData
            The data to transfer the model to.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then return the transfered model.
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormBase | None
            The transfered model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        save_dir = save_dir if save_dir is not None else model.save_dir + "_transfer"
        assert save_dir is not None
        self.save_dir = save_dir
        self.set_task_id("transfer", model, data)
        self.create_temp_and_log_dir()
        fn = self.get_transfer_chunk_fn(model, save_dir, **kwargs)
        self.submit_jobs(fn, data, mode="unary")
        return self.wait_or_finish(observe, None, data)

    def transfer_predict(
        self,
        model: NormativeModel,
        fit_data: NormData,
        predict_data: Optional[NormData] = None,
        save_dir: Optional[str] = None,
        observe: bool = True,
        **kwargs,
    ) -> NormativeModel | None:
        """
        Transfer a normative model to a new dataset and predict on another dataset.

        Parameters
        ----------
        model : NormativeModel
            The normative model to transfer.
        fit_data : NormData
            The data to transfer the model to.
        predict_data : Optional[NormData], optional
            The data to predict on. Can be None if cross-validation is used.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormBase | No   ne
            The transfered model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        save_dir = save_dir if save_dir is not None else model.save_dir + "_transfer"
        assert save_dir is not None
        self.save_dir = save_dir
        self.set_task_id("transfer_predict", model, fit_data)
        self.create_temp_and_log_dir()
        fn = self.get_transfer_predict_chunk_fn(model, save_dir, **kwargs)
        self.submit_jobs(fn, fit_data, predict_data, mode="binary")
        return self.wait_or_finish(observe, None, fit_data, predict_data)

    def extend(
        self,
        model: NormativeModel,
        data: NormData,
        save_dir: Optional[str] = None,
        observe: bool = True,
        **kwargs,
    ) -> NormativeModel | None:
        """
        Extend a normative model on a dataset.

        Parameters
        ----------
        model : NormativeModel
            The normative model to extend.
        data : NormData
            The data to extend the model on.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            Whether to observe the jobs. If true, the function will wait for the jobs to finish and then load the model into the model object.
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormativeModel | None
            The extended model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        save_dir = save_dir if save_dir is not None else model.save_dir + "_extend"
        assert save_dir is not None
        self.save_dir = save_dir
        self.set_task_id("extend", model, data)
        self.create_temp_and_log_dir()
        fn = self.get_extend_chunk_fn(model, save_dir, **kwargs)
        self.submit_jobs(fn, data, mode="unary")
        return self.wait_or_finish(observe, None, data)

    def extend_predict(
        self,
        model: NormativeModel,
        fit_data: NormData,
        predict_data: Optional[NormData] = None,
        save_dir: Optional[str] = None,
        observe: bool = True,
        **kwargs,
    ) -> NormativeModel | None:
        """
        Extend a normative model on a dataset and predict on another dataset.

        Parameters
        ----------
        model : NormativeModel
            The normative model to extend.
        fit_data : NormData
            The data to extend the model on.
        predict_data : Optional[NormData], optional
            The data to predict on. Can be None if cross-validation is used.
        save_dir : Optional[str], optional
            The directory to save the model to. If None, the model will be saved in the model's save directory.
        observe : bool, optional
            If false, the function will dispatch the jobs and return.

        Returns
        -------
        NormativeModel | None
            The extended model. If observe is true, the function will wait for the jobs to finish and return the model object. If observe is false, the function will return None.
        """
        save_dir = save_dir if save_dir is not None else model.save_dir + "_extend"
        assert save_dir is not None
        self.save_dir = save_dir
        self.set_task_id("extend_predict", model, fit_data)
        self.create_temp_and_log_dir()
        fn = self.get_extend_predict_chunk_fn(model, save_dir, **kwargs)
        self.submit_jobs(fn, fit_data, predict_data, mode="binary")
        return self.wait_or_finish(observe, None, fit_data, predict_data)

    def get_fit_chunk_fn(self, model: NormativeModel, save_dir: str) -> Callable:
        """Returns a callable that fits a model on a chunk of data"""
        if self.cross_validate:

            def kfold_fit_chunk_fn(chunk: NormData):
                for i_fold, (fit_idx, predict_idx) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    self.register_fold_indices(
                        model.save_dir, i_fold, (fit_idx, predict_idx)
                    )
                    train_data = copy.deepcopy(chunk.isel(observations=fit_idx))
                    fold_norm_model: NormativeModel = deepcopy(model)
                    fold_norm_model.set_save_dir(
                        os.path.join(save_dir, "folds", f"fold_{i_fold}")
                    )
                    train_data.attrs["name"] = (
                        train_data.attrs["name"] + "_fold_" + str(i_fold) + "_fit"
                    )
                    fold_norm_model.fit(train_data)

            return kfold_fit_chunk_fn
        else:

            def fit_chunk_fn(chunk: NormData):
                model.set_save_dir(save_dir)
                model.fit(chunk)

            return fit_chunk_fn

    def get_fit_predict_chunk_fn(
        self, model: NormativeModel, save_dir: str
    ) -> Callable:
        """Returns a callable that fits a model on a chunk of data and predicts on another chunk of data"""
        if self.cross_validate:

            def kfold_fit_predict_chunk_fn(
                all_data: NormData, unused_predict_data: Optional[NormData] = None
            ):
                if unused_predict_data is not None:
                    Output.warning(
                        Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION
                    )
                for i_fold, (fit_idx, predict_idx) in enumerate(
                    all_data.kfold_split(self.cv_folds)
                ):
                    self.register_fold_indices(
                        model.save_dir, i_fold, (fit_idx, predict_idx)
                    )
                    fit_data = copy.deepcopy(all_data.isel(observations=fit_idx))
                    predict_data = copy.deepcopy(
                        all_data.isel(observations=predict_idx)
                    )
                    fold_norm_model: NormativeModel = deepcopy(model)
                    fold_norm_model.set_save_dir(
                        os.path.join(save_dir, "folds", f"fold_{i_fold}")
                    )
                    fit_data.attrs["name"] = (
                        fit_data.attrs["name"] + "_fold_" + str(i_fold) + "_train"
                    )
                    predict_data.attrs["name"] = (
                        predict_data.attrs["name"] + "_fold_" + str(i_fold) + "_predict"
                    )
                    fold_norm_model.fit_predict(fit_data, predict_data)

            return kfold_fit_predict_chunk_fn
        else:

            def fit_predict_chunk_fn(
                fit_data: NormData, predict_data: Optional[NormData]
            ):
                if predict_data is None:
                    raise ValueError(
                        Output.error(
                            Errors.ERROR_PREDICT_DATA_REQUIRED_FOR_FIT_PREDICT_WITHOUT_CROSS_VALIDATION
                        )
                    )
                assert predict_data is not None  # Make the linter happy
                model.set_save_dir(save_dir)
                model.fit_predict(fit_data, predict_data)

            return fit_predict_chunk_fn

    def get_predict_chunk_fn(self, model: NormativeModel, save_dir: str) -> Callable:
        """Loads each fold model and predicts on the corresponding fold of data. Model n is used to predict on fold n."""
        if self.cross_validate:
            raise ValueError(
                Output.error(
                    Errors.ERROR_PREDICT_DATA_NOT_SUPPORTED_FOR_CROSS_VALIDATION
                )
            )
        else:

            def predict_chunk_fn(chunk: NormData):
                model.set_save_dir(save_dir)
                model.predict(chunk)

            return predict_chunk_fn

    def get_transfer_chunk_fn(
        self, model: NormativeModel, save_dir: str, **kwargs
    ) -> Callable:
        """Returns a callable that transfers a model on a chunk of data"""
        if self.cross_validate:

            def kfold_transfer_chunk_fn(chunk: NormData):
                for i_fold, (fit_idx, predict_idx) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    self.register_fold_indices(
                        model.save_dir, i_fold, (fit_idx, predict_idx)
                    )
                    train_data = copy.deepcopy(chunk.isel(observations=fit_idx))
                    train_data.attrs["name"] = (
                        train_data.attrs["name"] + "_fold_" + str(i_fold) + "_fit"
                    )
                    model.transfer(
                        train_data,
                        save_dir=os.path.join(save_dir, "folds", f"fold_{i_fold}"),
                        **kwargs,
                    )

            return kfold_transfer_chunk_fn
        else:

            def transfer_chunk_fn(data: NormData):
                model.set_save_dir(save_dir)
                model.transfer(data, save_dir=model.save_dir, **kwargs)

            return transfer_chunk_fn

    def get_transfer_predict_chunk_fn(
        self, model: NormativeModel, save_dir: str, **kwargs
    ) -> Callable:
        if self.cross_validate:

            def kfold_transfer_predict_chunk_fn(
                chunk: NormData, unused_predict_data: Optional[NormData] = None
            ):
                if unused_predict_data is not None:
                    Output.warning(
                        Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION
                    )
                for i_fold, (fit_idx, predict_idx) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    self.register_fold_indices(
                        model.save_dir, i_fold, (fit_idx, predict_idx)
                    )
                    train_data = copy.deepcopy(chunk.isel(observations=fit_idx))
                    predict_data = copy.deepcopy(chunk.isel(observations=predict_idx))
                    train_data.attrs["name"] = (
                        train_data.attrs["name"] + "_fold_" + str(i_fold) + "_fit"
                    )
                    predict_data.attrs["name"] = (
                        predict_data.attrs["name"] + "_fold_" + str(i_fold) + "_predict"
                    )
                    model.transfer_predict(
                        train_data,
                        predict_data,
                        save_dir=os.path.join(save_dir, "folds", f"fold_{i_fold}"),
                        **kwargs,
                    )

            return kfold_transfer_predict_chunk_fn
        else:

            def transfer_predict_chunk_fn(train_data: NormData, predict_data: NormData):
                if predict_data is None:
                    raise ValueError(Output.error(Errors.ERROR_PREDICT_DATA_REQUIRED))
                model.transfer_predict(
                    train_data, predict_data, save_dir=save_dir, **kwargs
                )

            return transfer_predict_chunk_fn

    def get_extend_chunk_fn(
        self, model: NormativeModel, save_dir: str, **kwargs
    ) -> Callable:
        if self.cross_validate:

            def kfold_extend_chunk_fn(chunk: NormData):
                for i_fold, (fit_idx, predict_idx) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    self.register_fold_indices(
                        model.save_dir, i_fold, (fit_idx, predict_idx)
                    )
                    train_data = copy.deepcopy(chunk.isel(observations=fit_idx))
                    train_data.attrs["name"] = (
                        train_data.attrs["name"] + "_fold_" + str(i_fold) + "_fit"
                    )
                    model.extend(
                        data=train_data,
                        save_dir=os.path.join(save_dir, "folds", f"fold_{i_fold}"),
                        **kwargs,
                    )

            return kfold_extend_chunk_fn
        else:

            def extend_chunk_fn(data: NormData):
                model.set_save_dir(save_dir)
                model.extend(data=data, save_dir=save_dir, **kwargs)

            return extend_chunk_fn

    def get_extend_predict_chunk_fn(
        self, model: NormativeModel, save_dir: str, **kwargs
    ) -> Callable:
        if self.cross_validate:

            def kfold_extend_predict_chunk_fn(
                chunk: NormData, unused_predict_data: Optional[NormData] = None
            ):
                if unused_predict_data is not None:
                    Output.warning(
                        Warnings.PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION
                    )
                for i_fold, (fit_idx, predict_idx) in enumerate(
                    chunk.kfold_split(self.cv_folds)
                ):
                    self.register_fold_indices(
                        model.save_dir, i_fold, (fit_idx, predict_idx)
                    )
                    train_data = copy.deepcopy(chunk.isel(observations=fit_idx))
                    predict_data = copy.deepcopy(chunk.isel(observations=predict_idx))
                    train_data.attrs["name"] = (
                        train_data.attrs["name"] + "_fold_" + str(i_fold) + "_fit"
                    )
                    predict_data.attrs["name"] = (
                        predict_data.attrs["name"] + "_fold_" + str(i_fold) + "_predict"
                    )
                    model.extend_predict(
                        extend_data=train_data,
                        predict_data=predict_data,
                        save_dir=os.path.join(save_dir, "folds", f"fold_{i_fold}"),
                        **kwargs,
                    )

            return kfold_extend_predict_chunk_fn
        else:

            def extend_predict_chunk_fn(train_data: NormData, predict_data: NormData):
                if predict_data is None:
                    raise ValueError(Output.error(Errors.ERROR_PREDICT_DATA_REQUIRED))
                model.set_save_dir(save_dir)
                model.extend_predict(
                    extend_data=train_data,
                    predict_data=predict_data,
                    save_dir=save_dir,
                    **kwargs,
                )

            return extend_predict_chunk_fn

    def register_fold_indices(
        self, save_dir: str, i_fold: int, indices: tuple[int, int]
    ):
        fit_idx, predict_idx = indices
        os.makedirs(os.path.join(save_dir, "folds", f"fold_{i_fold}"), exist_ok=True)
        with open(
            os.path.join(save_dir, "folds", f"fold_{i_fold}/fit_observations.txt"), "w"
        ) as f:
            f.truncate(0)
            f.write(str(fit_idx))
        with open(
            os.path.join(save_dir, "folds", f"fold_{i_fold}/predict_observations.txt"),
            "w",
        ) as f:
            f.truncate(0)
            f.write(str(predict_idx))

    def save_callable_and_data(
        self,
        job_name: int | str,
        fn: Callable,
        chunk: tuple[NormData] | tuple[NormData, NormData | None],
    ) -> tuple[str, str]:
        python_callable_path = self.get_python_callable_path(job_name)
        data_path = self.get_data_path(job_name)
        os.makedirs(os.path.dirname(self.unique_temp_dir), exist_ok=True)
        with open(python_callable_path, "wb") as f:
            pickle.dump(fn, f)
        with open(data_path, "wb") as f:
            pickle.dump(chunk, f)
        return python_callable_path, data_path

    def get_data_path(self, job_name):
        data_path = os.path.join(self.unique_temp_dir, f"data_{job_name}.pkl")
        return data_path

    def get_python_callable_path(self, job_name):
        python_callable_path = os.path.join(
            self.unique_temp_dir, f"python_callable_{job_name}.pkl"
        )
        return python_callable_path

    def check_job_status(self, job_name: str) -> tuple[bool, bool, Optional[str]]:
        """Check if a job has failed by looking for success file.

        Returns
        -------
        tuple[bool, bool, Optional[str]]
            (is_running, finished_with_error, error_message)
            If job is still running, returns (True, False, None)
            If job finished successfully, returns (False, False, None)
            If job failed, returns (False, True, error_message)
        """
        job_id = self.active_jobs[job_name]

        # For cluster jobs, first check if job is still in queue/running
        is_running = False
        if self.job_type == "slurm":
            result = subprocess.run(
                ["squeue", "-j", job_id], capture_output=True, text=True
            )
            is_running = job_id in result.stdout
        elif self.job_type == "torque":
            result = subprocess.run(["qstat", job_id], capture_output=True, text=True)
            is_running = result.returncode == 0
        if is_running:
            return True, False, None  # Still running

        # Job not running, check for success file
        success_file = os.path.join(self.unique_log_dir, f"{job_name}.success")
        if os.path.exists(success_file):
            return False, False, None  # Finished successfully

        # Job finished but no success file - read error output
        error_file = os.path.join(self.unique_log_dir, f"{job_name}.err")
        if os.path.exists(error_file):
            with open(error_file, "r") as f:
                return False, True, f.read().strip()
        return False, True, "Job failed without error output"

    def check_jobs_status(
        self,
    ) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Check all jobs in active_job_ids for errors.

        Returns
        -------
        tuple[Dict[str, str], Dict[str, str], Dict[str, str]]
            A tuple containing:
            - A dictionary mapping job names to job IDs for running jobs
            - A dictionary mapping job names to error messages for failed jobs
            - A dictionary mapping job names to job IDs for finished jobs
        """
        running_jobs = {}
        finished_jobs = {}
        failed_jobs = {}
        for job_name, job_id in self.active_jobs.items():
            is_running, finished_with_error, error_msg = self.check_job_status(job_name)
            if not is_running:
                if finished_with_error:
                    failed_jobs[job_name] = error_msg
                else:
                    finished_jobs[job_name] = job_id
            elif is_running:
                running_jobs[job_name] = job_id
        return running_jobs, finished_jobs, failed_jobs

    def submit_jobs(
        self,
        fn: Callable,
        first_data_source: NormData,
        second_data_source: Optional[NormData] = None,
        mode: Literal["unary", "binary"] = "unary",
    ) -> None:
        """Submit jobs to the job scheduler.

        The predict_data argument is optional, and if it is not provided, None is passed to the function.

        Parameters
        ----------
        fn : Callable
            Function to call. It should take two arguments.
        fit_data : NormData
            Data to fit the model on
        predict_data : Optional[NormData], optional
            Data to predict on, by default None
        """

        if self.parallelize:
            if self.batch_size is not None and self.n_batches is not None:
                if not (self.n_response_vars // self.n_batches == self.batch_size):
                    raise ValueError(
                        Output.error(
                            Errors.ERROR_BATCH_SIZE_AND_N_BATCHES_MISMATCH,
                            batch_size=self.batch_size,
                            n_batches=self.n_batches,
                            n_response_vars=len(first_data_source.response_vars),
                        )
                    )
            if self.n_batches is None and self.batch_size is not None:
                self.n_batches = len(first_data_source.response_vars) // self.batch_size
            elif self.n_batches is not None and self.batch_size is None:
                self.batch_size = len(first_data_source.response_vars) // self.n_batches
            else:
                raise ValueError("Either n_batches or batch_size must be specified")

            first_chunks = first_data_source.chunk(self.n_batches)
            second_chunks = (
                [None] * self.n_batches
                if second_data_source is None
                else second_data_source.chunk(self.n_batches)
            )

            self.active_jobs.clear()
            self.job_commands.clear()

            for i, (first_chunk, second_chunk) in enumerate(
                zip(first_chunks, second_chunks)
            ):
                job_name = f"{self.task_id}_job_{i}"
                if mode == "unary":
                    chunk_tuple = (first_chunk,)
                else:
                    chunk_tuple = (first_chunk, second_chunk)
                python_callable_path, data_path = self.save_callable_and_data(
                    job_name, fn, chunk_tuple
                )

                command = self.wrap_in_job(job_name, python_callable_path, data_path)

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = process.communicate()
                try:
                    if self.job_type == "slurm":
                        job_id = re.search(r"Submitted batch job (\d+)", stdout).group(
                            1
                        )  # type: ignore
                    elif self.job_type == "torque":
                        job_id = re.search(r"(.*)", stdout).group(1).strip()  # type: ignore
                except AttributeError:
                    raise ValueError(
                        Output.error(
                            Errors.ERROR_SUBMITTING_JOB, job_id=job_name, stderr=stderr
                        )
                    )

                self.active_jobs[job_name] = job_id
                self.job_commands[job_name] = command

        else:
            if mode == "unary":
                chunk_tuple = (first_data_source,)
            elif mode == "binary":
                chunk_tuple = (first_data_source, second_data_source)
            fn(*chunk_tuple)

    def wrap_in_job(self, job_name, python_callable_path, data_path):
        if self.job_type == "slurm":
            command = self.wrap_in_slurm_job(job_name, python_callable_path, data_path)
        elif self.job_type == "torque":
            command = self.wrap_in_torque_job(job_name, python_callable_path, data_path)
        return command

    def wrap_in_slurm_job(
        self, job_name: int | str, python_callable_path: str, data_path: str
    ) -> list[str]:
        job_path, out_file, err_file, success_file = self.get_all_job_file_paths(
            job_name
        )
        current_file_path = os.path.abspath(__file__)
        with open(job_path, "w") as f:
            f.write(
                f"""#!/bin/bash
    
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={self.n_cores}
#SBATCH --time={self.time_limit_str}
#SBATCH --mem={self.memory}
#SBATCH --error={err_file}
#SBATCH --output={out_file}


{self.preamble}
source activate {self.environment}
# Force Python to use unbuffered output
export PYTHONUNBUFFERED=1
# Ensure that it does not spawn multiple threads randomly.
export OMP_NUM_THREADS={self.n_cores}
export MKL_NUM_THREADS={self.n_cores}
export OPENBLAS_NUM_THREADS={self.n_cores}
export NUMEXPR_NUM_THREADS={self.n_cores}
# Force stdout/stderr to be unbuffered
exec 1> >(tee -a {out_file})
exec 2> >(tee -a {err_file})
python {current_file_path} {python_callable_path} {data_path} {self.max_retries} 

exit_code=$?
if [ $exit_code -eq 0 ]; then
    touch {success_file}
fi
exit $exit_code
"""
            )
        return ["sbatch", job_path]

    def get_all_job_file_paths(self, job_name):
        job_path = os.path.join(self.unique_temp_dir, f"{job_name}.sh")
        out_file = os.path.join(self.unique_log_dir, f"{job_name}.out")
        err_file = os.path.join(self.unique_log_dir, f"{job_name}.err")
        success_file = os.path.join(self.unique_log_dir, f"{job_name}.success")
        create_incremental_backup(job_path)
        create_incremental_backup(out_file)
        create_incremental_backup(err_file)
        return job_path, out_file, err_file, success_file

    def wrap_in_torque_job(
        self, job_name: int | str, python_callable_path: str, data_path: str
    ) -> list[str]:
        job_path, out_file, err_file, success_file = self.get_all_job_file_paths(
            job_name
        )
        current_file_path = os.path.abspath(__file__)
        with open(job_path, "w") as f:
            f.write(
                f"""#!/bin/sh

#PBS -N {job_name}
#PBS -l nodes=1:ppn={self.n_cores}
#PBS -l walltime={self.time_limit_str}
#PBS -l mem={self.memory}
#PBS -o {out_file}
#PBS -e {err_file}
#PBS -m a

{self.preamble}
source activate {self.environment}
# Force Python to use unbuffered output
export PYTHONUNBUFFERED=1
# Ensure that it does not spawn multiple threads randomly.
export OMP_NUM_THREADS={self.n_cores}
export MKL_NUM_THREADS={self.n_cores}
export OPENBLAS_NUM_THREADS={self.n_cores}
export NUMEXPR_NUM_THREADS={self.n_cores}
# Force stdout/stderr to be unbuffered
exec 1> >(tee -a {out_file})
exec 2> >(tee -a {err_file})
python {current_file_path} {python_callable_path} {data_path} {self.max_retries} {self.random_sleep_scale}

exit_code=$?
if [ $exit_code -eq 0 ]; then
    touch {success_file}
fi
exit $exit_code
"""
            )

        return ["qsub", job_path]

    def load_model(
        self, fold_index: Optional[int] = 0, into: NormativeModel | None = None
    ) -> NormativeModel:
        if self.cross_validate:
            path = os.path.join(self.save_dir, "folds", f"fold_{fold_index}")
            return NormativeModel.load(path, into=into)
        else:
            return NormativeModel.load(self.save_dir, into=into)

    def load_data(self, data_source: NormData, fold_index: Optional[int] = 0) -> None:
        if self.cross_validate:
            Output.warning(Warnings.LOADING_DATA_NOT_SUPPORTED_FOR_CROSS_VALIDATION)
            # path = os.path.join(self.save_dir, "folds", f"fold_{fold_index}", "results")
            # original_name = data_source.name
            # data_source.name = f"{original_name}_fold_{fold_index}_train"
            # data_source.load_results(path)
            # data_source.name = f"{original_name}_fold_{fold_index}_predict"
            # data_source.load_results(path)
            # data_source.name = original_name
        else:
            data_source.load_results(os.path.join(self.save_dir, "results"))

    def save(self) -> None:
        """Save the runner state to a JSON file in the save directory."""
        runner_state = {
            "active_jobs": self.active_jobs,
            "log_dir": self.log_dir,
            "temp_dir": self.temp_dir,
            "task_id": self.task_id,
            "unique_log_dir": self.unique_log_dir,
            "unique_temp_dir": self.unique_temp_dir,
            "job_type": self.job_type,
            "n_batches": self.n_batches,
            "batch_size": self.batch_size,
            "save_dir": self.save_dir,
            "job_commands": self.job_commands,
            "failed_jobs": self.failed_jobs,
            "environment": self.environment,
        }

        runner_file = os.path.join(self.unique_temp_dir, "runner_state.json")
        create_incremental_backup(runner_file)
        Output.print(Messages.SAVING_RUNNER_STATE, runner_file=runner_file)
        with open(runner_file, "w") as f:
            json.dump(runner_state, f, indent=4)

    @classmethod
    def load_from_state(cls, runner_file: str) -> "Runner":
        """Load a runner from a saved state.

        Parameters
        ----------
        runner_file : str
            Path to the runner state file

        Returns
        -------
        Runner
            A runner instance with the saved state
        """
        Output.print(Messages.LOADING_RUNNER_STATE, runner_file=runner_file)
        with open(runner_file, "r") as f:
            state = json.load(f)

        runner = cls(
            job_type=state["job_type"],
            n_batches=state["n_batches"],
            log_dir=state["log_dir"],
            temp_dir=state["temp_dir"],
        )
        runner.task_id = state["task_id"]
        runner.unique_log_dir = state["unique_log_dir"]
        runner.unique_temp_dir = state["unique_temp_dir"]
        runner.save_dir = state["save_dir"]
        runner.job_commands = state["job_commands"]
        runner.active_jobs = state["active_jobs"]
        runner.environment = state["environment"]
        runner.batch_size = state["batch_size"]
        runner.active_jobs, runner.finished_jobs, runner.failed_jobs = (
            runner.check_jobs_status()
        )
        Output.print(
            Messages.RUNNER_LOADED,
            n_active_jobs=len(runner.active_jobs),
            n_finished_jobs=len(runner.finished_jobs),
            n_failed_jobs=len(runner.failed_jobs),
        )
        return runner

    @classmethod
    def from_args(cls, args: dict) -> "Runner":
        filtered_args = {
            k: v for k, v in args.items() if k in list(cls.__dict__.keys())
        }
        return cls(**filtered_args)

    def re_submit_failed_jobs(self, observe: bool = True) -> None:
        for job_name, command in self.job_commands.items():
            if job_name in self.failed_jobs:
                python_callable_path = self.get_python_callable_path(job_name)
                data_path = self.get_data_path(job_name)
                command = self.wrap_in_job(job_name, python_callable_path, data_path)

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                stdout, stderr = process.communicate()

                try:
                    if self.job_type == "slurm":
                        job_id = re.search(r"Submitted batch job (\d+)", stdout).group(
                            1
                        )  # type: ignore
                    elif self.job_type == "torque":
                        job_id = re.search(r"(.*)", stdout).group(1).strip()  # type: ignore
                except AttributeError:
                    raise ValueError(
                        Output.error(
                            Errors.ERROR_SUBMITTING_JOB, job_id=job_name, stderr=stderr
                        )
                    )

                if job_id:
                    self.active_jobs[job_name] = job_id
                    self.job_commands[job_name] = command

        self.failed_jobs.clear()
        if observe:
            self.job_observer = JobObserver(
                self.active_jobs, self.job_type, self.unique_log_dir, self.task_id
            )
            self.job_observer.wait_for_jobs()
            _, self.finished_jobs, self.failed_jobs = self.check_jobs_status()
            self.merge_result_parts()

        self.save()


def load_and_execute(args):
    """Load a callable and data from a pickle file and execute it.

    Parameters
    ----------
    args : list[str]
        A list of arguments. The first argument is the path to the callable. The second argument is the path to the data. The third argument is the max number of retries.
    """
    retries = int(args[2])
    for i in range(retries + 1):
        # Sleep for a random amount of time.
        # Try to avoid some async access issues.
        time.sleep(random.uniform(0, 0.1))
        try:
            Output.print(Messages.LOADING_CALLABLE, path=args[0])
            with open(args[0], "rb") as executable_path:
                fn = pickle.load(executable_path)
            Output.print(Messages.LOADING_DATA, path=args[1])
            with open(args[1], "rb") as data_path:
                data = pickle.load(data_path)
            Output.print(Messages.EXECUTING_CALLABLE, attempt=i + 1, total=retries + 1)
            # Each job writes its own result part files (no shared-file lock or
            # read); load_results merges them later.
            job_name = os.path.basename(args[0])[len("python_callable_") : -len(".pkl")]
            with result_parts(job_name):
                fn(*data)
            Output.print(
                Messages.EXECUTION_SUCCESSFUL, attempt=i + 1, total=retries + 1
            )
            return
        except Exception as e:
            if i == retries:
                raise e
            else:
                Output.print(
                    Messages.EXECUTION_FAILED, attempt=i + 1, total=retries + 1, error=e
                )
                continue


if __name__ == "__main__":
    load_and_execute(sys.argv[1:])
