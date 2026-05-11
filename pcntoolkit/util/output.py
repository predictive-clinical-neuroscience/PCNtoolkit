import os
import warnings
from datetime import datetime


class Messages:
    FITTING_MODELS = "Fitting models on {n_models} response variables."
    FITTING_MODEL = "Fitting model for {model_name}."
    PREDICTING_MODELS = "Making predictions on {n_models} response variables."
    PREDICTING_MODEL = "Making predictions on {model_name}."
    FITTING_AND_PREDICTING_MODELS = "Fitting and predicting {n_models} response variables."
    FITTING_AND_PREDICTING_MODEL = "Fitting and predicting model for {model_name}."
    SAVING_MODEL = "Saving model to:\n\t{save_dir}."
    SAVING_RESULTS = "Saving results to:\n\t{save_dir}."
    SAVING_STATISTICS = "Saving statistics to:\n\t{save_dir}."
    SAVING_CENTILES = "Saving centiles to:\n\t{save_dir}."
    SAVING_ZSCORES = "Saving z-scores to:\n\t{save_dir}."
    TRANSFERRING_MODELS = "Transferring models on {n_models} response variables."
    TRANSFERRING_MODEL = "Transferring model for {model_name}."
    COMPUTING_CENTILES = "Computing centiles for {n_models} response variables."
    COMPUTING_CENTILES_MODEL = "Computing centiles for {model_name}."
    COMPUTING_ZSCORES = "Computing z-scores for {n_models} response variables."
    COMPUTING_ZSCORES_MODEL = "Computing z-scores for {model_name}."
    COMPUTING_LOGP = "Computing log-probabilities for {n_models} response variables."
    COMPUTING_LOGP_MODEL = "Computing log-probabilities for {model_name}."
    NORMATIVE_MODEL_CONFIGURATION_VALID = "Configuration of normative model is valid."
    LOADING_ROI_MASK = "Loading ROI mask ..."
    GENERATING_MASK_AUTOMATICALLY = "Generating mask automatically ..."
    EXTRACTING_CIFTI_SURFACE_DATA = "Extracting cifti surface data to {outstem} ..."
    EXTRACTING_CIFTI_VOLUME_DATA = "Extracting cifti volume data to {niiname} ..."
    BLR_RESTARTING_ESTIMATION_AT_HYP = "Restarting estimation at hyp = {hyp}, due to: {e}"
    BLR_HYPERPARAMETERS_HAVE_NOT_CHANGED = "Hyperparameters have not changed, exiting"
    REGRESSION_MODEL_CONFIGURATION_VALID = "Configuration of regression model is valid."
    JOB_STATUS_MONITOR = """
---------------------------------------------------------
              PCNtoolkit Job Status Monitor ®
---------------------------------------------------------
Task ID: {task_id}
---------------------------------------------------------
Job ID      Name          State      Time      Nodes
---------------------------------------------------------
"""

    NO_MORE_RUNNING_JOBS = """
---------------------------------------------------------
No more running jobs!
---------------------------------------------------------
"""

    JOB_STATUS_SUMMARY = """
---------------------------------------------------------
Total active jobs: {total_active_jobs}
Total completed jobs: {total_completed_jobs}
Total failed jobs: {total_failed_jobs}
---------------------------------------------------------
"""
    JOB_STATUS_LINE = "{:<11} {:<9} {:<10} {:<9} {:<14}"
    NO_PYTHON_PATH_SPECIFIED = "No python path specified. Using interpreter path of current process: {python_path}"
    NO_LOG_DIR_SPECIFIED = "No log directory specified. Using default log directory: {log_dir}"
    NO_TEMP_DIR_SPECIFIED = "No temporary directory specified. Using default temporary directory: {temp_dir}"
    SAVING_RUNNER_STATE = "Saving runner state to:\n\t{runner_file}"
    LOADING_RUNNER_STATE = "Loading runner state from:\n\t{runner_file}"
    RUNNER_LOADED = (
        "Runner loaded\n"
        "---------------------------------------------------------"
        "Active jobs: {n_active_jobs}\n"
        "Finished jobs: {n_finished_jobs}\n"
        "Failed jobs: {n_failed_jobs}\n"
        "---------------------------------------------------------"
    )
    TASK_ID_CREATED = "Task ID created: {task_id}"
    LOG_DIR_CREATED = "Log directory created:\n\t{log_dir}"
    TEMP_DIR_CREATED = "Temporary directory created:\n\t{temp_dir}"
    HARMONIZING_DATA = "Harmonizing data on {n_models} response variables."
    HARMONIZING_DATA_MODEL = "Harmonizing data for {model_name}."
    SYNTHESIZING_DATA = "Synthesizing data for {n_models} response variables."
    SYNTHESIZING_DATA_MODEL = "Synthesizing data for {model_name}."
    LOADING_CALLABLE = "Loading callable from {path}."
    LOADING_DATA = "Loading data from {path}."
    EXECUTING_CALLABLE = "Executing callable, attempt {attempt} of {total}."
    EXECUTION_FAILED = "Execution of callable failed, attempt {attempt} of {total} with error: \n{error}"
    EXECUTION_SUCCESSFUL = "Execution of callable successful, attempt {attempt} of {total}."
    DATASET_CREATED = """Dataset \"{name}\" created.
    - {n_observations} observations
    - {n_subjects} unique subjects
    - {n_covariates} covariates
    - {n_response_vars} response variables
    - {n_batch_effects} batch effects:
    {batch_effects}
    """
    COMPUTING_YHAT = "Computing yhat for {n_models} response variables."
    COMPUTING_YHAT_MODEL = "Computing yhat for {model_name}."
    LOADING_DATA_UNDER_KFOLD_CV = (
        "Automatically loading data under KFold CV is not implemented yet. Please load the data using NormData.load_results."
    )


class Warnings:
    BATCH_SIZE_AND_N_BATCHES_SPECIFIED = "Batch size and number of batches specified. Batch size will be ignored."
    MISSING_COVARIATES = "The dataset {dataset_name} is missing the following covariates: {covariates}"
    EXTRA_COVARIATES = "The dataset {dataset_name} has too many covariates: {covariates}"
    EXTRA_RESPONSE_VARS = "The dataset {dataset_name} has too many response variables: {response_vars}"
    UNKNOWN_BATCH_EFFECTS = "The dataset {dataset_name} has unknown batch effects: {batch_effects}"
    BLR_ESTIMATION_OF_POSTERIOR_DISTRIBUTION_FAILED = (
        "Posterior estimation failed: \n{error}. "
        "\nThe optimizer could not find a stable solution. "
        "Retrying optimization."
    )
    ERROR_GETTING_JOB_STATUSES = "Error getting job statuses: {stderr}"
    ERROR_PARSING_JOB_STATUS_LINE = "Error parsing job status line: {line} - {error}"
    PREDICT_DATA_NOT_USED_IN_KFOLD_CROSS_VALIDATION = "Predict data not used in k-fold cross-validation"
    DIR_DOES_NOT_EXIST = "{dir_attr_str} ({dir_attr}) does not exist, creating it for you."
    BLR_Y_NOT_PROVIDED = "y is not provided, setting self.y to zeros"
    BLR_VAR_X_NOT_PROVIDED = "var_X is not provided, setting self.var_X to zeros"
    BLR_BATCH_EFFECTS_NOT_PROVIDED = "batch_effects is not provided, setting self.batch_effects to zeros"
    HBR_BATCH_EFFECTS_NOT_PROVIDED = BLR_BATCH_EFFECTS_NOT_PROVIDED
    ERROR_SUBMITTING_JOB = "Error submitting job {job_id}: {stderr}"
    NO_RESPONSE_VARS = "No response variables provided for dataset {dataset_name}. Please provide a list of response variables for which you want to fit or predict."
    RESPONSE_VAR_NOT_FOUND = "Response variable {response_var} not found in dataset {dataset_name}. Setting to NaN."
    MULTIPLE_JOBS_FOUND_FOR_JOB_ID = (
        "Multiple jobs found for job ID {job_id}: {job_name}. Please check the job statuses and try again."
    )
    DATA_NOT_SCALED = "Data is not scaled, skipping scaling back to original scale."
    DATA_ALREADY_SCALED = "Data is already scaled, skipping scaling back to original scale."
    NO_COVARIATES = "No covariates provided for dataset {dataset_name}."
    SYNTHESIZE_N_SAMPLES_IGNORED = "{n_samples} sample(s) ignored because data is provided."
    CENTILES_ALREADY_COMPUTED_FOR_CENTILES = "Centiles are already computed for {dataset_name} for centiles {centiles}, skipping computation. Force recompute by passing recompute=True to compute_centiles"
    THRIVELINES_ALREADY_COMPUTED_FOR = "Thrivelines are already computed for {dataset_name} for offsets {offsets}, skipping computation. Force recompute by passing recompute=True to compute_thrivelines"
    REMOVE_NAN_SET_TO_FALSE = (
        "Warning: remove_NAN is set to False. Missing (NaN) values may cause errors during model creation or training."
    )
    REMOVE_NAN_SET_TO_FALSE = (
        "Dataframe contains NaNs, but remove_Nan is set to False. Pass remove_Nan=True to NormData.from_dataframe to remove them."
    )
    SUBJECT_ID_MULTIPLE_COLUMNS = "Subject ID file contains multiple columns. Using the first column for subject IDs."
    SUBJECT_ID_UNEXPECTED_SHAPE = "Subject ID data has an unexpected shape. Expected 1D array or 2D array with one column. Using flattened data or first column."
    MULTIPLE_BATCH_EFFECT_SUMMARY = "Multiple batch effect dimensions found. The summary printout currently uses the first dimension for 'unique_batch_effects' display."
    LOAD_CIFTI_GENERIC_EXCEPTION = "A general exception occurred while loading CIFTI file: {}"
    LOAD_NIFTI_GENERIC_EXCEPTION = "A general exception occurred while loading NIFTI file: {}"
    LOADING_DATA_NOT_SUPPORTED_FOR_CROSS_VALIDATION = (
        "Automatic data loading by the Runner is not supported for cross-validation."
    )
    RENAMED = "'{old_name}' has been renamed to '{new_name}'."
    BLR_CG_NOT_SUPPORTED_WITH_WARP = (
        "The 'cg' optimizer requires analytical gradients, which are not implemented for warped models. "
        "Falling back to the default 'l-bfgs-b' optimizer. "
        "Please set optimizer='l-bfgs-b' to avoid this warning."
    )
    BLR_CG_NOT_SUPPORTED_WITH_HETEROSKEDASTIC = (
        "The 'cg' optimizer requires analytical gradients, which are not implemented for heteroskedastic models. "
        "Falling back to the default 'l-bfgs-b' optimizer. "
        "Please set optimizer='l-bfgs-b' to avoid this warning."
    )
    TRANSFER_DATA_FEWER_BATCH_EFFECTS = (
        "Transfer data contain fewer batch effect levels than training data. "
        "This may lead to biased results."
    )
    MODEL_SAVED_WITH_NEWER_VERSION = (
        "This model was saved with PCNtoolkit v{saved_version}, "
        "but you are running v{current_version}. "
        "Some features may not load correctly. "
        "Please update PCNtoolkit: pip install --upgrade pcntoolkit"
    )
    MODEL_MIGRATION_APPLIED = (
        "This model was saved with PCNtoolkit v{saved_version}, "
        "but you are running v{current_version}. "
        "Loading this model in v{current_version}..."
    )


class Errors:
    ERROR_BATCH_SIZE_AND_N_BATCHES_MISMATCH = "Batch size ({batch_size}) and number of batches ({n_batches}) are both specified, but do not match the number of response variables ({n_response_vars})"
    ERROR_ENVIRONMENT_NOT_FOUND = (
        "Environment {environment} not found. Please specify the path to the python environment using the environment keyword."
    )
    INVALID_ENVIRONMENT = "The python environment {environment} is invalid because it has no /bin/python file. Please specify a valid python environment."
    SAMPLE_BATCH_EFFECTS = (
        "Cannot sample {n_samples} batch effects, because some batch effects have more levels than the number of samples."
    )
    NORMATIVE_MODEL_CONFIGURATION_PROBLEMS = (
        "The following problems have been detected in the normative model configuration:\n{problems}"
    )
    REGRESSION_MODEL_CONFIGURATION_PROBLEMS = (
        "The following problems have been detected in the regression model configuration:\n{problems}"
    )
    UNKNOWN_FILE_TYPE = "Unknown file type: {filename}"
    NO_FLOAT_DATA_TYPE = "Only float data types currently handled, not {data_type}"
    BLR_HYPERPARAMETER_VECTOR_INVALID_LENGTH = "Hyperparameter vector invalid length"
    BLR_MODEL_NOT_FITTED = "Model must be fitted before computing log probabilities"
    ERROR_PARSING_TIME_LIMIT = "Cannot parse {time_limit_str} as time limit"
    ERROR_CROSS_VALIDATION_FOLDS = "If cross-validation is enabled, cv_folds must be greater than 1"
    ERROR_PREDICT_DATA_REQUIRED = "Predict data is required for fit_predict without cross-validation"
    ERROR_SUBMITTING_JOB = "Error submitting job {job_id}: {stderr}"
    BLR_X_NOT_PROVIDED = "X is not provided"
    ERROR_UNKNOWN_FUNCTION = "Unknown function {func}"
    ERROR_ARGUMENT_SPECIFIED_TWICE = "Argument {key} is specified twice."
    ERROR_BLR_CG_NOT_SUPPORTED_WITH_WARP = (
        "The 'cg' optimizer requires analytical gradients, which are not "
        "implemented for warped models. "
        "Please set optimizer='l-bfgs-b' to avoid this error."
    )
    BLR_ERROR_NO_DESIGN_MATRIX_CREATED = "No design matrix created"
    ERROR_UNKNOWN_CLASS = "Unknown class {class_name}"
    ERROR_FILE_NOT_FOUND = "File not found: {path}"
    ERROR_MODEL_NOT_FITTED = "Model needs to be fitted before it can be transferred"
    ERROR_BLR_VAR_X_NOT_PROVIDED = "Variance of covariates (var_X) is required for models with variance."
    ERROR_BLR_PENALTY_NOT_RECOGNIZED = "Requested penalty ({penalty}) not recognized, choose between 'L1' or 'L2'."
    ERROR_BLR_HYPERPARAMETER_VECTOR_INVALID_LENGTH = "Hyperparameter vector invalid length"
    ERROR_BLR_WARPS_NOT_PROVIDED = "A list of warp functions is required"
    ERROR_HBRDATA_X_NOT_PROVIDED = "X must be provided"
    ERROR_UNKNOWN_LIKELIHOOD = "Unsupported likelihood ({likelihood})"
    ERROR_HBR_Y_NOT_PROVIDED = "y must be provided for z-score computation"
    ERROR_HBR_FITTED_BUT_NO_IDATA = "HBR model is fitted but does not have idata. This should not happen."
    ERROR_HBR_COULD_NOT_LOAD_IDATA = "Could not load idata from {path}"
    ERROR_UNKNOWN_MAPPING = "Unknown mapping ({mapping})"
    ERROR_UNKNOWN_DISTRIBUTION = "Unknown distribution ({dist_name})"
    ERROR_SOURCE_ARRAY_NOT_FOUND = "Source array {source_array_name} does not exist in the data."
    ERROR_BASIS_FUNCTION_NOT_FITTED = "Basis function is not fitted. Please fit the basis function first."
    ERROR_DATA_MUST_BE_1D = "Data must be a 1D array or a N-dimensional array with a single column"
    ERROR_BATCH_EFFECTS_NOT_LIST = "Items of the batch_effect dict be a list or a string, not {batch_effect_type}"
    ERROR_SCALER_TYPE_NOT_FOUND = "Dictionary must contain 'scaler_type' key"
    ERROR_UNKNOWN_SCALER_TYPE = "Undefined scaler type: {scaler_type}"
    ERROR_SCALER_NOT_FITTED = "Scaler must be fitted before {method}"
    ERROR_PREDICT_DATA_REQUIRED_FOR_FIT_PREDICT_WITHOUT_CROSS_VALIDATION = (
        "Predict data is required for fit_predict without cross-validation"
    )
    ERROR_NO_ENVIRONMENT_SPECIFIED = (
        "No python environment specified. Please specify the path to the python environment using the environment keyword."
    )
    ERROR_ZSCORES_INVERSE = "Z-scores inverse has invalid shape: Z_shape={Z_shape}, X_shape={X_shape}"
    ERROR_CDF_SHAPE = "CDF shape {cdf_shape} does not match data shape {data_shape}"
    ERROR_Y_NOT_FOUND = "y not found in data"
    HBR_MODEL_NOT_FITTED = "HBR model is not fitted"
    ERROR_BLR_TRANSFER_NOT_IMPLEMENTED = "BLR transfer not implemented"
    ERROR_MULTIPLE_COVARIATE_DIMS = "Multiple covariate dimensions found: {covariate_dims}"
    ERROR_PREDICT_DATA_NOT_SUPPORTED_FOR_CROSS_VALIDATION = (
        "Predict with cross-validation is not supported. Please use fit_predict instead."
    )
    ERROR_WARP_STRING_INVALID = "Invalid warp string: {warp_string}"
    ENSURE_POSITIVE_DISTRIBUTION = "Distribution for {name} needs to be positive."
    OFFSETS_NOT_1D = "Offsets must be a 1-d array or list"
    OFFSET_NOT_VALID = "Invalid list of offsets provided"
    WB_COMMAND_FAILED = "wb_command failed with error: {error}"
    WB_COMMAND_NOT_FOUND = "wb_command not found in PATH"
    ERROR_BLR_POWELL = (
        "Powell optimizer failed. "
        "We recommend running your code again and "
        "setting optimizer='l-bfgs-b'."
    )


class Output:
    _show_messages = True  # Default to showing output
    _show_warnings = True
    _show_pid = True
    _show_timestamp = True

    @classmethod
    def print(cls, message: str, *args, **kwargs) -> None:
        """Print message only if show_messages mode is enabled"""
        if cls._show_messages:
            message = message.format(*args, **kwargs)
            if cls._show_timestamp:
                message = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - " + message
            if cls._show_pid:
                message = "Process: " + str(os.getpid()) + " - " + message
            print(message)

    @classmethod
    def warning(
        cls,
        message: str,
        *args,
        category: type = UserWarning,
        **kwargs,
    ) -> None:
        """Print warning only if show_warnings mode is enabled."""
        if cls._show_warnings:
            message = message.format(*args, **kwargs)
            if cls._show_timestamp:
                message = (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + " - "
                    + message
                )
            if cls._show_pid:
                message = (
                    "Process: " + str(os.getpid()) + " - " + message
                )
            # Emit with the requested warning category
            warnings.warn(message, category)

    @classmethod
    def error(cls, message: str, *args, **kwargs) -> str:
        """Print error message"""
        message = message.format(*args, **kwargs)
        if cls._show_timestamp:
            message = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - " + message
        if cls._show_pid:
            message = "Process: " + str(os.getpid()) + " - " + message
        return message

    @classmethod
    def get_show_pid(cls) -> bool:
        return cls._show_pid

    @classmethod
    def get_show_messages(cls) -> bool:
        return cls._show_messages

    @classmethod
    def get_show_warnings(cls) -> bool:
        return cls._show_warnings

    @classmethod
    def get_show_timestamp(cls) -> bool:
        return cls._show_timestamp

    @classmethod
    def set_show_messages(cls, value: bool) -> None:
        cls._show_messages = value

    @classmethod
    def set_show_pid(cls, value: bool) -> None:
        cls._show_pid = value

    @classmethod
    def set_show_timestamp(cls, value: bool) -> None:
        cls._show_timestamp = value

    @classmethod
    def set_show_warnings(cls, value: bool) -> None:
        cls._show_warnings = value
