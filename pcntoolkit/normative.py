"""
Module providing entry points for fitting and predicting with normative models from the command line.
"""

import argparse

import cloudpickle as pickle  # noqa: F401
import numpy as np

from pcntoolkit.dataio import fileio
from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.util.output import Errors, Output
from pcntoolkit.util.runner import Runner


def fit(conf_dict: dict) -> None:
    """
    Fit a new normative model.

    :param conf_dict: Dictionary containing configuration options
    """

    runner = Runner.from_args(conf_dict)
    fit_data = load_data(conf_dict)
    normative_model: NormativeModel = NormativeModel.from_args(**conf_dict)
    runner.fit(normative_model, fit_data)


def predict(conf_dict: dict) -> None:
    """
    Predict response variables using a saved normative model.

    :param conf_dict: Dictionary containing configuration options
    """
    runner = Runner.from_args(conf_dict)
    predict_data = load_data(conf_dict)
    normative_model = NormativeModel.load(conf_dict["save_dir"])
    runner.predict(normative_model, predict_data)


def fit_predict(conf_dict: dict) -> None:
    """
    Fit a normative model and predict response variables.

    :param conf_dict: Dictionary containing configuration options
    """

    runner = Runner.from_args(conf_dict)
    fit_data = load_data(conf_dict)
    predict_data = load_test_data(conf_dict)

    assert fit_data.check_compatibility(predict_data), (
        "Fit and predict data are not compatible."
    )

    normative_model: NormativeModel = NormativeModel.from_args(**conf_dict)
    runner.fit_predict(normative_model, fit_data, predict_data)


def load_data(conf_dict: dict) -> NormData:
    """Load the data from the configuration dictionary.

    Returns:
        NormData: NormData object containing the data
    """
    respfile = conf_dict.pop("resp")
    covfile = conf_dict.pop("cov")
    maskfile = conf_dict.pop("mask", None)

    X = fileio.load(covfile)
    Y, _ = load_response_vars(respfile, maskfile=maskfile)
    batch_effects = conf_dict.pop("be", None)
    if batch_effects:
        batch_effects = fileio.load(batch_effects)
    else:
        # If no batch effects are specified, create a zero array
        batch_effects = np.zeros((Y.shape[0], 1))
    data = NormData.from_ndarrays("fit_data", X, Y, batch_effects)
    return data


def load_test_data(conf_dict: dict) -> NormData:
    """Load the test data from the file specified in the configuration dictionary.

    Args:
        conf_dict (dict): dictionary containing the configuration options

    Returns:
        NormData: NormData object containing the test data
    """
    respfile = conf_dict.pop("t_resp")
    covfile = conf_dict.pop("t_cov")
    maskfile = conf_dict.pop("mask", None)

    X = fileio.load(covfile)
    Y, _ = load_response_vars(respfile, maskfile=maskfile)
    batch_effects = conf_dict.pop("t_be", None)
    if batch_effects:
        batch_effects = fileio.load(batch_effects)
    else:
        batch_effects = np.zeros((Y.shape[0], 1))

    data = NormData.from_ndarrays("predict_data", X, Y, batch_effects)

    return data


def load_response_vars(
    datafile: str, maskfile: str | None = None, vol: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load response variables from file. This will load the data and mask it if
    necessary. If the data is in ascii format it will be converted into a numpy
    array. If the data is in neuroimaging format it will be reshaped into a
    2D array (observations x variables) and a mask will be created if necessary.

    :param datafile: File containing the response variables
    :param maskfile: Mask file (nifti only)
    :param vol: If True, load the data as a 4D volume (nifti only)
    :returns Y: Response variables
    :returns volmask: Mask file (nifti only)
    """

    if fileio.file_type(datafile) == "nifti":
        dat = fileio.load_nifti(datafile, vol=vol)
        volmask = fileio.create_mask(dat, mask=maskfile)
        Y = fileio.vol2vec(dat, volmask).T
    else:
        Y = fileio.load(datafile)
        volmask = None
        if fileio.file_type(datafile) == "cifti":
            Y = Y.T

    return Y, volmask


def get_argparser() -> argparse.ArgumentParser:
    """Get an argument parser for the normative modeling functions.

    Returns:
        argparse.ArgumentParser: The argument parser
    """
    #  parse arguments
    parser = argparse.ArgumentParser(description="Normative Modeling")
    parser.add_argument("-a", "--alg", help="algorithm", dest="alg", default="gpr")
    parser.add_argument(
        "-f", "--func", help="Function to call", dest="func", default="fit"
    )
    parser.add_argument(
        "-r", "--responses", help="responses file", dest="resp", default=None
    )
    parser.add_argument(
        "-c", "--covariates", help="covariates file", dest="cov", default=None
    )
    parser.add_argument(
        "-t",
        "--test_responses",
        help="responses (test data)",
        dest="t_resp",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--test_covariates",
        help="covariates (test data)",
        dest="t_cov",
        default=None,
    )
    parser.add_argument("-m", "--mask", help="mask file", dest="mask", default=None)
    parser.add_argument(
        "-k", "--cvfolds", help="cross-validation folds", dest="cv_folds", default=None
    )
    return parser


def get_conf_dict_from_args() -> dict[str, str | int | float | bool]:
    """Parse the arguments and return a dictionary with the configuration options.

    Raises:
        ValueError: Raised if an argument is specified twice.

    Returns:
        dict[str, str | int | float | bool]: A dictionary with the configuration option, parsed to the correct type.
    """
    parser = get_argparser()
    known, unknown = parser.parse_known_args()

    conf_dict = {}

    for arg in vars(known):
        conf_dict[arg] = getattr(known, arg)

    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=")
            if key in conf_dict:
                raise ValueError(
                    Output.error(Errors.ERROR_ARGUMENT_SPECIFIED_TWICE, key=key)
                )
            conf_dict[key] = value

    for k, v in conf_dict.items():
        if v:
            try:
                conf_dict[k] = int(v)
            except ValueError:
                try:
                    conf_dict[k] = float(v)
                except ValueError:
                    try:
                        if v.lower() == "true":
                            conf_dict[k] = True
                        elif v.lower() == "false":
                            conf_dict[k] = False
                    except AttributeError:
                        pass
    cv = conf_dict.get("cv_folds", 1)
    cv = cv if cv else 1
    conf_dict["cv_folds"] = cv
    conf_dict["cross_validate"] = cv > 1
    return conf_dict


def make_synthetic_data() -> None:
    """Create synthetic data for testing."""
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    Y = np.random.randn(1000, 2)
    batch_effects = []
    for i in range(2):
        batch_effects.append(np.random.choice(list(range(i + 2)), size=1000))
    batch_effects = np.stack(batch_effects, axis=1)
    np.savetxt("covariates.csv", X)
    np.savetxt("responses.csv", Y)
    np.savetxt("batch_effects.csv", batch_effects)

    X = np.random.randn(1000, 2)
    Y = np.random.randn(1000, 2)
    batch_effects = []
    for i in range(2):
        batch_effects.append(np.random.choice(list(range(i + 2)), size=1000))
    batch_effects = np.stack(batch_effects, axis=1)
    np.savetxt("covariates_test.csv", X)
    np.savetxt("responses_test.csv", Y)
    np.savetxt("batch_effects_test.csv", batch_effects)


def main(*args) -> None:
    """Main function to run the normative modeling functions.

    Raises:
        ValueError: If the function specified in the configuration dictionary is unknown.

    """
    parsed_args = get_conf_dict_from_args()
    match parsed_args.get("func", "fit"):
        case "fit":
            fit(parsed_args)
        case "predict":
            predict(parsed_args)
        case "fit_predict":
            fit_predict(parsed_args)
        case _:
            raise ValueError(
                Output.error(Errors.ERROR_UNKNOWN_FUNCTION, func=parsed_args["func"])
            )


def entrypoint(*args):
    main(*args[1:])


if __name__ == "__main__":
    main()
