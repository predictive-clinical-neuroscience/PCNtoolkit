from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from pathlib import Path

from pcntoolkit.dataio.norm_data import NormData


def design_matrix(bandwidth: int, Sigma: np.ndarray) -> pd.DataFrame:
    """Constructs a design matrix according to: Buuren, S. Evaluation and prediction of individual growth trajectories. Ann. Hum. Biol. 50, 247–257 (2023).

    Args:
        bandwidth (int): The bandwidth for which the covariance has been computed
        Sigma np.ndarray: Covariate matrix with possibly missing values. The 0'th column represents an age of 0.
    Returns:
        pd.DataFrame: A design matrix with regressors and predictors. The matrix may have missing values in the 'y' column.
    """
    max_age = Sigma.shape[0] - 1
    Ages = np.arange(max_age)
    dfs = []
    for offset in range(1, bandwidth + 1):
        ages_i = Ages[: max_age - offset + 1]
        df_i = pd.DataFrame(index=ages_i)
        df_i["v0"] = 1
        df_i["V1"] = np.log(ages_i + (offset / 2))
        df_i["V2"] = np.log(offset)
        df_i["V3"] = 1 / offset
        df_i["V4"] = df_i["V1"] * df_i["V2"]
        df_i["V5"] = df_i["V1"] ** 2
        df_i["y"] = np.diagonal(Sigma, offset)
        dfs.append(df_i)
    df = pd.concat(dfs, axis=0)
    return df


def fill_missing(bandwidth: int, cors: np.ndarray) -> np.ndarray:
    """Fills in missing correlation values according to:

    Args:
        bandwidth (int): the bandwidth within which the indices are filled in
        cors (np.ndarray): possibly incomplete correlation matrix of shape [n_responsevars, n_ages, n_ages]

    Returns:
        np.ndarray: New matrix completed with predicted values
    """
    f_cors = fisher_transform(cors)
    max_age = f_cors.shape[1] - 1
    newcors = np.zeros_like(f_cors)
    # Loop over response variables
    for rv in range(f_cors.shape[0]):
        # Create design matrix
        Phi = design_matrix(bandwidth, f_cors[rv])
        # Drop rows with NaN
        Xy = Phi.dropna(axis=0, inplace=False)
        # Fit regressionmodel to cleaned data
        regmodel = LinearRegression(fit_intercept=False).fit(Xy.drop(columns="y", inplace=False), y=Xy[["y"]])
        # Use that to infer all rows including the rows with NaN
        y_pred = regmodel.predict(Phi.drop(columns="y"))
        # Fill in the predicted correlations
        for i, (age1, age2) in enumerate(offset_indices(max_age, bandwidth)):
            newcors[rv, age1, age2] = newcors[rv, age2, age1] = y_pred[i].item()
    # Inverse Fisher transform (tanh)
    newcors = np.tanh(newcors)
    # Take only the predicted values where there were missing values
    # newcors = np.where(np.isnan(f_cors), newcors, f_cors)
    return newcors


def offset_indices(max_age: int, bandwidth: int):
    """Generate pairs of indices that iterate over all the cells in the upper triangular region specified by the parameters.

    E.g:
    Offset_indices(3, 2) will yield (0,1) -> (0,2) -> (1,2) -> (1,3) -> (2,3)
    Which index these positions:
    _,0,1,_
    _,_,2,3
    _,_,_,4
    _,_,_,_

    Args:
        max_age (int): max age for which indices are generated (includes 0)
        bandwidth (int): the bandwidth within which the indices are computed

    Yields:
        (int, int): pairs of indices
    """
    acc = np.zeros((max_age + 1, max_age + 1))
    acc[np.triu_indices(max_age + 1, 1)] = 1
    acc[np.triu_indices(max_age + 1, bandwidth + 1)] = 0
    for pair in zip(*np.where(acc)):
        yield pair


def fisher_transform(cor):
    epsilon = 1e-13
    cor = np.clip(cor, -1 + epsilon, 1 + epsilon)
    return 0.5 * np.log((1 + cor) / (1 - cor))


def get_correlation_matrix(
    data: NormData,
    bandwidth: int,
    covariate_name: str = "age",
    matrix_path= None,
) -> xr.DataArray:
    """Compute correlations of Z scores between pairs of observations of the same subject at different ages

    Args:
        data (NormData): Data containing covariates, predicted Z-scores, batch effects and subject indices
        bandwidth (int): The age offset range within which correlations are computed
        covariate_name (str, optional): Covariate to use for grouping subjects. Defaults to "age".
        matrix_path (Optional[Union[str, Path]], optional): Path to directory containing precomputed
            velocity_objects.pkl files, one per response variable. If None, computes from data. Defaults to None.

    Returns:
        xr.DataArray: Correlations of shape [n_response_vars, n_ages, n_ages]
    """
    if matrix_path is not None:
        matrix_path = Path(matrix_path)
        if not matrix_path.exists():
            raise FileNotFoundError(f"matrix_path not found: {matrix_path}")

        loaded = {}
        for rv in data.response_vars.to_numpy():
            pkl_path = matrix_path / rv / "velocity_objects.pkl"
            if not pkl_path.exists():
                print(f"Warning: no velocity_objects.pkl for '{rv}', skipping")
                continue
            obj = pd.read_pickle(pkl_path)
            loaded[rv] = obj['A_sparse_predict'].toarray()

        if not loaded:
            raise FileNotFoundError(f"No velocity_objects.pkl found for any response var in {matrix_path}")

        loaded_rvs = list(loaded.keys())
        stacked = np.stack([loaded[rv] for rv in loaded_rvs], axis=0)
        n_ages = stacked.shape[1]
        
        return xr.DataArray(
            stacked,
            dims=("response_vars", f"{covariate_name}_1", f"{covariate_name}_2"),
            coords={
                "response_vars": data.response_vars.to_numpy(),
                f"{covariate_name}_1": np.arange(n_ages),
                f"{covariate_name}_2": np.arange(n_ages),
            },
        )

    # Compute from data
    df = data.to_dataframe()[["X", "Z", "batch_effects", "subject_ids"]].droplevel(level=0, axis=1)
    grps = df.groupby(covariate_name).indices | defaultdict(list)
    max_age = int(max(list(grps.keys())))
    n_responsevars = len(data.response_vars.to_numpy())
    cors = np.tile(np.eye(max_age + 1), (n_responsevars, 1, 1))

    for age1, age2 in offset_indices(max_age, bandwidth):
        merged = pd.merge(df.iloc[grps[age1]], df.iloc[grps[age2]], how="inner", on="subject_ids")
        if len(merged) >= 4:
            for i, rv in enumerate(data.response_vars.to_numpy()):
                cors[i, age2, age1] = cors[i, age1, age2] = merged[f"{rv}_x"].corr(merged[f"{rv}_y"])
        elif age1 != age2:
            cors[:, age2, age1] = cors[:, age1, age2] = np.nan

    newcors = fill_missing(bandwidth, cors)

    return xr.DataArray(
        newcors,
        dims=("response_vars", f"{covariate_name}_1", f"{covariate_name}_2"),
        coords={
            "response_vars": data.response_vars.to_numpy(),
            f"{covariate_name}_1": np.arange(newcors.shape[1]),
            f"{covariate_name}_2": np.arange(newcors.shape[2]),
        },
    )


def get_thrive_Z_X(cors, start_x, start_z, span, z_thrive=1.96):
    assert cors.shape[0] == cors.shape[1]
    padded_cors = np.pad(cors, ((0, span + 1), (0, span + 1)), mode="edge")

    start_x_int = np.round(np.asarray(start_x).ravel()).astype(int)
    end_x_int = start_x_int + span

    # Direct correlation between start age and end age
    this_cors = padded_cors[end_x_int, start_x_int]  # (n_obs,)
    breakpoint()
    end_z = start_z.values * this_cors + np.sqrt(1 - this_cors**2) * z_thrive

    thrive_Z = xr.DataArray(
        np.stack([start_z.values, end_z], axis=1),  # (n_obs, 2)
        dims=("dummy_obs", "offset"),
        coords={"offset": [0, span]}
    )
    thrive_X = xr.DataArray(
        np.stack([start_x_int, end_x_int], axis=1).astype(float),  # (n_obs, 2)
        dims=("dummy_obs", "offset"),
        coords={"offset": [0, span]}
    )
    return thrive_Z, thrive_X

# def get_single_thriveline(cors, acc=0, z_thrive=-1.96):
#     thriveline = np.zeros_like(cors)
#     thriveline = np.append(acc, thriveline)
#     for i, c in enumerate(cors):
#         acc = c * acc + np.sqrt(1 - c**2) * z_thrive
#         thriveline[i+1] = acc
#     return thriveline

# def get_thrivelines(cors, start_age, stop_age, step_age, length_v, start_z, end_z, z_thrive):
#     thrivelines_ = np.zeros(((end_z - start_z) * (stop_age - start_age) // step_age, length_v))
#     ages_ = np.zeros_like(thrivelines_)
#     i = 0
#     for age in range(start_age, stop_age, step_age):
#         cor_slice = cors[age : age + length_v - 1]
#         ages = np.arange(age, age + length_v)
#         for z in range(start_z, end_z):
#             line = get_single_thriveline(cor_slice, acc=z, z_thrive=z_thrive)
#             to_pad = length_v - len(line)
#             thrivelines_[i] = np.pad(line, ((0, to_pad)), mode="edge")
#             ages_[i] = ages
#             i += 1
#     thrivelines_ = np.array(thrivelines_)
#     ages_ = np.array(ages_)
#     return thrivelines_, ages_[: len(thrivelines_)]

# def compute_thrivelines(data, bandwidth=2, age_range=(0, 10, 1),
#                         trajectory_length=3, centile_params=(-3, 3, -1.96)):
#     """Returns (ages, z_scores) as plottable arrays"""
#     cors, newcors = get_correlation_matrix(data, bandwidth)
#     diagonal = np.diagonal(newcors.to_numpy(), offset=-1)
#     age_start, age_end, age_step = age_range
#     z_min, z_max, threshold = centile_params
#     thrive_Z, ages = get_thrivelines(diagonal, age_start, age_end, age_step,
#                                       trajectory_length, z_min, z_max, threshold)
#     shaped_lines = thrive_Z.reshape((-1, trajectory_length))
#     shaped_ages = ages.reshape((-1, trajectory_length))
#     return shaped_ages.T, shaped_lines.T


# def plot_thrivelines(data, **kwargs):
#     ages, z_scores = compute_thrivelines(data, **kwargs)
#     plt.plot(ages, z_scores)
#     plt.show()