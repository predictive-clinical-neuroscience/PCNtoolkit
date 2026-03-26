"""
norm_data module
===============

This module provides functionalities for normalizing and converting different types of data into a NormData object.

The NormData object is an xarray.Dataset that contains the data, covariates, batch effects, and response variables, and it
is used by all the models in the toolkit.
"""

from __future__ import annotations

import copy
import json
import os
from collections import defaultdict
from functools import reduce

# pylint: disable=deprecated-class
from typing import (
    Any,
    Dict,
    Generator,
    Hashable,
    List,
    LiteralString,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# pylint: enable=deprecated-class
import numpy as np
import pandas as pd  # type: ignore
import xarray as xr
from nibabel.loadsave import load
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split  # type: ignore

# import datavars from xarray
from xarray.core.types import DataVars

from filelock import FileLock

from pcntoolkit.dataio.fileio import load
from pcntoolkit.util.output import Messages, Output, Warnings


class NormData(xr.Dataset):
    """A class for handling normative modeling data, extending xarray.Dataset.

    This class provides functionality for loading data for normative modeling.
    It supports various data formats.

    Parameters
    ----------
    name : str
        The name of the dataset
    data_vars : DataVars
        Data variables for the dataset
    coords : Mapping[Any, Any]
        Coordinates for the dataset
    attrs : Mapping[Any, Any] | None, optional
        Additional attributes for the dataset, by default None

    Attributes
    ----------
    X : xr.DataArray
        Covariate data
    y : xr.DataArray
        Response variable data
    batch_effects : xr.DataArray
        Batch effect data
    Z: xr.DataArray
        Z-score data
    centiles: xr.DataArray
        Centile data


    Examples
    --------
    >>> data = NormData.from_dataframe("my_data", df, covariates, batch_effects, response_vars)
    >>> train_data, test_data = data.train_test_split([0.8, 0.2])
    """

    __slots__ = (
        "unique_batch_effects",
        "batch_effect_counts",
        "batch_effect_covariate_ranges",
        "covariate_ranges",
        "real_ids",
        "thrive_covariate",  # Whether the ids are real or synthetic
    )

    def __init__(
        self,
        name: str,
        data_vars: DataVars,
        coords: Mapping[Any, Any],
        attrs: Mapping[Any, Any] | None = None,
    ) -> None:
        """
        Initialize a NormData object.

        Parameters
        ----------
        name : str
            The name of the dataset.
        data_vars : DataVars
            Data variables for the dataset.
        coords : Mapping[Any, Any]
            Coordinates for the dataset.
        attrs : Mapping[Any, Any] | None, optional
            Additional attributes for the dataset, by default None.
        """
        if attrs is None:
            attrs = {}
        attrs["is_scaled"] = False  # type: ignore
        attrs["name"] = name  # type: ignore
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)
        self["batch_effects"] = self["batch_effects"].astype(str)
        attrs["real_ids"] = attrs.get("real_ids", False)
        self.register_batch_effects()
        be_str = (
            "\t" + ("".join([f"\t{be} ({len(self.unique_batch_effects[be])})\n" for be in self.unique_batch_effects])).strip()
        )
        Output.print(
            Messages.DATASET_CREATED,
            name=name,
            n_subjects=len(np.unique(self.subject_ids)),
            n_observations=len(self.observations),
            n_covariates=len(self.covariates),
            n_response_vars=len(self.response_vars),
            n_batch_effects=len(self.unique_batch_effects),
            batch_effects=be_str,
        )

    @classmethod
    def from_ndarrays(
        cls,
        name: str,
        X: np.ndarray,
        Y: np.ndarray,
        batch_effects: np.ndarray | None = None,
        subject_ids: np.ndarray | None = None,
        attrs: Mapping[str, Any] | None = None,
        remove_outliers: bool = False,
        z_threshold: float = 3.0,
        remove_Nan: bool = False,
    ) -> NormData:
        """Create a NormData object from numpy arrays via DataFrame conversion."""
        attrs = attrs or {}

        # Create DataFrame from arrays
        df_data = {}
        for array, key, default_prefix in [
            (X, "covariates", "covariate"),
            (Y, "response_vars", "response_var"),
            (batch_effects, "batch_effect_dims", "batch_effect"),
        ]:
            if array is not None:
                if array.ndim == 1:
                    array = array[:, None]
                names = attrs.get(key, [f"{default_prefix}_{i}" for i in range(array.shape[1])])
                for i, dataname in enumerate(names):
                    df_data[dataname] = array[:, i]

        if subject_ids is not None:
            df_data["subject_ids"] = subject_ids

        return cls.from_dataframe(
            name=name,
            dataframe=pd.DataFrame(df_data),
            covariates=attrs.get("covariates", [f"covariate_{i}" for i in range(X.shape[1])] if X is not None else None),
            batch_effects=attrs.get(
                "batch_effect_dims",
                [f"batch_effect_{i}" for i in range(batch_effects.shape[1])] if batch_effects is not None else None,
            ),
            response_vars=attrs.get("response_vars", [f"response_var_{i}" for i in range(Y.shape[1])] if Y is not None else None),
            subject_ids="subject_ids" if subject_ids is not None else None,
            attrs=attrs,
            remove_outliers=remove_outliers,
            z_threshold=z_threshold,
            remove_Nan=remove_Nan,
        )

    @classmethod
    def from_paths(cls, name: str, covariates_path: str, responses_path: str, batch_effects_path: str, **kwargs) -> "NormData":  # type: ignore
        """
        Load a normative dataset from a dictionary of paths.
        """
        covs = load(covariates_path, **kwargs)
        resp = load(responses_path, **kwargs)
        batch_effects = load(batch_effects_path, **kwargs)
        return cls.from_ndarrays(name, covs, resp, batch_effects, **kwargs)

    @classmethod
    def from_fsl(cls, fsl_folder, config_params) -> "NormData":  # type: ignore
        """
        Load a normative dataset from a FSL file.

        Parameters
        ----------
        fsl_folder : str
            Path to the FSL folder.
        config_params : dict
            Configuration parameters for loading the dataset.

        Returns
        -------
        NormData
            An instance of NormData.
        """
        raise NotImplementedError("from_fsl is not yet implemented.")

    @classmethod
    def from_bids(cls, bids_folder, config_params) -> "NormData":  # type: ignore
        """
        Load a normative dataset from a BIDS dataset.

        Parameters
        ----------
        bids_folder : str
            Path to the BIDS folder.
        config_params : dict
            Configuration parameters for loading the dataset.

        Returns
        -------
        NormData
            An instance of NormData.
        """
        raise NotImplementedError("from_bids is not yet implemented.")

    @classmethod
    def from_xarray(cls, name: str, xarray_dataset: xr.Dataset) -> NormData:
        """
        Load a normative dataset from an xarray dataset.

        Parameters
        ----------
        name : str
            The name of the dataset.
        xarray_dataset : xr.Dataset
            The xarray dataset to load.

        Returns
        -------
        NormData
            An instance of NormData.
        """
        return cls(
            name,
            xarray_dataset.data_vars,
            xarray_dataset.coords,
            xarray_dataset.attrs,
        )

    @classmethod
    def from_netcdf(cls, name: str, netcdf_path: str) -> NormData:
        """
        Load a normative dataset from a netcdf file.

        Parameters
        ----------
        name: str
            The name of the dataset.
        netcdf_path: str
            The path to the netcdf file.

        Returns
        -------
        NormData
            An instance of NormData.
        """
        xr_dset = xr.open_dataset(netcdf_path)

        # Deserialize the attributes.
        for attr in xr_dset.attrs:
            if attr in xr_dset.attrs:
                xr_dset.attrs[attr] = json.loads(xr_dset.attrs[attr])

        if "batch_effect_counts" in xr_dset.attrs and xr_dset.attrs["batch_effect_counts"]:
            # Convert the batch_effect_counts to a defaultdict
            xr_dset.attrs["batch_effect_counts"] = defaultdict(lambda: 0, xr_dset.attrs["batch_effect_counts"])
        return cls.from_xarray(name=name, xarray_dataset=xr_dset)

    # pylint: disable=arguments-differ
    @classmethod
    def from_dataframe(  # type:ignore
        cls,
        name: str,
        dataframe: pd.DataFrame,
        covariates: List[str] | None = None,
        batch_effects: List[str] | None = None,
        response_vars: List[str | LiteralString] | None = None,
        subject_ids: str | None = None,
        remove_Nan: bool = False,
        remove_outliers: bool = False,
        z_threshold: float = 3.0,
        attrs: Mapping[str, Any] | None = None,
    ) -> NormData:
        """
        Load a normative dataset from a pandas DataFrame.

        Parameters
        ----------
        name : str
            The name you want to give to the dataset. Will be used to name saved results.
        dataframe : pd.DataFrame
            The pandas DataFrame to load.
        covariates : List[str]
            The list of column names to be used as covariates in the dataset.
        batch_effects : List[str]
            The list of column names to be used as batch effects in the dataset.
        response_vars : List[str]
            The list of column names to be used as response variables in the dataset.
        subject_ids: str
            The name of the column containing the subject IDs
        attrs : Mapping[str, Any] | None, optional
            Additional attributes for the dataset, by default None.
        remove_Nan: bool
            Wheter or not to remove NAN values from the dataframe before creationg of the class object. By default False

        Returns
        -------
        NormData
            An instance of NormData.
        """

        all_colums = []
        if covariates:
            all_colums += covariates
        if response_vars:
            all_colums += response_vars
            continuous_vars = all_colums.copy()
        if batch_effects:
            all_colums += batch_effects
        if subject_ids:
            all_colums += [subject_ids]
        dataframe = dataframe[all_colums]
        if remove_Nan:
            dataframe = cls.remove_nan(dataframe)
        else:
            if np.sum(dataframe.isna().sum()) > 0:
                Output.warning(Warnings.REMOVE_NAN_SET_TO_FALSE)
        if remove_outliers:
            dataframe = cls.remove_outliers(dataframe, continuous_vars, z_threshold=z_threshold)

        data_vars = {}
        coords = {}
        attrs = attrs or {}

        if subject_ids is not None:
            attrs["real_ids"] = True  # type: ignore
            data_vars["subject_ids"] = (["observations"], dataframe[subject_ids].to_numpy())
        else:
            attrs["real_ids"] = False  # type: ignore
            data_vars["subject_ids"] = (["observations"], list(np.arange(len(dataframe))))

        coords["observations"] = list(np.arange(len(dataframe)))

        if response_vars is not None and len(response_vars) > 0:
            for respvar in response_vars:
                if respvar not in dataframe.columns:
                    dataframe[respvar] = np.nan
            data_vars["Y"] = (["observations", "response_vars"], dataframe[response_vars].to_numpy())
            coords["response_vars"] = response_vars

        if covariates is not None and len(covariates) > 0:
            data_vars["X"] = (["observations", "covariates"], dataframe[covariates].to_numpy())
            coords["covariates"] = covariates

        if batch_effects is not None and len(batch_effects) > 0:
            data_vars["batch_effects"] = (["observations", "batch_effect_dims"], dataframe[batch_effects].to_numpy())
            coords["batch_effect_dims"] = batch_effects
        else:
            # Initialize batch effects as zeros
            data_vars["batch_effects"] = (["observations", "batch_effect_dims"], np.zeros((len(dataframe), 1)))
            coords["batch_effect_dims"] = ["dummy_batch_effect"]

        return cls(
            name,
            data_vars,
            coords,
            attrs,
        )

    def to_netcdf(self, netcdf_path: str) -> None:
        """
        Save the NormData object to a netcdf file.

        Parameters
        ----------
        netcdf_path: str
            The path to the netcdf file.

        Returns
        -------
        None
        """
        ds = self.copy(deep=False)
        # Serialize the attributes using json so that they can be saved to netcdf
        for attr in ds.attrs:
            if attr in ds.attrs:
                ds.attrs[attr] = json.dumps(ds.attrs[attr])
        xr.Dataset.to_netcdf(ds, netcdf_path, invalid_netcdf=False, format="NETCDF4")

    @classmethod
    def remove_nan(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove NaN values from the dataframe.
        """
        cleaned = dataframe.dropna(axis=0)
        Output.print(f"Removed {len(dataframe) - len(cleaned)} NANs")
        return cleaned

    @classmethod
    def remove_outliers(cls, dataframe: pd.DataFrame, continuous_vars: List[str], z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from the dataframe.
        """
        if z_threshold == 0.0:
            return dataframe
        idx = np.full(len(dataframe), True)
        for covar in continuous_vars:
            zscores = stats.zscore(dataframe[covar])
            outliers = np.abs(zscores) > z_threshold
            if sum(outliers) > 0:
                Output.print(f"Removed {sum(outliers)} outliers for {covar}")
            idx = idx & (~outliers)
        Output.print(f"Removed {np.sum(~idx)} outliers")
        return dataframe.loc[idx]

    def merge(self, other: NormData, name: str | None = None) -> NormData:
        """
        Merge two NormData objects.

        Drops all columns that are not present in both datasets.
        """

        new_data_vars = {}
        new_coords = {}

        if self.attrs["real_ids"] or other.attrs["real_ids"]:
            new_data_vars["subject_ids"] = (
                ["observations"],
                list(np.concatenate([self.subject_ids.to_numpy(), other.subject_ids.to_numpy()])),
            )
        else:
            new_data_vars["subject_ids"] = (["observations"], list(np.arange(self.X.shape[0] + other.X.shape[0])))

        new_coords["observations"] = list(np.arange(self.X.shape[0] + other.X.shape[0]))
        covar_intersection = [c for c in self.covariates.to_numpy() if c in other.covariates.to_numpy()]
        respvar_intersection = [r for r in self.response_vars.to_numpy() if r in other.response_vars.to_numpy()]
        batch_effect_dims_intersection = [b for b in self.batch_effect_dims.to_numpy() if b in other.batch_effect_dims.to_numpy()]
        new_coords["covariates"] = covar_intersection
        new_coords["response_vars"] = respvar_intersection
        new_coords["batch_effect_dims"] = batch_effect_dims_intersection

        if hasattr(self, "X") and hasattr(other, "X"):
            new_X = xr.DataArray(
                np.zeros((self.X.shape[0] + other.X.shape[0], len(covar_intersection))),
                dims=["observations", "covariates"],
                coords={"covariates": covar_intersection, "observations": new_coords["observations"]},
            )
            for covar in covar_intersection:
                new_X.loc[{"covariates": covar}] = np.concatenate(
                    [self.X.sel(covariates=covar).values, other.X.sel(covariates=covar).values], axis=0
                )
            new_data_vars["X"] = (["observations", "covariates"], new_X.data)

        if hasattr(self, "Y") and hasattr(other, "Y"):
            new_Y = xr.DataArray(
                np.zeros((new_X.shape[0], len(respvar_intersection))),
                dims=["observations", "response_vars"],
                coords={"response_vars": respvar_intersection, "observations": new_coords["observations"]},
            )
            for respvar in respvar_intersection:
                new_Y.loc[{"response_vars": respvar}] = np.concatenate(
                    [self.Y.sel(response_vars=respvar).values, other.Y.sel(response_vars=respvar).values], axis=0
                )
            new_data_vars["Y"] = (["observations", "response_vars"], new_Y.data)

        if hasattr(self, "Y_harmonized") and hasattr(other, "Y_harmonized"):
            new_Y_harmonized = xr.DataArray(
                np.zeros((new_X.shape[0], len(respvar_intersection))),
                dims=["observations", "response_vars"],
                coords={"response_vars": respvar_intersection, "observations": new_coords["observations"]},
            )
            for respvar in respvar_intersection:
                new_Y_harmonized.loc[{"response_vars": respvar}] = np.concatenate(
                    [self.Y_harmonized.sel(response_vars=respvar).values, other.Y_harmonized.sel(response_vars=respvar).values],
                    axis=0,
                )
            new_data_vars["Y_harmonized"] = (["observations", "response_vars"], new_Y_harmonized.data)

        if hasattr(self, "Z") and hasattr(other, "Z"):
            new_Z = xr.DataArray(
                np.zeros((new_X.shape[0], len(respvar_intersection))),
                dims=["observations", "response_vars"],
                coords={"response_vars": respvar_intersection, "observations": new_coords["observations"]},
            )
            for respvar in respvar_intersection:
                new_Z.loc[{"response_vars": respvar}] = np.concatenate(
                    [self.Z.sel(response_vars=respvar).values, other.Z.sel(response_vars=respvar).values], axis=0
                )
            new_data_vars["Z"] = (["observations", "response_vars"], new_Z.data)

        if hasattr(self, "centiles") and hasattr(other, "centiles"):
            if self.centile.to_numpy() == other.centile.to_numpy():
                new_centiles = xr.DataArray(
                    np.zeros((new_X.shape[0], len(respvar_intersection), len(self.centile.to_numpy()))),
                    dims=["observations", "response_vars", "centile"],
                    coords={
                        "response_vars": respvar_intersection,
                        "centile": self.centile.to_numpy(),
                        "observations": new_coords["observations"],
                    },
                )
                for respvar in respvar_intersection:
                    for centile in self.centile.to_numpy():
                        new_centiles.loc[{"response_vars": respvar, "centile": centile}] = np.concatenate(
                            [
                                self.centiles.sel(response_vars=respvar, centile=centile).values,
                                other.centiles.sel(response_vars=respvar, centile=centile).values,
                            ],
                            axis=0,
                        )
                new_data_vars["centiles"] = (["observations", "response_vars", "centile"], new_centiles.data)
                new_coords["centile"] = self.centile.to_numpy()

        if hasattr(self, "batch_effects") and hasattr(other, "batch_effects"):
            new_batch_effects = xr.DataArray(
                np.zeros((new_X.shape[0], len(batch_effect_dims_intersection))).astype(str),
                dims=["observations", "batch_effect_dims"],
                coords={"batch_effect_dims": batch_effect_dims_intersection, "observations": new_coords["observations"]},
            )
            for batch_effect_dim in batch_effect_dims_intersection:
                new_batch_effects.loc[{"batch_effect_dims": batch_effect_dim}] = np.concatenate(
                    [
                        self.batch_effects.sel(batch_effect_dims=batch_effect_dim).values,
                        other.batch_effects.sel(batch_effect_dims=batch_effect_dim).values,
                    ],
                    axis=0,
                )
            new_data_vars["batch_effects"] = (["observations", "batch_effect_dims"], new_batch_effects.data)

        new_normdata = NormData(
            name=name or (self.attrs["name"] + "_+_" + other.attrs["name"]),
            data_vars=new_data_vars,
            coords=new_coords,
            attrs=self.attrs,
        )
        return new_normdata

    def get_single_batch_effect(self) -> Dict[str, List[str]]:
        """
        Get a single batch effect for each dimension.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary mapping each batch effect dimension to a list containing a single value.
        """
        return {k: [v[0]] for k, v in self.unique_batch_effects.items()}

    def concatenate_string_arrays(self, *arrays: Any) -> np.ndarray:
        """
        Concatenate arrays of strings.

        Parameters
        ----------
        arrays : List[np.ndarray]
            A list of numpy arrays containing strings.

        Returns
        -------
        np.ndarray
            A single concatenated numpy array of strings.
        """
        return reduce(np.char.add, arrays)

    def chunk(self, n_chunks: int) -> Generator[NormData]:
        """
        Split the data into n_chunks with roughly equal number of response variables

        Parameters
        ----------
        n_chunks : int
            The number of chunks to split the data into.

        Returns
        -------
        Generator[NormData]
            A generator of NormData instances.
        """
        for i in range(n_chunks):
            yield self.isel(response_vars=slice(i, None, n_chunks))

    def train_test_split(
        self,
        splits: Tuple[float, ...] | List[float] | float = 0.8,
        split_names: Tuple[str, ...] | None = None,  # type: ignore
        random_state: int = 42,
    ) -> Tuple[NormData, ...]:
        """
        Split the data into training and testing datasets.

        Parameters
        ----------
        splits : Tuple[float, ...] | List[float] | float
            A tuple (train_size, test_size), specifying the proportion of data for each split. Or a float specifying the proportion of data for the train set.
        split_names : Tuple[str, ...] | None, optional
            Names for the splits, by default None.
        random_state: int , optional
            Random state for splits, by default 42.

        Returns
        -------
        Tuple[NormData, ...]
            A tuple containing the training and testing NormData instances.
        """
        if isinstance(splits, float):
            splits = (splits, 1 - splits)
        elif isinstance(splits, list):
            splits = tuple(splits)
        assert isinstance(splits, tuple)
        assert all([isinstance(i, float) for i in splits]), "Splits must be a list of floats"
        assert sum(list(splits)) == 1, "Splits must sum to 1"
        assert len(splits) > 1, "Splits must contain at least two elements"
        batch_effects_stringified = self.concatenate_string_arrays(
            *[self.batch_effects[:, i].astype(str) for i in range(self.batch_effects.shape[1])]
        )
        train_idx, test_idx = train_test_split(
            np.arange(self.X.shape[0]),
            test_size=splits[1],
            random_state=random_state,
            stratify=batch_effects_stringified,
        )
        split1 = self.isel(observations=train_idx)
        split1.attrs = copy.deepcopy(self.attrs)
        split2 = self.isel(observations=test_idx)
        split2.attrs = copy.deepcopy(self.attrs)
        if split_names is not None:
            split1.attrs["name"] = split_names[0]
            split2.attrs["name"] = split_names[1]
        else:
            split1.attrs["name"] = f"{self.attrs['name']}_train"
            split2.attrs["name"] = f"{self.attrs['name']}_test"
        return split1, split2

    def kfold_split(self, k: int) -> Generator[Tuple[ArrayLike[int], ArrayLike[int]], Any, Any]:
        """
        Perform k-fold splitting of the data.

        Parameters
        ----------
        k : int
            The number of folds.

        Returns
        -------
        Generator[Tuple[ArrayLike[int], ArrayLike[int]], Any, Any]
            A generator yielding training and testing indices for each fold.
        """
        # Returns an iterator of (NormData, NormData) objects, split into k folds
        stratified_kfold_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        batch_effects_stringified = self.concatenate_string_arrays(
            *[self.batch_effects[:, i].astype(str) for i in range(self.batch_effects.shape[1])]
        )
        for train_idx, test_idx in stratified_kfold_split.split(self.X, batch_effects_stringified):
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            yield train_idx, test_idx

    def batch_effects_split(
        self,
        batch_effects: Dict[str, List[str]],
        names: Optional[Tuple[str, str]],
    ) -> Tuple[NormData, NormData]:
        """
        Split the data into two datasets, one with the specified batch effects and one without.
        """
        if names is None:
            names = ["selected", "not_selected"]  # type:ignore
        assert names is not None, "names can not be None"
        A = self.select_batch_effects(names[0], batch_effects)
        B = self.select_batch_effects(names[1], batch_effects, invert=True)
        return A, B

    def has_registered_metadata(self) -> bool:
        """
        Check if the batch effect and covariate metadata have been registered and are non-empty.

        Returns
        -------
        bool
            True if all required metadata attributes exist and are not empty, False otherwise.
        """
        required_attrs = [
            "unique_batch_effects",
            "batch_effect_counts",
            "covariate_ranges",
            "batch_effect_covariate_ranges",
        ]

        for attr in required_attrs:
            # Check if attribute exists and is not an empty dict/defaultdict
            if attr not in self.attrs or not self.attrs[attr]:
                return False

        return True

    def register_batch_effects(self) -> None:
        """
        Create a mapping of batch effects to unique values.
        """
        if self.has_registered_metadata():
            return
        my_be: xr.DataArray = self.batch_effects
        # create a dictionary with for each column in the batch effects, a dict from value to int
        self.attrs["unique_batch_effects"] = {}
        self.attrs["batch_effect_counts"] = defaultdict(lambda: 0)
        self.attrs["covariate_ranges"] = {}
        self.attrs["batch_effect_covariate_ranges"] = {}

        # Vectorized implementation using pandas groupby/agg
        be_cols = self.batch_effect_dims.to_numpy()
        be_df = pd.DataFrame(my_be.values, columns=be_cols)

        x_available = "X" in self.data_vars
        if x_available:
            covs = self.covariates.to_numpy()
            X_df = pd.DataFrame(self.X.values, columns=covs)

        for dim in be_cols:
            vc = be_df[dim].value_counts(sort=False)
            self.attrs["unique_batch_effects"][dim] = vc.index.astype(str).tolist()
            self.attrs["batch_effect_counts"][dim] = {str(k): int(v) for k, v in vc.to_dict().items()}
            self.attrs["batch_effect_covariate_ranges"][dim] = {}

            if x_available:
                grouped = X_df.groupby(be_df[dim], sort=False).agg(["min", "max"])
                for u, row in grouped.iterrows():
                    self.attrs["batch_effect_covariate_ranges"][dim][u] = {
                        c: {"min": float(row[(c, "min")]), "max": float(row[(c, "max")])} for c in covs
                    }

        if x_available:
            overall = X_df.agg(["min", "max"])
            for c in covs:
                self.attrs["covariate_ranges"][c] = {
                    "min": float(overall.loc["min", c]),
                    "max": float(overall.loc["max", c]),
                }

    def check_compatibility(self, other: NormData) -> bool:
        """
        Check if the data is compatible with another dataset.

        Parameters
        ----------
        other : NormData
            Another NormData instance to compare with.

        Returns
        -------
        bool
            True if compatible, False otherwise
        """
        missing_covariates = [i for i in other.covariates.values if i not in self.covariates.values]
        if len(missing_covariates) > 0:
            Output.warning(
                Warnings.MISSING_COVARIATES,
                dataset_name=self.name,
                covariates=missing_covariates,
            )

        extra_covariates = [i for i in self.covariates.values if i not in other.covariates.values]
        if len(extra_covariates) > 0:
            Output.warning(
                Warnings.EXTRA_COVARIATES,
                dataset_name=self.name,
                covariates=extra_covariates,
            )

        extra_response_vars = [i for i in self.response_vars.values if i not in other.response_vars.values]
        if len(extra_response_vars) > 0:
            Output.warning(
                Warnings.EXTRA_RESPONSE_VARS,
                dataset_name=self.name,
                response_vars=extra_response_vars,
            )

        if len(missing_covariates) > 0 or len(extra_covariates) > 0:
            return False

        return self.unique_batch_effects.keys() == other.unique_batch_effects.keys()

    def make_compatible(self: NormData, other: NormData):
        """Ensures datasets are compatible by merging the batch effects maps"""
        myu = self.unique_batch_effects
        otu = other.unique_batch_effects
        all_unique_batch_effects = {dim: list(set(val).union(set(otu[dim]))) for dim, val in myu.items()}

        mycr = self.covariate_ranges
        otcr = other.covariate_ranges
        ncr = {
            cov: {"min": min(mycr[cov]["min"], otcr[cov]["min"]), "max": max(mycr[cov]["max"], otcr[cov]["max"])}
            for cov in self.covariates.to_numpy()
        }

        mybecr = self.batch_effect_covariate_ranges
        otbecr = other.batch_effect_covariate_ranges
        nbecr = {}
        for dim, uniques in myu.items():
            nbecr[dim] = {}
            for u in uniques:
                nbecr[dim][u] = {}
                for c in self.covariates.to_numpy():
                    match (u in mybecr[dim], u in otbecr[dim]):
                        case True, True:
                            nbecr[dim][u][c] = {
                                "min": min(mybecr[dim][u][c]["min"], otbecr[dim][u][c]["min"]),
                                "max": max(mybecr[dim][u][c]["max"], otbecr[dim][u][c]["max"]),
                            }
                        case True, False:
                            nbecr[dim][u][c] = mybecr[dim][u][c]
                        case False, True:
                            nbecr[dim][u][c] = otbecr[dim][u][c]
                        case False, False:
                            raise ValueError("This should never happen")

        # Update instance attributes
        self.unique_batch_effects = copy.deepcopy(all_unique_batch_effects)
        other.unique_batch_effects = copy.deepcopy(all_unique_batch_effects)
        self.covariate_ranges = copy.deepcopy(ncr)
        other.covariate_ranges = copy.deepcopy(ncr)
        self.batch_effect_covariate_ranges = copy.deepcopy(nbecr)
        other.batch_effect_covariate_ranges = copy.deepcopy(nbecr)

        # Update xarray attrs dicts to make them in sync.
        self.attrs["unique_batch_effects"] = copy.copy(self.unique_batch_effects)
        other.attrs["unique_batch_effects"] = copy.copy(other.unique_batch_effects)
        self.attrs["covariate_ranges"] = copy.copy(self.covariate_ranges)
        other.attrs["covariate_ranges"] = copy.copy(other.covariate_ranges)
        self.attrs["batch_effect_covariate_ranges"] = copy.copy(self.batch_effect_covariate_ranges)
        other.attrs["batch_effect_covariate_ranges"] = copy.copy(other.batch_effect_covariate_ranges)

    def scale_forward(self, inscalers: Dict[str, Any], outscalers: Dict[str, Any]) -> None:
        """
        Scale the data forward in-place using provided scalers.

        Parameters
        ----------
        inscalers : Dict[str, Any]
            Scalers for the covariate data.
        outscalers : Dict[str, Any]
            Scalers for the response variable data.
        """
        if not self.attrs["is_scaled"]:
            # Scale X column-wise using the inscalers
            if "X" in self.data_vars:
                scaled_X = np.zeros(self.X.shape)
                for i, covariate in enumerate(self.covariates.to_numpy()):
                    scaled_X[:, i] = inscalers[covariate].transform(self.X.sel(covariates=covariate).data)
                self["X"] = xr.DataArray(
                    scaled_X,
                    coords=self.X.coords,
                    dims=self.X.dims,
                    attrs=self.X.attrs,
                )
            if "thrive_X" in self.data_vars:
                self["thrive_X"] = xr.DataArray(
                    inscalers[self.attrs["thrive_covariate"]].transform(self.thrive_X.data),
                    coords=self.thrive_X.coords,
                    dims=self.thrive_X.dims,
                    attrs=self.thrive_X.attrs,
                )
            # Scale y column-wise using the outscalers
            if "Y" in self.data_vars:
                scaled_y = np.zeros(self.Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_y[:, i] = outscalers[responsevar].transform(self.Y.sel(response_vars=responsevar).data)
                self["Y"] = xr.DataArray(
                    scaled_y,
                    coords=self.Y.coords,
                    dims=self.Y.dims,
                    attrs=self.Y.attrs,
                )
            if "Y_harmonized" in self.data_vars:
                scaled_Y_harmonized = np.zeros(self.Y_harmonized.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_Y_harmonized[:, i] = outscalers[responsevar].transform(
                        self.Y_harmonized.sel(response_vars=responsevar).data
                    )
                self["Y_harmonized"] = xr.DataArray(
                    scaled_Y_harmonized,
                    coords=self.Y_harmonized.coords,
                    dims=self.Y_harmonized.dims,
                    attrs=self.Y_harmonized.attrs,
                )

            if "Yhat" in self.data_vars:
                scaled_Yhat = np.zeros(self.Yhat.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_Yhat[:, i] = outscalers[responsevar].transform(self.Yhat.sel(response_vars=responsevar).data)
                self["Yhat"] = xr.DataArray(
                    scaled_Yhat,
                    coords=self.Yhat.coords,
                    dims=self.Yhat.dims,
                    attrs=self.Yhat.attrs,
                )

            if "centiles" in self.data_vars:
                scaled_centiles = np.zeros(self.centiles.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_centiles[:, :, i] = outscalers[responsevar].transform(
                        self.centiles.sel(response_vars=responsevar).data
                    )
                self["centiles"] = xr.DataArray(
                    scaled_centiles,
                    coords=self.centiles.coords,
                    dims=self.centiles.dims,
                    attrs=self.centiles.attrs,
                )
            if "thrive_Y" in self.data_vars:
                scaled_thrive_Y = np.zeros(self.thrive_Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    scaled_thrive_Y[:, i, :] = outscalers[responsevar].transform(
                        self.thrive_Y.sel(response_vars=responsevar).data
                    )
                self["thrive_Y"] = xr.DataArray(
                    scaled_thrive_Y,
                    coords=self.thrive_Y.coords,
                    dims=self.thrive_Y.dims,
                    attrs=self.thrive_Y.attrs,
                )
            self.attrs["is_scaled"] = True

    def scale_backward(self, inscalers: Dict[str, Any], outscalers: Dict[str, Any]) -> None:
        """
        Scale the data backward using provided scalers.

        Parameters
        ----------
        inscalers : Dict[str, Any]
            Scalers for the covariate data.
        outscalers : Dict[str, Any]
            Scalers for the response variable data.
        """
        if self.attrs["is_scaled"]:
            if "X" in self.data_vars:
                unscaled_X = np.zeros(self.X.shape)
                for i, covariate in enumerate(self.covariates.to_numpy()):
                    unscaled_X[:, i] = inscalers[covariate].inverse_transform(self.X.sel(covariates=covariate).data)
                self["X"] = xr.DataArray(
                    unscaled_X,
                    coords=self.X.coords,
                    dims=self.X.dims,
                    attrs=self.X.attrs,
                )
            if "thrive_X" in self.data_vars:
                self["thrive_X"] = xr.DataArray(
                    inscalers[self.attrs["thrive_covariate"]].inverse_transform(self.thrive_X.data),
                    coords=self.thrive_X.coords,
                    dims=self.thrive_X.dims,
                    attrs=self.thrive_X.attrs,
                )
            if "Y" in self.data_vars:
                unscaled_y = np.zeros(self.Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_y[:, i] = outscalers[responsevar].inverse_transform(self.Y.sel(response_vars=responsevar).data)
                self["Y"] = xr.DataArray(
                    unscaled_y,
                    coords=self.Y.coords,
                    dims=self.Y.dims,
                    attrs=self.Y.attrs,
                )

            if "Y_harmonized" in self.data_vars:
                unscaled_Y_harmonized = np.zeros(self.Y_harmonized.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_Y_harmonized[:, i] = outscalers[responsevar].inverse_transform(
                        self.Y_harmonized.sel(response_vars=responsevar).data
                    )
                self["Y_harmonized"] = xr.DataArray(
                    unscaled_Y_harmonized,
                    coords=self.Y_harmonized.coords,
                    dims=self.Y_harmonized.dims,
                    attrs=self.Y_harmonized.attrs,
                )

            if "Yhat" in self.data_vars:
                unscaled_Yhat = np.zeros(self.Yhat.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_Yhat[:, i] = outscalers[responsevar].inverse_transform(self.Yhat.sel(response_vars=responsevar).data)
                self["Yhat"] = xr.DataArray(
                    unscaled_Yhat,
                    coords=self.Yhat.coords,
                    dims=self.Yhat.dims,
                    attrs=self.Yhat.attrs,
                )

            if "centiles" in self.data_vars:
                unscaled_centiles = np.zeros(self.centiles.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_centiles[:, :, i] = outscalers[responsevar].inverse_transform(
                        self.centiles.sel(response_vars=responsevar).data
                    )
                self["centiles"] = xr.DataArray(
                    unscaled_centiles,
                    coords=self.centiles.coords,
                    dims=self.centiles.dims,
                    attrs=self.centiles.attrs,
                )

            if "thrive_Y" in self.data_vars:
                unscaled_thrive_Y = np.zeros(self.thrive_Y.shape)
                for i, responsevar in enumerate(self.response_vars.to_numpy()):
                    unscaled_thrive_Y[:, i, :] = outscalers[responsevar].inverse_transform(
                        self.thrive_Y.sel(response_vars=responsevar).data
                    )
                self["thrive_Y"] = xr.DataArray(
                    unscaled_thrive_Y,
                    coords=self.thrive_Y.coords,
                    dims=self.thrive_Y.dims,
                    attrs=self.thrive_Y.attrs,
                )

            self.attrs["is_scaled"] = False

    def select_batch_effects(self, name, batch_effects: Dict[str, List[str]], invert: bool = False) -> NormData:
        """
        Select only the specified batch effects.

        Parameters
        ----------
        batch_effects : Dict[str, List[str]]
            A dictionary specifying which batch effects to select.

        Returns
        -------
        NormData
            A NormData instance with the selected batch effects.
        """
        mask = np.zeros(self.batch_effects.shape[0], dtype=bool)
        for key, values in batch_effects.items():
            this_batch_effect = self.batch_effects.sel(batch_effect_dims=key)
            for value in values:
                mask = np.logical_or(mask, this_batch_effect == value)
        if invert:
            mask = ~mask

        to_return = self.where(mask).dropna(dim="observations", how="all")
        if isinstance(to_return, xr.Dataset):
            to_return = NormData.from_xarray(name, to_return)
        to_return.register_batch_effects()
        return to_return

    def to_dataframe(self, dim_order: Sequence[Hashable] | None = None) -> pd.DataFrame:
        """
        Convert the NormData instance to a pandas DataFrame.

        Parameters
        ----------
        dim_order : Sequence[Hashable] | None, optional
            The order of dimensions for the DataFrame, by default None.

        Returns
        -------
        pd.DataFrame
            A DataFrame representation of the NormData instance.
        """
        acc = []
        x_columns = [col for col in ["X"] if hasattr(self, col)]
        y_columns = [col for col in ["Y", "Y_harmonized", "Z"] if hasattr(self, col)]
        acc.append(
            xr.Dataset.to_dataframe(self[x_columns], dim_order)
            .reset_index(drop=False)
            .pivot(index="observations", columns="covariates", values=x_columns)
        )
        acc.append(
            xr.Dataset.to_dataframe(self[y_columns], dim_order)
            .reset_index(drop=False)
            .pivot(index="observations", columns="response_vars", values=y_columns)
        )
        be = (
            xr.DataArray.to_dataframe(self.batch_effects, dim_order)
            .reset_index(drop=False)
            .pivot(index="observations", columns="batch_effect_dims", values="batch_effects")
        )
        be.columns = [("batch_effects", col) for col in be.columns]

        acc.append(be)

        subject_ids = xr.DataArray.to_dataframe(self.subject_ids, dim_order)[["subject_ids"]]
        subject_ids.columns = [
            ("subject_ids", "subject_ids"),
        ]
        acc.append(subject_ids)
        if hasattr(self, "centiles"):
            centiles = (
                xr.DataArray.to_dataframe(self.centiles, dim_order)
                .reset_index(drop=False)
                .pivot(
                    index="observations",
                    columns=["response_vars", "centile"],
                    values="centiles",
                )
            )
            centiles.columns = [("centiles", col) for col in centiles.columns]
            acc.append(centiles)
        pandas_df = pd.concat(acc, axis=1)
        pandas_df.index = self.observations.values.astype(str)
        return pandas_df

    def save_zscores(self, save_dir: str) -> None:
        zdf = self.Z.to_dataframe().unstack(level="response_vars")
        zdf.columns = zdf.columns.droplevel(0)
        zdf = zdf.merge(self.subject_ids.to_dataframe(), on="observations", how="left")
        zdf = zdf[["subject_ids", *[z for z in sorted(zdf.columns.tolist()) if z not in ["subject_ids"]]]]
        zdf.index = zdf.index.astype(str)
        res_path = os.path.join(save_dir, f"Z_{self.name}.csv")
        lock_path = res_path + ".lock"
        with FileLock(lock_path):
            with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
                f.seek(0)
                old_results = pd.read_csv(f) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    old_results["observations"] = old_results["observations"].astype(str)
                    old_results.set_index(["observations"], inplace=True)
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(zdf, on="observations", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = zdf
                f.seek(0)
                f.truncate()
                if "observations" in new_results.columns:
                    new_results = new_results.sort_values(
                        by="observations",
                        key=lambda col: pd.to_numeric(col, errors="coerce"),
                    )
                    new_results["observations"] = new_results["observations"].astype(str)
                else:
                    new_results = new_results.sort_index(
                        key=lambda idx: pd.to_numeric(idx, errors="coerce")
                    )
                    new_results.index = new_results.index.astype(str)
                new_results.to_csv(f)

    def load_zscores(self, save_dir) -> None:
        Z_path = os.path.join(save_dir, f"Z_{self.name}.csv")
        if os.path.isfile(Z_path):
            df = pd.read_csv(Z_path)
            non_index_columns = [i for i in list(df.columns) if not i == "observations"]
            self["Z"] = xr.DataArray(
                df[non_index_columns],
                dims=("observations", "response_vars"),
                coords={"observations": df["observations"], "response_vars": non_index_columns},
            )

    def save_centiles(self, save_dir: str) -> None:
        centiles = self.centiles.to_dataframe().unstack(level="response_vars")
        centiles.columns = centiles.columns.droplevel(0)
        centiles.index = centiles.index.set_levels(centiles.index.levels[1].astype(str), level=1)
        subject_ids = self.subject_ids.to_dataframe()
        subject_ids.index = subject_ids.index.astype(str)
        subject_ids.columns = pd.MultiIndex.from_tuples([("subject_ids", "X")], names=["subject_ids", "centile"])
        for c in self.centile.to_numpy():
            subject_ids[("subject_ids", c)] = subject_ids[("subject_ids", "X")]
        subject_ids = subject_ids.drop(columns=[("subject_ids", "X")])
        subject_ids = subject_ids.stack(level="centile")
        centiles = centiles.merge(subject_ids, on=["observations", "centile"], how="left")
        centiles = centiles[["subject_ids", *[z for z in sorted(centiles.columns.tolist()) if z not in ["subject_ids"]]]]
        res_path = os.path.join(save_dir, f"centiles_{self.name}.csv")
        lock_path = res_path + ".lock"
        with FileLock(lock_path):
            with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
                f.seek(0)
                old_results = pd.read_csv(f) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    old_results["observations"] = old_results["observations"].astype(str)
                    old_results.set_index(["observations", "centile"], inplace=True)
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(centiles, on=["observations", "centile"], how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = centiles
                f.seek(0)
                f.truncate()
                if "observations" in new_results.columns:
                    new_results = new_results.sort_values(
                        by="observations",
                        key=lambda col: pd.to_numeric(col, errors="coerce"),
                    )
                    new_results["observations"] = new_results["observations"].astype(str)
                else:
                    new_results = new_results.sort_index(
                        key=lambda idx: pd.to_numeric(idx, errors="coerce")
                    )
                    # new_results.index = new_results.index.astype(str)
                new_results.to_csv(f)

    def load_centiles(self, save_dir) -> None:
        C_path = os.path.join(save_dir, f"centiles_{self.name}.csv")
        if os.path.isfile(C_path):
            df = pd.read_csv(C_path)
            response_vars = [i for i in list(df.columns) if not (i == "observations" or i == "centile" or i == "subject_ids")]
            centiles = np.unique(df["centile"])
            obs = np.unique(df["observations"])
            obs.sort()
            A = np.zeros((len(centiles), len(obs), len(response_vars)))
            for i, c in enumerate(centiles):
                sub = df[df["centile"] == c]
                sub = sub.sort_values(by="observations")
                for j, rv in enumerate(response_vars):
                    A[i, :, j] = sub[rv]

            self["centiles"] = xr.DataArray(
                A,
                dims=("centile", "observations", "response_vars"),
                coords={"centile": centiles, "observations": obs, "response_vars": response_vars},
            )

    def save_logp(self, save_dir: str) -> None:
        logp = self.logp.to_dataframe().unstack(level="response_vars")
        logp.columns = logp.columns.droplevel(0)
        logp = logp.merge(self.subject_ids.to_dataframe(), on="observations", how="left")
        logp = logp[["subject_ids", *[z for z in sorted(logp.columns.tolist()) if z not in ["subject_ids"]]]]
        logp.index = logp.index.astype(str)
        res_path = os.path.join(save_dir, f"logp_{self.name}.csv")
        lock_path = res_path + ".lock"
        with FileLock(lock_path):
            with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
                f.seek(0)
                old_results = pd.read_csv(f) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    old_results["observations"] = old_results["observations"].astype(str)
                    old_results.set_index(["observations"], inplace=True)
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(logp, on="observations", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = logp
                f.seek(0)
                f.truncate()
                if "observations" in new_results.columns:
                    new_results = new_results.sort_values(
                        by="observations",
                        key=lambda col: pd.to_numeric(col, errors="coerce"),
                    )
                    new_results["observations"] = new_results["observations"].astype(str)
                else:
                    new_results = new_results.sort_index(
                        key=lambda idx: pd.to_numeric(idx, errors="coerce")
                    )
                    new_results.index = new_results.index.astype(str)
                new_results.to_csv(f)

    def load_logp(self, save_dir) -> None:
        logp_path = os.path.join(save_dir, f"logp_{self.name}.csv")
        if os.path.isfile(logp_path):
            df = pd.read_csv(logp_path)
            non_index_columns = [i for i in list(df.columns) if not i == "observations"]
            self["logp"] = xr.DataArray(
                df[non_index_columns],
                dims=("observations", "response_vars"),
                coords={"observations": df["observations"], "response_vars": non_index_columns},
            )

    def save_statistics(self, save_dir: str) -> None:
        mdf = self.statistics.to_dataframe().unstack(level="response_vars")
        mdf.columns = mdf.columns.droplevel(0)
        res_path = os.path.join(save_dir, f"statistics_{self.name}.csv")
        lock_path = res_path + ".lock"
        with FileLock(lock_path):
            with open(res_path, mode="a+" if os.path.exists(res_path) else "w", encoding="utf-8") as f:
                f.seek(0)
                old_results = pd.read_csv(f, index_col=0) if os.path.getsize(res_path) > 0 else None
                if old_results is not None:
                    # Merge on observations, keeping right (new) values for overlapping columns
                    new_results = old_results.merge(mdf, on="statistic", how="outer", suffixes=("_old", ""))
                    # Drop columns ending with '_old' as they're the duplicates from old_results
                    new_results = new_results.loc[:, ~new_results.columns.str.endswith("_old")]
                else:
                    new_results = mdf
                f.seek(0)
                f.truncate()
                new_results.to_csv(f)

    def load_statistics(self, save_dir) -> None:
        logp_path = os.path.join(save_dir, f"statistics_{self.name}.csv")
        if os.path.isfile(logp_path):
            df = pd.read_csv(logp_path)
            df = df.set_index("statistic")
            statistics = list(df.index)
            self["statistics"] = xr.DataArray(
                df.T,
                dims=("response_vars", "statistic"),
                coords={"statistic": statistics},
            )

    def save_results(self, save_dir: str) -> None:
        """Saves the results (zscores, centiles, logp, statistics) to disk

        Args:
            save_dir (str): Where the results are saved. I.e.: {save_dir}/Z_fit_test.csv
        """
        self.save_zscores(save_dir)
        self.save_centiles(save_dir)
        self.save_logp(save_dir)
        self.save_statistics(save_dir)

    def load_results(self, save_dir: str) -> None:
        """Loads the results (zscores, centiles, logp, statistics) back into the data

        Args:
            save_dir (str): Where the results are saved. I.e.: {save_dir}/Z_fit_test.csv
        """
        self.load_zscores(save_dir)
        self.load_centiles(save_dir)
        self.load_logp(save_dir)
        self.load_statistics(save_dir)

    def create_statistics_group(self) -> None:
        """
        Initializes a DataArray for statistics with NaN values.

        This method creates a DataArray with dimensions 'response_vars' and 'statistics',
        where 'response_vars' corresponds to the response variables in the dataset,
        and 'statistics' includes statistics such as Rho, RMSE, SMSE, EXPV, NLL, and ShapiroW.
        The DataArray is filled with NaN values initially.
        """
        rv = self.response_vars.to_numpy().copy().tolist()

        self["statistics"] = xr.DataArray(
            np.nan * np.ones((len(rv), 6)),
            dims=("response_vars", "statistics"),
            coords={
                "response_vars": np.arange(len(rv)),
                "statistics": ["Rho", "Rho_p", "R2", "RMSE", "SMSE", "MSLL", "NLL", "ShapiroW", "MACE", "MAPE", "EXPV"],
            },
        )

    def get_statistics_df(self) -> pd.DataFrame:
        """
        Get the statistics as a pandas DataFrame.
        """
        assert "statistics" in self.data_vars, "Measures DataArray not found"
        statistics_df: pd.DataFrame = (
            self.statistics.to_dataframe()
            .reset_index()
            .pivot(index=["response_vars"], columns=["statistic"], values="statistics")
        )
        return statistics_df

    @property
    def name(self) -> str:
        """
        Get the name of the dataset.

        Returns
        -------
        str
            The name of the dataset.
        """
        return self.attrs["name"]

    @name.setter
    def name(self, name: str) -> None:
        self.attrs["name"] = name

    @property
    def response_var_list(self) -> xr.DataArray:
        return self.response_vars.to_numpy().copy().tolist()
