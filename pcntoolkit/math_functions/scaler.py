"""Data scaling and normalization module for PCNToolkit.

This module provides various scaling implementations for data preprocessing,
including standardization, min-max scaling, robust scaling, and identity scaling.
All scalers implement a common interface defined by the abstract base class Scaler.

The module supports the following scaling operations:
    - Standardization (zero mean, unit variance)
    - Min-max scaling (to [0,1] range)
    - Robust min-max scaling (using percentiles)
    - Identity scaling (no transformation)

Each scaler supports:
    - Fitting to training data
    - Transforming new data
    - Inverse transforming scaled data
    - Serialization to/from dictionaries
    - Optional outlier adjustment

Available Classes
---------------
Scaler : ABC
    Abstract base class defining the scaler interface
StandardScaler
    Standardizes features to zero mean and unit variance
MinMaxScaler
    Scales features to a fixed range [0, 1]
RobustMinMaxScaler
    Scales features using robust statistics based on percentiles
IdentityScaler
    Passes data through unchanged

Examples
--------
>>> from pcntoolkit.dataio.scaler import StandardScaler
>>> import numpy as np
>>> X = np.array([[1, 2], [3, 4], [5, 6]])
>>> scaler = StandardScaler()
>>> X_scaled = scaler.fit_transform(X)

Notes
-----
All scalers support both fitting to the entire dataset and transforming specific
indices of features, allowing for flexible scaling strategies. The scalers can
be serialized to dictionaries for saving/loading trained parameters.

See Also
--------
pcntoolkit.normative_model : Uses scalers for data preprocessing
pcntoolkit.dataio.basis_expansions : Complementary data transformations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

from pcntoolkit.util.output import Errors, Output
from pcntoolkit.util.migration import registry


class Scaler(ABC):
    """Abstract base class for data scaling operations.

    This class defines the interface for all scaling operations in PCNToolkit.
    Concrete implementations must implement fit, transform, inverse_transform,
    and to_dict methods.

    Parameters
    ----------
    adjust_outliers : bool, optional
        Whether to clip values to valid ranges, by default True

    Notes
    -----
    All scaling operations support both fitting to the entire dataset and
    transforming specific indices of features, allowing for flexible scaling
    strategies.
    """

    def __init__(self, adjust_outliers: bool = False) -> None:
        self.adjust_outliers = adjust_outliers

    @abstractmethod
    def fit(self, X: NDArray) -> None:
        """Compute the parameters needed for scaling.

        Parameters
        ----------
        X : NDArray
            Training data to fit the scaler on, shape (n_samples, n_features)
        """

    @abstractmethod
    def transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        """Transform the data using the fitted scaler.

        Parameters
        ----------
        X : NDArray
            Data to transform, shape (n_samples, n_features)
        index : Optional[NDArray], optional
            Indices of features to transform, by default None (transform all)

        Returns
        -------
        NDArray
            Transformed data

        Raises
        ------
        ValueError
            If the scaler has not been fitted
        """

    @abstractmethod
    def inverse_transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        """Inverse transform scaled data back to original scale.

        Parameters
        ----------
        X : NDArray
            Data to inverse transform, shape (n_samples, n_features)
        index : Optional[NDArray], optional
            Indices of features to inverse transform, by default None (transform all)

        Returns
        -------
        NDArray
            Inverse transformed data

        Raises
        ------
        ValueError
            If the scaler has not been fitted
        """

    def fit_transform(self, X: NDArray) -> NDArray:
        """Fit the scaler and transform the data in one step.

        Parameters
        ----------
        X : NDArray
            Data to fit and transform, shape (n_samples, n_features)

        Returns
        -------
        NDArray
            Transformed data
        """
        self.fit(X)
        return self.transform(X)

    @abstractmethod
    def to_dict(self) -> Dict[str, Union[bool, str, float, List[float]]]:
        """Convert scaler instance to dictionary."""

    @classmethod
    def from_dict(
        cls,
        my_dict: Dict[str, Union[bool, str, float, List[float]]],
        version: str | None = None,
    ) -> "Scaler":
        """Create a scaler instance from a dictionary.

        Parameters
        ----------
        my_dict : Dict[str, Union[bool, str, float, List[float]]]
            Dictionary containing scaler parameters.
            Must include 'scaler_type' key.
        version : str | None, optional
            The ptk_version of the saved model, by default None.
            Used to apply any registered Scaler migrations.

        Returns
        -------
        Scaler
            Instance of the appropriate scaler subclass

        Raises
        ------
        ValueError
            If scaler_type is missing or invalid
        """
        # Apply any registered Scaler migrations for this version.
        my_dict = registry.migrate("Scaler", my_dict, version=version)

        if "scaler_type" not in my_dict:
            raise ValueError(Output.error(Errors.ERROR_SCALER_TYPE_NOT_FOUND))

        scaler_type: str = my_dict["scaler_type"]  # type: ignore
        scalers: Dict[str, Type[Scaler]] = {
            "standardize": StandardScaler,
            "minmax": MinMaxScaler,
            "robminmax": RobustMinMaxScaler,
            "id": IdentityScaler,
            "none": IdentityScaler,
        }

        if scaler_type not in scalers:
            raise ValueError(Output.error(Errors.ERROR_UNKNOWN_SCALER_TYPE, scaler_type=scaler_type))

        return scalers[scaler_type].from_dict(my_dict)

    @staticmethod
    def from_string(scaler_type: str, **kwargs: Any) -> "Scaler":
        """Create a scaler instance from a string identifier.

        Parameters
        ----------
        scaler_type : str
            The type of scaling to apply. Options are:
            - "standardize": zero mean, unit variance
            - "minmax": scale to range [0,1]
            - "robminmax": robust minmax scaling using percentiles
            - "id" or "none": no scaling
        **kwargs : dict
            Additional arguments to pass to the scaler constructor

        Returns
        -------
        Scaler
            Instance of the appropriate scaler class
        """
        scalers: Dict[str, Type[Scaler]] = {
            "standardize": StandardScaler,
            "minmax": MinMaxScaler,
            "robminmax": RobustMinMaxScaler,
            "id": IdentityScaler,
            "none": IdentityScaler,
        }

        if scaler_type not in scalers:
            raise ValueError(Output.error(Errors.ERROR_UNKNOWN_SCALER_TYPE, scaler_type=scaler_type))

        return scalers[scaler_type](**kwargs)


class StandardScaler(Scaler):
    """Standardize features by removing the mean and scaling to unit variance.

    This scaler transforms the data to have zero mean and unit variance:
    z = (x - μ) / σ

    Parameters
    ----------
    adjust_outliers : bool, optional
        Whether to clip extreme values, by default True

    Attributes
    ----------
    m : Optional[NDArray]
        Mean of the training data
    s : Optional[NDArray]
        Standard deviation of the training data

    Examples
    --------
    >>> import numpy as np
    >>> from pcntoolkit.dataio.scaler import StandardScaler
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> print(X_scaled.mean(axis=0))  # approximately [0, 0]
    >>> print(X_scaled.std(axis=0))  # approximately [1, 1]
    """

    def __init__(self, adjust_outliers: bool = False) -> None:
        super().__init__(adjust_outliers)
        self.m: Optional[NDArray] = None
        self.s: Optional[NDArray] = None

    def fit(self, X: NDArray) -> None:
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.m = np.mean(X, axis=0)
        self.s = np.std(X, axis=0)

    def transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        if self.m is None or self.s is None:
            raise ValueError(Output.error(Errors.ERROR_SCALER_NOT_FITTED, method="transform"))
        if index is None:
            return (X - self.m) / self.s
        return (X - self.m[index]) / self.s[index]

    def inverse_transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        if self.m is None or self.s is None:
            raise ValueError(Output.error(Errors.ERROR_SCALER_NOT_FITTED, method="inverse_transform"))
        if index is None:
            return X * self.s + self.m
        return X * self.s[index] + self.m[index]

    def to_dict(self) -> Dict[str, Union[bool, str, float, List[float]]]:
        if self.m is None or self.s is None:
            raise ValueError(Output.error(Errors.ERROR_SCALER_NOT_FITTED, method="to_dict"))
        return {
            "scaler_type": "standardize",
            "adjust_outliers": self.adjust_outliers,
            "m": self.m.tolist(),
            "s": self.s.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> "StandardScaler":
        instance = cls(adjust_outliers=bool(my_dict["adjust_outliers"]))
        instance.m = np.array(my_dict["m"])
        instance.s = np.array(my_dict["s"])
        instance.min = np.array(my_dict["min"])
        instance.max = np.array(my_dict["max"])
        return instance


class MinMaxScaler(Scaler):
    """Scale features to a fixed range (0, 1).

    Transforms features by scaling each feature to a given range (default [0, 1]):
    X_scaled = (X - X_min) / (X_max - X_min)

    Parameters
    ----------
    adjust_outliers : bool, optional
        Whether to clip transformed values to [0, 1], by default True

    Attributes
    ----------
    min : Optional[NDArray]
        Minimum value for each feature from training data
    max : Optional[NDArray]
        Maximum value for each feature from training data

    Examples
    --------
    >>> import numpy as np
    >>> from pcntoolkit.dataio.scaler import MinMaxScaler
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = MinMaxScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> print(X_scaled.min(axis=0))  # [0, 0]
    >>> print(X_scaled.max(axis=0))  # [1, 1]
    """

    def __init__(self, adjust_outliers: bool = False) -> None:
        super().__init__(adjust_outliers)
        self.min: Optional[NDArray] = None
        self.max: Optional[NDArray] = None

    def fit(self, X: NDArray) -> None:
        # Add a small epsilon to avoid division by zero
        min = np.min(X, axis=0)
        max = np.max(X, axis=0)
        epsilon = 1e-10
        self.min = min - epsilon
        self.max = max + epsilon

    def transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        if self.min is None or self.max is None:
            raise ValueError(Output.error(Errors.ERROR_SCALER_NOT_FITTED, method="transform"))
        if index is None:
            X_scaled = (X - self.min) / (self.max - self.min)
        else:
            X_scaled = (X - self.min[index]) / (self.max[index] - self.min[index])

        if self.adjust_outliers:
            X_scaled = np.clip(X_scaled, 0, 1)
        return X_scaled

    def inverse_transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        if self.min is None or self.max is None:
            raise ValueError(Output.error(Errors.ERROR_SCALER_NOT_FITTED, method="inverse_transform"))
        if index is None:
            return X * (self.max - self.min) + self.min
        return X * (self.max[index] - self.min[index]) + self.min[index]

    def to_dict(self) -> Dict[str, Union[bool, str, float, List[float]]]:
        if self.min is None or self.max is None:
            raise ValueError(Output.error(Errors.ERROR_SCALER_NOT_FITTED, method="to_dict"))
        return {
            "scaler_type": "minmax",
            "adjust_outliers": self.adjust_outliers,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> "MinMaxScaler":
        instance = cls(adjust_outliers=bool(my_dict["adjust_outliers"]))
        instance.min = np.array(my_dict["min"])
        instance.max = np.array(my_dict["max"])
        return instance


class RobustMinMaxScaler(MinMaxScaler):
    """Scale features using robust statistics based on percentiles.

    Similar to MinMaxScaler but uses percentile-based statistics to be
    robust to outliers.

    Parameters
    ----------
    adjust_outliers : bool, optional
        Whether to clip transformed values to [0, 1], by default True
    tail : float, optional
        The percentile to use for computing robust min/max, by default 0.05
        (5th and 95th percentiles)

    Attributes
    ----------
    min : Optional[NDArray]
        Robust minimum for each feature from training data
    max : Optional[NDArray]
        Robust maximum for each feature from training data
    tail : float
        The percentile value used for robust statistics

    Examples
    --------
    >>> import numpy as np
    >>> from pcntoolkit.dataio.scaler import RobustMinMaxScaler
    >>> X = np.array([[1, 2], [3, 4], [100, 6]])  # with outlier
    >>> scaler = RobustMinMaxScaler(tail=0.1)
    >>> X_scaled = scaler.fit_transform(X)
    """

    def __init__(self, adjust_outliers: bool = False, tail: float = 0.05) -> None:
        super().__init__(adjust_outliers)
        self.tail = tail

    def fit(self, X: NDArray) -> None:
        reshape = len(X.shape) == 1
        if reshape:
            X = X.reshape(-1, 1)

        self.min = np.zeros(X.shape[1])
        self.max = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            sorted_vals = np.sort(X[:, i])
            lower_idx = int(np.round(X.shape[0] * self.tail))
            upper_idx = -int(np.round(X.shape[0] * self.tail))
            self.min[i] = np.median(sorted_vals[0:lower_idx])
            self.max[i] = np.median(sorted_vals[upper_idx:])

    def to_dict(self) -> Dict[str, Union[bool, str, float, List[float]]]:
        if self.min is None or self.max is None:
            raise ValueError(Output.error(Errors.ERROR_SCALER_NOT_FITTED, method="to_dict"))
        return {
            "scaler_type": "robminmax",
            "adjust_outliers": self.adjust_outliers,
            "tail": self.tail,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> "RobustMinMaxScaler":
        instance = cls(
            adjust_outliers=bool(my_dict["adjust_outliers"]),
            tail=float(my_dict["tail"]),  # type: ignore
        )
        instance.min = np.array(my_dict["min"])
        instance.max = np.array(my_dict["max"])
        return instance


class IdentityScaler(Scaler):
    """A scaler that returns the input unchanged.

    This scaler is useful as a placeholder when no scaling is desired
    but a scaler object is required by the API.

    Parameters
    ----------
    adjust_outliers : bool, optional
        Has no effect for this scaler, included for API compatibility

    Examples
    --------
    >>> import numpy as np
    >>> from pcntoolkit.dataio.scaler import IdentityScaler
    >>> X = np.array([[1, 2], [3, 4]])
    >>> scaler = IdentityScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> np.array_equal(X, X_scaled)
    True
    """

    def fit(self, X: NDArray) -> None:
        self.min = np.min(X)
        self.max = np.max(X)
        pass

    def transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        return X.copy()

    def inverse_transform(self, X: NDArray, index: Optional[NDArray] = None) -> NDArray:
        return X.copy()

    def to_dict(self) -> Dict[str, Union[bool, str, float, List[float]]]:
        return {
            "scaler_type": "id",
            "adjust_outliers": self.adjust_outliers,
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Union[bool, str, float, List[float]]]) -> "IdentityScaler":
        instance = cls(adjust_outliers=bool(my_dict["adjust_outliers"]))
        instance.min = np.array(my_dict["min"])
        instance.max = np.array(my_dict["max"])
        return instance
