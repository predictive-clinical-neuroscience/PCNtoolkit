from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

# Import heavy toolkit types only during static type checking.
if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel


class LongitudinalScore(ABC):
    """Abstract base class for longitudinal scores.

    A longitudinal score asks whether a subject changed over time more or less
    than a fitted normative model would expect. Concrete subclasses such as
    ``ZDiffScore`` and ``ZGainScore`` define the exact scoring rule.

    Parameters
    ----------
    normative_model : NormativeModel
        A fitted normative model.
    reference_data : NormData
        Longitudinal reference cohort, usually healthy controls, used to learn
        how much change is typical.
    subject_id_col : str
        Name of the subject identifier column used when the data was built.
    """

    def __init__(
        self,
        normative_model: NormativeModel,
        reference_data: NormData,
        subject_id_col: str,
    ):
        self.normative_model = normative_model
        self.reference_data = reference_data
        self.subject_id_col = subject_id_col

        # Fail early if the user passed an unfitted model.
        self._check_model_is_fitted(normative_model)

    @abstractmethod
    def score(
        self,
        test_data: NormData,
        subject_id_col: str | None = None,
        timepoint_col: str = "visit",
    ) -> xr.DataArray:
        """Score subjects in ``test_data``.

        Parameters
        ----------
        test_data : NormData
            Longitudinal cohort to score.
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.
        timepoint_col : str, default "visit"
            Column used to order repeated visits within each subject.

        Returns
        -------
        xr.DataArray
            One longitudinal score per subject and response variable.
        """

    @staticmethod
    def _get_subject_ids(data: NormData) -> np.ndarray:
        # Make sure the dataset carries subject identifiers.
        if not hasattr(data, "subject_ids"):
            # Tell the user how to build the data correctly.
            raise ValueError(
                "The NormData has no 'subject_ids'. Build it with "
                "NormData.from_dataframe(..., "
                "subject_ids='<your subject column>')."
            )
        # Return ids as a plain NumPy array for grouping logic.
        return np.asarray(data.subject_ids.values)

    @staticmethod
    def _get_observation_column(data: NormData, name: str) -> np.ndarray:
        """Return a per-observation 1D array for ``name``.

        Batch effects are checked first, then covariates.
        """
        # First look for the column among batch effects.
        if (
            hasattr(data, "batch_effect_dims")
            and name in [str(b) for b in data.batch_effect_dims.values]
        ):
            # Return the matching batch-effect values per observation.
            return np.asarray(data.batch_effects.sel(
                batch_effect_dims=name).values)
        # If needed, look for the column among covariates.
        if (
            hasattr(data, "covariates")
            and name in [str(c) for c in data.covariates.values]
        ):
            # Return the matching covariate values per observation.
            return np.asarray(data.X.sel(covariates=name).values)
        # Stop when the requested ordering column is absent.
        raise ValueError(
            f"Could not find timepoint column '{name}' in the data. "
            "Include it as a batch effect or covariate when building the "
            "NormData."
        )

    @classmethod
    def _get_timepoint_values(
        cls,
        data: NormData,
        timepoint_col: str,
    ) -> np.ndarray:
        """Return per-observation timepoint labels used to order a subject's 
        visits.

        Use ``timepoint_col`` when it is present. Otherwise fall back to the
        first covariate, usually age, as a practical visit-order proxy.
        """
        # Prefer the user-named timepoint column when available.
        try:
            # Read the requested visit-order column.
            values = cls._get_observation_column(data, timepoint_col)
        # If that column is absent, try a practical fallback.
        except ValueError as error:
            # When visits are not stored explicitly, age is often enough to
            # put measurements into the right time order.
            # Stop if there is nothing left to order visits with.
            if not hasattr(data, "covariates") or len(data.covariates.values) == 0:
                # Explain why visit ordering cannot be recovered.
                raise ValueError(
                    f"Could not find timepoint column '{timepoint_col}' and "
                    "no covariate is available to order visits."
                ) from error
            # Use the first covariate as a simple visit-order proxy.
            ordering_covariate = str(data.covariates.values[0])
            # Read that fallback column from the dataset.
            values = cls._get_observation_column(data, ordering_covariate)
        # Convert labels into a form that sorts cleanly.
        return cls._as_sortable(values)

    @staticmethod
    def _ordered_unique(values: np.ndarray) -> np.ndarray:
        """Unique values in order of first appearance."""
        # Preserve subject order as it first appears in the data.
        return pd.unique(np.asarray(values))

    @staticmethod
    def _as_sortable(values: np.ndarray) -> np.ndarray:
        """Cast labels to float when possible so visits sort numerically."""
        # Try numeric sorting first for ages or numbered visits.
        try:
            # Convert labels to floats when that makes sense.
            return np.asarray(values, dtype=float)
        # Keep original labels when numeric conversion is impossible.
        except (ValueError, TypeError):
            # Return the labels unchanged for lexicographic sorting.
            return np.asarray(values)

    # ------------------------------------------------------------------ #
    # Validation functions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_model_is_fitted(normative_model: NormativeModel) -> None:
        # Reject models that have not been fit yet.
        if not getattr(normative_model, "is_fitted", False):
            # Explain how to create the missing fitted state.
            raise ValueError(
                "The normative model must be fitted before computing "
                "longitudinal scores. Call model.fit(...) or "
                "model.fit_predict(...) first."
            )

    @staticmethod
    def _check_is_predicted(data: NormData) -> None:
        # Check the prediction fields needed by longitudinal scores.
        for var in ("Yhat", "Z"):
            # Stop if a required prediction output is missing.
            if var not in data.data_vars:
                # Tell the user to run prediction before scoring.
                raise ValueError(
                    f"The data is missing '{var}'. "
                    "Run model.predict(data) before computing "
                    "longitudinal scores."
                )

    @classmethod
    def _check_is_longitudinal(cls, data: NormData) -> None:
        # TODO: This is for LONG DATAFRAME, not for WIDE.

        # Read subject ids so we can count visits per person.
        ids = cls._get_subject_ids(data)
        # Count how many observations belong to each subject.
        _, counts = np.unique(ids, return_counts=True)
        # At least one subject must have repeated measurements.
        if not np.any(counts >= 2):
            raise ValueError(
                "The data appears to have only single-visit observations. "
                "Longitudinal scores require multiple visits per subject. "
                "Remove cross-sectional subjects before scoring."
            )
