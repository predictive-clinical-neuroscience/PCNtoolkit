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
        score_data: NormData,
        subject_id_col: str | None = None,
    ) -> xr.DataArray:
        """Score subjects in ``score_data``.

        Parameters
        ----------
        score_data : NormData
            Longitudinal cohort to score. Must include visit labels via
            ``NormData.from_dataframe(..., visits='<column>')``.
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.

        Returns
        -------
        xr.DataArray
            One longitudinal score per subject and response variable.
        """

    @staticmethod
    def _get_subject_ids(data: NormData) -> np.ndarray:
        # Make sure the dataset carries subject identifiers.
        if not hasattr(data, "subject_ids"):
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

        # If no column is found then throw error
        raise ValueError(
            f"Could not find column '{name}' in the data. "
            "Include it as a batch effect or covariate when building the "
            "NormData."
        )

    @staticmethod
    def _get_timepoint_values(data: NormData) -> np.ndarray:
        """Return per-observation visit labels used to order a subject's visits."""
        if "visits" not in data.data_vars:
            raise ValueError(
                "NormData has no visit labels. Build it with "
                "NormData.from_dataframe(..., visits='<visit column>') before "
                "computing longitudinal scores."
            )
        return LongitudinalScore._as_sortable(np.asarray(data.visits.values))

    @staticmethod
    def _ordered_unique(values: np.ndarray) -> np.ndarray:
        """Preserve subject order as it first appears in the data."""
        return pd.unique(np.asarray(values))

    @staticmethod
    def _as_sortable(values: np.ndarray) -> np.ndarray:
        """Cast labels to float when possible so visits sort numerically."""
        # Try numeric sorting first for ages or numbered visits.
        try:
            return np.asarray(values, dtype=float)
        # Keep original labels when numeric conversion is impossible.
        except (ValueError, TypeError):
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
        # Read subject ids so we can count visits per person.
        ids = cls._get_subject_ids(data)
        visits = cls._get_timepoint_values(data)

        # Count how many observations belong to each subject.
        _, counts = np.unique(ids, return_counts=True)
        # At least one subject must have repeated measurements.
        if not np.any(counts >= 2):
            raise ValueError(
                "The data appears to have only single-visit observations. "
                "Longitudinal scores require multiple visits per subject. "
                "Remove cross-sectional subjects before scoring."
            )

        # Repeated rows must correspond to distinct visit labels per subject.
        ages: np.ndarray | None = None
        if hasattr(data, "covariates") and "age" in [
            str(c) for c in data.covariates.values
        ]:
            ages = cls._get_observation_column(data, "age").astype(float)

        for subject in cls._ordered_unique(ids):
            mask = ids == subject
            if mask.sum() < 2:
                continue
            subject_visits = visits[mask]
            if len(np.unique(subject_visits)) < 2:
                raise ValueError(
                    f"Subject {subject!r} has multiple rows with identical "
                    "visit labels. Longitudinal data requires distinct visits "
                    "per subject — check for duplicated wide-format columns."
                )
            if ages is not None:
                subject_ages = ages[mask]
                if len(np.unique(subject_ages)) < 2:
                    raise ValueError(
                        f"Subject {subject!r} has multiple visits with "
                        "identical age values. Longitudinal scores require "
                        "age to differ between visits within a subject."
                    )
