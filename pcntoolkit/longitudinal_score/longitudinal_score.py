from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel


class LongitudinalScore(ABC):
    """Abstract base class for longitudinal deviation scores.

    A longitudinal score quantifies whether a subject's change across visits
    departs from the change that is expected under a fitted normative model.
    Concrete subclasses (e.g. ``ZDiffScore``, ``ZGainScore``) implement the
    specific scoring formula in :meth:`score`.

    Parameters
    ----------
    normative_model : NormativeModel
        A fitted normative model.
    norm_data : NormData
        Longitudinal data used to estimate the score's normalisation
        statistics (e.g. the within-subject residual variance). Must contain
        multiple visits per subject, with z-scores already predicted.
    subject_id : str
        Name of the column that identifies subjects. Subject identifiers are
        read from the ``subject_ids`` field of the ``NormData``.
    """

    def __init__(self, normative_model: "NormativeModel", norm_data: "NormData", subject_id: str):
        self.normative_model = normative_model
        self.norm_data = norm_data
        self.subject_id = subject_id

        self._check_model_is_fitted(normative_model)
        self._check_is_longitudinal(norm_data)

    @abstractmethod
    def score(self, data: "NormData", subject_id: str | None = None, timepoint_col: str = "visit") -> xr.DataArray:
        """Score subjects in ``data``.

        Returns
        -------
        xr.DataArray
            A ``(subjects, response_vars)`` array of longitudinal scores.
        """

    # ------------------------------------------------------------------ #
    # Shared validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_model_is_fitted(normative_model: "NormativeModel") -> None:
        if not getattr(normative_model, "is_fitted", False):
            raise ValueError(
                "The normative model must be fitted before computing longitudinal scores. "
                "Call model.fit(...) or model.fit_predict(...) first."
            )

    @staticmethod
    def _check_is_predicted(data: "NormData") -> None:
        for var in ("Y", "Yhat", "Z"):
            if not hasattr(data, var):
                raise ValueError(
                    f"The data is missing '{var}'. Run model.predict(data) so that predictions "
                    "and z-scores are available before computing longitudinal scores."
                )

    @classmethod
    def _check_is_longitudinal(cls, data: "NormData") -> None:
        ids = cls._get_subject_ids(data)
        _, counts = np.unique(ids, return_counts=True)
        if not np.any(counts >= 2):
            raise ValueError(
                "The data appears to be cross-sectional: no subject has two or more timepoints. "
                "Longitudinal scores require multiple visits per subject. Remove cross-sectional "
                "subjects before computing longitudinal scores."
            )

    # ------------------------------------------------------------------ #
    # Shared accessors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_subject_ids(data: "NormData") -> np.ndarray:
        if not hasattr(data, "subject_ids"):
            raise ValueError(
                "The NormData has no 'subject_ids'. Build it with "
                "NormData.from_dataframe(..., subject_ids='<your subject column>')."
            )
        return np.asarray(data.subject_ids.values)

    @staticmethod
    def _get_observation_column(data: "NormData", name: str) -> np.ndarray:
        """Return a per-observation 1D array for ``name``.

        Looks in the batch effects first, then the covariates. This is how a
        timepoint indicator (e.g. ``'visit'``) is resolved from a ``NormData``.
        """
        if hasattr(data, "batch_effect_dims") and name in [str(b) for b in data.batch_effect_dims.values]:
            return np.asarray(data.batch_effects.sel(batch_effect_dims=name).values)
        if hasattr(data, "covariates") and name in [str(c) for c in data.covariates.values]:
            return np.asarray(data.X.sel(covariates=name).values)
        raise ValueError(
            f"Could not find timepoint column '{name}' in the data. Include it as a batch effect "
            "or covariate when building the NormData."
        )

    @classmethod
    def _get_timepoint_values(cls, data: "NormData", timepoint_col: str) -> np.ndarray:
        """Return per-observation timepoint labels used to order a subject's visits.

        Uses ``timepoint_col`` when it is present in the data (as a batch effect
        or covariate). Otherwise falls back to the first covariate (typically
        age), which increases monotonically across visits in longitudinal data.
        """
        try:
            values = cls._get_observation_column(data, timepoint_col)
        except ValueError:
            ordering_covariate = str(data.covariates.values[0])
            values = cls._get_observation_column(data, ordering_covariate)
        return cls._as_sortable(values)

    @staticmethod
    def _ordered_unique(values: np.ndarray) -> np.ndarray:
        """Unique values in order of first appearance."""
        return pd.unique(np.asarray(values))

    @staticmethod
    def _as_sortable(values: np.ndarray) -> np.ndarray:
        """Cast timepoint labels to float when possible so visits sort numerically."""
        try:
            return np.asarray(values, dtype=float)
        except (ValueError, TypeError):
            return np.asarray(values)
