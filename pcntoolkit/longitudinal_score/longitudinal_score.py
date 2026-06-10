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
    specific scoring formula in the function `score`.

    Parameters
    ----------
    normative_model : NormativeModel
        A fitted normative model. The training data may be cross-sectional
        or longitudinal.
    reference_data : NormData
        A longitudinal **reference / calibration cohort** (typically healthy
        controls) used to estimate the normative variability of change. Must
        contain multiple visits per subject and have predictions already
        computed (``model.predict(reference_data)``). This is **not** the
        subjects being scored — it is used solely to calibrate the score
        (e.g. estimate σ² for z-diff or the correlation matrix for z-gain).
        Pass a held-out longitudinal cohort when the model was trained on
        cross-sectional data.
    subject_id_col : str
        Name of the column that identifies subjects inside both
        ``reference_data`` and the ``test_data`` passed to :meth:`score`.
        Subject identifiers are read from the ``subject_ids`` field of each
        ``NormData``.
    """

    def __init__(
        self,
        normative_model: "NormativeModel",
        reference_data: "NormData",
        subject_id_col: str,
    ):
        self.normative_model = normative_model
        self.reference_data = reference_data
        self.subject_id_col = subject_id_col

        self._check_model_is_fitted(normative_model)

    @abstractmethod
    def score(
        self,
        test_data: "NormData",
        subject_id_col: str | None = None,
        timepoint_col: str = "visit",
    ) -> xr.DataArray:
        """Score subjects in ``test_data``.

        Parameters
        ----------
        test_data : NormData
            The **subjects to be scored** — the clinical or held-out cohort
            you want longitudinal deviation scores for. Predictions must
            already be computed (``model.predict(test_data)``).
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.
        timepoint_col : str, default "visit"
            Column (batch effect or covariate) used to order visits within
            each subject.

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
        for var in ("Yhat", "Z"):
            if var not in data.data_vars:
                raise ValueError(
                    f"The data is missing '{var}'. "
                    "Run model.predict(data) before computing "
                    "longitudinal scores."
                )

    @classmethod
    def _check_is_longitudinal(cls, data: "NormData") -> None:
        ids = cls._get_subject_ids(data)
        _, counts = np.unique(ids, return_counts=True)
        if not np.any(counts >= 2):
            raise ValueError(
                "The data appears to have cross-sectional (single time-point) observations. "
                "Longitudinal scores require multiple visits per subject. Remove cross-sectional "
                "subjects from your data before computing longitudinal scores."
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
