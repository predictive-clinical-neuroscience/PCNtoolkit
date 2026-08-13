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
    reference_data : NormData, optional
        Longitudinal reference cohort, usually healthy controls, used to learn
        how much change is typical. Only needed by scores that estimate
        something from a cohort at scoring time, such as ``ZDiffScore``.
    subject_id_col : str, optional
        Name of the subject identifier column used when the data was built.
    """

    def __init__(
        self,
        normative_model: NormativeModel,
        reference_data: NormData | None = None,
        subject_id_col: str | None = None,
    ):
        self.normative_model = normative_model
        self.reference_data = reference_data
        self.subject_id_col = subject_id_col

        # Fail early if the user passed an unfitted model.
        normative_model.check_is_fitted()

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
            Longitudinal cohort to score. Must include numeric visit labels
            via ``NormData.from_dataframe(..., visits='<column>')``.
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.

        Returns
        -------
        xr.DataArray
            One longitudinal score per subject and response variable.
        """

    @staticmethod
    def _ordered_unique(values: np.ndarray) -> np.ndarray:
        """Preserve subject order as it first appears in the data."""
        return pd.unique(np.asarray(values))

    # ------------------------------------------------------------------ #
    # Validation functions
    # ------------------------------------------------------------------ #
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
        ids = data.get_subject_ids()
        visits = data.get_visits()

        # Count rows and distinct visit labels per subject
        grouped = pd.DataFrame({"subject": ids, "visit": visits}).groupby(
            "subject", sort=False
        )["visit"]
        n_rows = grouped.size()
        n_distinct_visits = grouped.nunique()

        # At least one subject must have repeated measurements.
        if not (n_rows >= 2).any():
            raise ValueError(
                "The data appears to have only single-visit observations. "
                "Longitudinal scores require multiple visits per subject. "
                "Remove cross-sectional subjects before scoring."
            )

        # A subject with repeated rows but only one distinct visit label is
        # invalid
        offenders = n_distinct_visits[(n_rows >= 2) & (n_distinct_visits < 2)]
        if len(offenders) > 0:
            raise ValueError(
                f"Subject {offenders.index[0]!r} has multiple rows with "
                "identical visit labels. Longitudinal data requires distinct "
                "visits per subject."
            )
