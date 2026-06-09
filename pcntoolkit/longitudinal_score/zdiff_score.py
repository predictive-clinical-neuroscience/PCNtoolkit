from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pcntoolkit.regression_model.blr import BLR

from .longitudinal_score import LongitudinalScore

if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData
    from pcntoolkit.normative_model import NormativeModel


class ZDiffScore(LongitudinalScore):
    """Two-visit z-diff score (Rehák Bučková et al., 2025).

    The z-diff score quantifies whether a subject's change between two visits
    differs from the change expected under a fitted normative model. For each
    subject the residual change is computed in warped space::

        Δr = [φ(y₂) − φ(ŷ₂)] − [φ(y₁) − φ(ŷ₁)]

    where ``φ`` is the model's fitted warp (identity for a non-warped BLR), and
    ``y``/``ŷ`` are the observed and predicted values. The score standardises
    this change by the within-subject longitudinal variability::

        z_diff = Δr / sqrt(2·σ²·(1−ρ))

    The denominator ``2·σ²·(1−ρ)`` is estimated directly from the longitudinal
    ``norm_data`` passed at construction as the mean squared residual change
    over its subjects, and is cached per response variable.

    Notes
    -----
    - Requires a **BLR** (or warped BLR) normative model.
    - Supports **at most two timepoints** per subject.
    - The ``norm_data`` provided at construction is general longitudinal test
      data — whether it represents controls, patients, or a mix is the user's
      choice.
    """

    def __init__(self, normative_model: "NormativeModel", norm_data: "NormData", subject_id: str):
        super().__init__(normative_model, norm_data, subject_id)
        self._check_model_is_blr(normative_model)
        # σ²(1−ρ) per response variable, estimated lazily on first score() call.
        self.sigma2_rho: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def score(self, data: "NormData", subject_id: str | None = None, timepoint_col: str = "visit") -> xr.DataArray:
        """Compute the z-diff score for every subject in ``data``.

        Parameters
        ----------
        data : NormData
            Longitudinal data to score, with two visits per subject and
            predictions/z-scores already computed (``model.predict(data)``).
        subject_id : str, optional
            Subject id column name (kept for API symmetry; subject ids are read
            from the ``NormData`` ``subject_ids`` field). Defaults to the value
            passed at construction.
        timepoint_col : str, default "visit"
            Column (batch effect or covariate) used to order the two visits.

        Returns
        -------
        xr.DataArray
            ``(subjects, response_vars)`` z-diff scores. Subjects without
            exactly two timepoints are ``NaN``.
        """
        subject_id = subject_id or self.subject_id
        self._check_is_predicted(data)
        self._check_is_longitudinal(data)
        self._check_at_most_two_timepoints(data)

        response_vars = [str(r) for r in data.response_vars.values]
        subjects = self._ordered_unique(self._get_subject_ids(data))

        scores = np.full((len(subjects), len(response_vars)), np.nan, dtype=float)
        subject_index = {s: i for i, s in enumerate(subjects)}

        for j, rv in enumerate(response_vars):
            denominator = np.sqrt(self._get_sigma2_rho(rv, timepoint_col))
            deltas = self._residual_change(data, rv, timepoint_col)
            for subject, delta in deltas.items():
                scores[subject_index[subject], j] = delta / denominator

        return xr.DataArray(
            scores,
            dims=("subjects", "response_vars"),
            coords={"subjects": subjects, "response_vars": response_vars},
            name="zdiff",
        )

    def warped_residual(self, data: "NormData", responsevar: str) -> np.ndarray:
        """Per-observation residual ``φ(y) − φ(ŷ)`` for one response variable.

        For a warped BLR the observed and predicted values are first mapped to
        the space in which the warp was fitted (using the model's outscaler,
        which is identity if the user chose no scaling) and then warped. For a
        non-warped BLR this reduces to ``y − ŷ`` in the original space.
        """
        blr = self.normative_model[responsevar]
        y = np.asarray(data.Y.sel(response_vars=responsevar).values, dtype=float)
        yhat = np.asarray(data.Yhat.sel(response_vars=responsevar).values, dtype=float)

        if getattr(blr, "warp", None) is not None:
            outscaler = self.normative_model.outscalers[responsevar]
            y = blr.warp.f(outscaler.transform(y), blr.gamma)
            yhat = blr.warp.f(outscaler.transform(yhat), blr.gamma)

        return y - yhat

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _get_sigma2_rho(self, responsevar: str, timepoint_col: str) -> float:
        """Estimate (and cache) ``2·σ²·(1−ρ)`` from the construction ``norm_data``."""
        if responsevar not in self.sigma2_rho:
            self._check_is_predicted(self.norm_data)
            deltas = self._residual_change(self.norm_data, responsevar, timepoint_col)
            if len(deltas) == 0:
                raise ValueError(
                    f"Cannot estimate σ²(1−ρ) for '{responsevar}': no subject in norm_data has "
                    "exactly two timepoints."
                )
            delta_values = np.fromiter(deltas.values(), dtype=float)
            self.sigma2_rho[responsevar] = float(np.mean(delta_values**2))
        return self.sigma2_rho[responsevar]

    def _residual_change(self, data: "NormData", responsevar: str, timepoint_col: str) -> dict:
        """Map each two-visit subject to its warped residual change ``Δr = r₂ − r₁``."""
        residuals = self.warped_residual(data, responsevar)
        subject_ids = self._get_subject_ids(data)
        timepoints = self._get_timepoint_values(data, timepoint_col)

        deltas: dict = {}
        for subject in self._ordered_unique(subject_ids):
            idx = np.where(subject_ids == subject)[0]
            if len(idx) != 2:
                continue
            ordered = idx[np.argsort(timepoints[idx])]
            r1, r2 = residuals[ordered[0]], residuals[ordered[1]]
            deltas[subject] = r2 - r1
        return deltas

    def _check_at_most_two_timepoints(self, data: "NormData") -> None:
        ids = self._get_subject_ids(data)
        _, counts = np.unique(ids, return_counts=True)
        if np.any(counts > 2):
            raise ValueError(
                "ZDiffScore supports at most two timepoints per subject. Some subjects have more "
                "than two visits. Use ZGainScore for three or more timepoints."
            )

    @staticmethod
    def _check_model_is_blr(normative_model: "NormativeModel") -> None:
        template = getattr(normative_model, "template_regression_model", None)
        if not isinstance(template, BLR):
            raise ValueError(
                "ZDiffScore requires a BLR (or warped BLR) normative model. "
                "Use ZGainScore for other regression models."
            )
