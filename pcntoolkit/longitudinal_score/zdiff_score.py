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
    differs from the change expected under a fitted normative model.

        Δr = (y₂ - ŷ₂) - (y₁ - ŷ₁)

    where ``y``/``ŷ`` are the observed and predicted values. The score
    standardises this change by the within-subject longitudinal variability::

        z_diff = Δr / sqrt(2·σ²·(1−ρ))

    Notes
    -----
    The full paper denominator contains two terms: an aleatoric term
    ``sqrt(2·σ²·(1−ρ))`` (expected healthy between-visit variance) and
    an epistemic term ``[Φ(x₂)−Φ(x₁)]ᵀ A⁻¹ [Φ(x₂)−Φ(x₁)]`` (model
    uncertainty from the covariate shift). This implementation omits the
    epistemic term and estimates the aleatoric term empirically as
    ``sqrt(mean(Δr²))`` over ``reference_data``. The epistemic term is
    negligible in practice (order of 10⁻³ vs 10⁻¹), so the simplification
    has minimal impact on results.

    - Requires a **BLR** (or warped BLR) normative model.
    - Supports **at most two timepoints** per subject.
    """

    def __init__(
        self,
        normative_model: "NormativeModel",
        reference_data: "NormData",
        subject_id_col: str,
    ):
        super().__init__(normative_model, reference_data, subject_id_col)
        self._check_model_is_blr(normative_model)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def score(
        self,
        test_data: "NormData",
        subject_id_col: str | None = None,
        timepoint_col: str = "visit",
    ) -> xr.DataArray:
        """Compute the z-diff score for every subject in ``test_data``.

        Parameters
        ----------
        test_data : NormData
            The **subjects to be scored**: longitudinal data with exactly two
            visits per subject and predictions already computed
            (``model.predict(test_data)``).
        subject_id_col : str, optional
            Subject id column name override. Defaults to the value supplied
            at construction.
        timepoint_col : str, default "visit"
            Column (batch effect or covariate) used to order the two visits.

        Returns
        -------
        xr.DataArray
            ``(subjects, response_vars)`` z-diff scores. Subjects without
            exactly two timepoints are ``NaN``.
        """
        subject_id_col = subject_id_col or self.subject_id_col

        self._check_is_predicted(self.reference_data)
        self._check_is_longitudinal(self.reference_data)
        self._check_is_predicted(test_data)
        self._check_is_longitudinal(test_data)
        self._check_at_most_two_timepoints(test_data)

        response_vars = [str(r) for r in test_data.response_vars.values]
        subjects = self._ordered_unique(self._get_subject_ids(test_data))

        scores = np.full((len(subjects), len(response_vars)),
                         np.nan, dtype=float)
        subject_index = {s: i for i, s in enumerate(subjects)}

        for j, rv in enumerate(response_vars):
            norm_deltas = self._residual_change(
                self.reference_data, rv, timepoint_col
            )
            if len(norm_deltas) == 0:
                raise ValueError(
                    f"Cannot estimate denominator for '{rv}': "
                    "no subject in reference_data has exactly two timepoints."
                )
            delta_values = np.fromiter(norm_deltas.values(), dtype=float)
            denominator = np.sqrt(float(np.mean(delta_values**2)))
            deltas = self._residual_change(test_data, rv, timepoint_col)
            for subject, delta in deltas.items():
                scores[subject_index[subject], j] = delta / denominator

        return xr.DataArray(
            scores,
            dims=("subjects", "response_vars"),
            coords={"subjects": subjects, "response_vars": response_vars},
            name="zdiff",
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _residual_change(self, data: "NormData", responsevar: str, timepoint_col: str) -> dict:
        """Map each two-visit subject to its warped residual change ``Δr = r₂ − r₁``."""
        residuals = self._warped_residual(data, responsevar)
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

    def _warped_residual(self, data: "NormData", responsevar: str) -> np.ndarray:
        """Per-observation residual ``φ(y) − φ(ŷ)`` for one response variable.

        For a warped BLR the observed and predicted values are first mapped to
        the space in which the warp was fitted (using the model's outscaler,
        which is identity if the user chose no scaling) and then warped. For a
        non-warped BLR this reduces to ``y − ŷ`` in the original space.
        """
        blr = self.normative_model[responsevar]
        y = np.asarray(data.Y.sel(
            response_vars=responsevar).values, dtype=float)
        yhat = np.asarray(data.Yhat.sel(
            response_vars=responsevar).values, dtype=float)

        if getattr(blr, "warp", None) is not None:
            outscaler = self.normative_model.outscalers[responsevar]
            y = blr.warp.f(outscaler.transform(y), blr.gamma)
            yhat = blr.warp.f(outscaler.transform(yhat), blr.gamma)

        return y - yhat

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
