"""Tests for CorrelationMatrix, ZGainScore, and ZDiffScore."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pcntoolkit.longitudinal_score.zdiff_score import ZDiffScore
from pcntoolkit.longitudinal_score.zgain_score import ZGainScore
from pcntoolkit.math_functions.correlation_matrix import CorrelationMatrix

from test.test_longitudinal.conftest import (
    expected_zgain,
    make_cross_sectional_norm_data,
    set_z_scores,
)


# ------------------------------------------------------------------ #
# CorrelationMatrix
# ------------------------------------------------------------------ #


def test_correlation_matrix_compute_from_longitudinal_cohort(longitudinal_cohort_norm_data):
    corr = CorrelationMatrix.compute(
        longitudinal_cohort_norm_data,
        bandwidth=1,
        covariate="age",
    )

    assert corr.n_subjects == 30
    assert corr.estimated_range == (10, 11)
    assert corr.matrix.dims == ("response_vars", "age_1", "age_2")
    assert np.allclose(np.diagonal(corr.matrix.values[0]), 1.0)


def test_correlation_matrix_compute_rejects_cross_sectional(predicted_norm_data_factory):
    data = make_cross_sectional_norm_data(predicted_norm_data_factory)
    with pytest.raises(ValueError, match="single-visit|cross-sectional|multiple visits"):
        CorrelationMatrix.compute(data, bandwidth=1)


def test_correlation_matrix_compute_rejects_missing_z(longitudinal_cohort_norm_data):
    data = longitudinal_cohort_norm_data.drop_vars("Z")
    with pytest.raises(ValueError, match="'Z'"):
        CorrelationMatrix.compute(data, bandwidth=1)


def test_correlation_matrix_invalid_max_correlation(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory()
    with pytest.raises(ValueError, match="max_correlation"):
        CorrelationMatrix(matrix, max_correlation=1.0)


def test_correlation_matrix_save_not_implemented(correlation_matrix):
    with pytest.raises(NotImplementedError):
        correlation_matrix.save("/tmp/unused.pkl")


@pytest.mark.parametrize(
    ("age_1", "age_2", "expected_r"),
    [
        (0, 1, 0.8),
        (1, 0, 0.8),
        (-5, 1, 0.8),
        (99, 4, 0.8),
    ],
    ids=["forward", "symmetric", "clamp_low", "clamp_high"],
)
def test_correlation_matrix_get_clamps_ages(
    correlation_matrix,
    age_1,
    age_2,
    expected_r,
):
    assert correlation_matrix.get("metric_a", age_1, age_2) == pytest.approx(expected_r)


def test_correlation_matrix_get_respects_max_correlation(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(offset_correlations={1: 0.999})
    corr = CorrelationMatrix(matrix, max_correlation=0.5)
    assert corr.get("metric_a", 0, 1) == pytest.approx(0.5)


def test_correlation_matrix_get_warns_outside_estimated_range(correlation_matrix):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlation_matrix.get("metric_a", 99, 100)

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)


def test_correlation_matrix_get_unknown_response_var_raises(correlation_matrix):
    with pytest.raises(KeyError, match="metric_b"):
        correlation_matrix.get("metric_b", 0, 1)


# ------------------------------------------------------------------ #
# ZGainScore — formula and edge cases
# ------------------------------------------------------------------ #


def _zgain_setup(
    fitted_norm_blr_model,
    correlation_matrix_array_factory,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
    *,
    z_values: np.ndarray,
    ages_subject_a: tuple[float, float] = (20.0, 21.0),
    visits_subject_a: tuple[int, int] = (1, 2),
    r: float = 0.8,
    extra_visits: list[int] | None = None,
):
    df = blr_longitudinal_dataframe_factory(extra_visits=extra_visits)
    if extra_visits is None:
        mask = df["sub_id"] == "a"
        df.loc[mask, "visit"] = list(visits_subject_a)
        df.loc[mask, "covariate_0"] = list(ages_subject_a)
    data = blr_predicted_norm_data_factory(df, z_values=z_values)
    matrix = correlation_matrix_array_factory(
        max_age=25,
        response_vars=["response_var_0"],
        offset_correlations={1: r},
        covariate="covariate_0",
    )
    corr = CorrelationMatrix(
        matrix,
        covariate="covariate_0",
        estimated_range=(0, 25),
    )
    scorer = ZGainScore(fitted_norm_blr_model, corr)
    return scorer, data, corr


def test_zgain_formula_with_nonzero_prior_z(
    fitted_norm_blr_model,
    correlation_matrix_array_factory,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    z_prev, z_last, r = 1.5, 0.2, 0.8
    scorer, data, _ = _zgain_setup(
        fitted_norm_blr_model,
        correlation_matrix_array_factory,
        blr_longitudinal_dataframe_factory,
        blr_predicted_norm_data_factory,
        z_values=np.array([[z_prev], [z_last], [0.0], [0.0]]),
    )
    scores = scorer.score(data)
    expected = expected_zgain(z_prev, z_last, r)
    assert scores.sel(subjects="a", response_vars="response_var_0").item() == pytest.approx(
        expected
    )


@pytest.mark.parametrize("r", [0.0, -0.5, 0.95])
def test_zgain_formula_across_correlation_values(
    fitted_norm_blr_model,
    correlation_matrix_array_factory,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
    r,
):
    z_prev, z_last = 0.5, 2.0
    scorer, data, _ = _zgain_setup(
        fitted_norm_blr_model,
        correlation_matrix_array_factory,
        blr_longitudinal_dataframe_factory,
        blr_predicted_norm_data_factory,
        z_values=np.array([[z_prev], [z_last], [0.0], [0.0]]),
        r=r,
    )
    scores = scorer.score(data)
    assert scores.sel(subjects="a", response_vars="response_var_0").item() == pytest.approx(
        expected_zgain(z_prev, z_last, r)
    )


def test_zgain_uses_last_two_visits_when_three_present(
    fitted_norm_blr_model,
    correlation_matrix_array_factory,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    """Three visits: score must use visits 2→3, not 1→2."""
    z1, z2, z3, r = 0.0, 5.0, 1.0, 0.8
    scorer, data, _ = _zgain_setup(
        fitted_norm_blr_model,
        correlation_matrix_array_factory,
        blr_longitudinal_dataframe_factory,
        blr_predicted_norm_data_factory,
        z_values=np.array([[z1], [z2], [z3], [0.0], [0.0]]),
        extra_visits=[1, 2, 3],
    )
    scores = scorer.score(data)
    assert scores.sel(subjects="a", response_vars="response_var_0").item() == pytest.approx(
        expected_zgain(z2, z3, r)
    )


def test_zgain_sorts_by_visit_label_not_row_order(
    fitted_norm_blr_model,
    correlation_matrix_array_factory,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    """Rows shuffled so visit 2 appears before visit 1."""
    df = blr_longitudinal_dataframe_factory(n_subjects=1, ages=(20.0, 21.0))
    df = df.sort_values("visit", ascending=False).reset_index(drop=True)
    z_prev, z_last, r = 1.0, 3.0, 0.8
    data = blr_predicted_norm_data_factory(
        df,
        z_values=np.array([[z_last], [z_prev]]),
    )
    matrix = correlation_matrix_array_factory(
        max_age=25,
        response_vars=["response_var_0"],
        offset_correlations={1: r},
        covariate="covariate_0",
    )
    corr = CorrelationMatrix(matrix, covariate="covariate_0", estimated_range=(0, 25))
    scores = ZGainScore(fitted_norm_blr_model, corr).score(data)
    assert scores.sel(subjects="a", response_vars="response_var_0").item() == pytest.approx(
        expected_zgain(z_prev, z_last, r)
    )


def test_zgain_leaves_single_visit_subject_as_nan(
    fitted_norm_blr_model,
    correlation_matrix_array_factory,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    df = blr_longitudinal_dataframe_factory(n_subjects=2)
    single = df[df["sub_id"] == "b"].iloc[[0]]
    mixed = pd.concat([df[df["sub_id"] == "a"], single], ignore_index=True)

    data = blr_predicted_norm_data_factory(
        mixed,
        z_values=np.array([[0.0], [1.0], [2.0]]),
    )
    matrix = correlation_matrix_array_factory(
        max_age=25,
        response_vars=["response_var_0"],
        covariate="covariate_0",
    )
    corr = CorrelationMatrix(matrix, covariate="covariate_0", estimated_range=(0, 25))
    scores = ZGainScore(fitted_norm_blr_model, corr).score(data)
    assert np.isnan(scores.sel(subjects="b", response_vars="response_var_0").item())
    assert np.isfinite(scores.sel(subjects="a", response_vars="response_var_0").item())


def test_zgain_caches_and_overwrites_on_rescore(
    fitted_norm_blr_model,
    blr_correlation_matrix,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    scorer = ZGainScore(fitted_norm_blr_model, blr_correlation_matrix)
    data_v1 = blr_predicted_norm_data_factory(
        blr_longitudinal_dataframe_factory(),
        z_values=np.array([[0.0], [1.0], [0.0], [0.0]]),
    )
    first = scorer.score(data_v1)
    assert scorer.zgain is first

    data_v2 = set_z_scores(data_v1, np.array([[0.0], [5.0], [0.0], [0.0]]))
    second = scorer.score(data_v2)
    assert scorer.zgain is second
    assert second.sel(subjects="a").item() != pytest.approx(first.sel(subjects="a").item())


def test_zgain_rejects_unpredicted_data(
    fitted_norm_blr_model,
    blr_correlation_matrix,
    blr_predicted_longitudinal_norm_data,
):
    scorer = ZGainScore(fitted_norm_blr_model, blr_correlation_matrix)
    bad = blr_predicted_longitudinal_norm_data.drop_vars("Z")
    with pytest.raises(ValueError, match="'Z'"):
        scorer.score(bad)


def test_zgain_rejects_duplicate_visit_labels(
    fitted_norm_blr_model,
    blr_correlation_matrix,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    scorer = ZGainScore(fitted_norm_blr_model, blr_correlation_matrix)
    data = blr_predicted_norm_data_factory(
        blr_longitudinal_dataframe_factory(duplicate_visits=True)
    )
    with pytest.raises(ValueError, match="identical visit labels"):
        scorer.score(data)


def test_zgain_subject_order_follows_first_appearance(
    fitted_norm_blr_model,
    correlation_matrix_array_factory,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    data = blr_predicted_norm_data_factory(blr_longitudinal_dataframe_factory(n_subjects=3))
    matrix = correlation_matrix_array_factory(
        response_vars=["response_var_0"],
        covariate="covariate_0",
        max_age=50,
    )
    corr = CorrelationMatrix(matrix, covariate="covariate_0", estimated_range=(0, 50))
    scores = ZGainScore(fitted_norm_blr_model, corr).score(data)
    assert list(scores.coords["subjects"].values) == ["a", "b", "c"]


# ------------------------------------------------------------------ #
# ZDiffScore — formula and edge cases
# ------------------------------------------------------------------ #


def test_zdiff_score_scales_by_reference_spread(
    fitted_norm_blr_model,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    ref_df = blr_longitudinal_dataframe_factory(n_subjects=2)
    ref_y = np.array([[10.0], [12.0], [11.0], [13.0]])
    ref_yhat = np.array([[10.0], [10.5], [11.0], [11.5]])
    reference = blr_predicted_norm_data_factory(ref_df, yhat_values=ref_yhat)
    reference["Y"] = (["observations", "response_vars"], ref_y)

    score_df = blr_longitudinal_dataframe_factory(n_subjects=1, ages=(20.0, 21.0))
    y = np.array([[10.0], [12.0]])
    yhat = np.array([[10.0], [10.0]])
    score_data = blr_predicted_norm_data_factory(score_df, yhat_values=yhat)
    score_data["Y"] = (["observations", "response_vars"], y)

    scorer = ZDiffScore(fitted_norm_blr_model, reference, subject_id_col="sub_id")
    scores = scorer.score(score_data)

    ref_deltas = list(scorer._compute_residual_change(reference, "response_var_0").values())
    denom = np.sqrt(np.mean(np.square(ref_deltas)))
    target_delta = scorer._compute_residual_change(score_data, "response_var_0")["a"]

    assert scores.sel(subjects="a", response_vars="response_var_0").item() == pytest.approx(
        target_delta / denom
    )
    assert scorer.zdiff is scores


def test_zdiff_doubles_when_target_change_doubles(
    fitted_norm_blr_model,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    reference = blr_predicted_norm_data_factory(blr_longitudinal_dataframe_factory())
    base_df = blr_longitudinal_dataframe_factory(n_subjects=1, ages=(20.0, 21.0))

    def _score_with_y(y_pair: tuple[float, float]) -> float:
        data = blr_predicted_norm_data_factory(base_df)
        y = np.array([[y_pair[0]], [y_pair[1]]])
        data["Y"] = (["observations", "response_vars"], y)
        scorer = ZDiffScore(fitted_norm_blr_model, reference, subject_id_col="sub_id")
        return float(scorer.score(data).sel(subjects="a").item())

    small = _score_with_y((10.0, 11.0))
    large = _score_with_y((10.0, 12.0))
    assert np.sign(large) == np.sign(small)
    assert abs(large) > abs(small)


def test_zdiff_visit_order_not_row_order(
    fitted_norm_blr_model,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    reference = blr_predicted_norm_data_factory(blr_longitudinal_dataframe_factory())
    score_df = blr_longitudinal_dataframe_factory(n_subjects=1)
    score_df = score_df.sort_values("visit", ascending=False).reset_index(drop=True)
    y = np.array([[5.0], [8.0]])
    score_data = blr_predicted_norm_data_factory(score_df, yhat_values=y.copy())

    scorer = ZDiffScore(fitted_norm_blr_model, reference, subject_id_col="sub_id")
    scores = scorer.score(score_data)
    assert np.isfinite(scores.sel(subjects="a").item())


def test_zdiff_zero_reference_variability_raises(
    fitted_norm_blr_model,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    df = blr_longitudinal_dataframe_factory()
    yhat = None
    flat = blr_predicted_norm_data_factory(df, yhat_values=np.zeros((4, 1)))
    flat["Yhat"] = (["observations", "response_vars"], flat.Y.values.copy())

    scorer = ZDiffScore(fitted_norm_blr_model, flat, subject_id_col="sub_id")
    target = blr_predicted_norm_data_factory(blr_longitudinal_dataframe_factory(n_subjects=1))
    with pytest.raises(ValueError, match="zero residual-change variability"):
        scorer.score(target)


def test_zdiff_rejects_non_blr_model(blr_predicted_longitudinal_norm_data):
    class _NotBlrModel:
        template_regression_model = object()

        def check_is_fitted(self) -> None:
            return None

    with pytest.raises(ValueError, match="BLR"):
        ZDiffScore(
            _NotBlrModel(),  # type: ignore[arg-type]
            blr_predicted_longitudinal_norm_data,
            subject_id_col="sub_id",
        )


def test_zdiff_rejects_more_than_two_visits(
    fitted_norm_blr_model,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    reference = blr_predicted_norm_data_factory(blr_longitudinal_dataframe_factory())
    score_data = blr_predicted_norm_data_factory(
        blr_longitudinal_dataframe_factory(n_subjects=1, extra_visits=[1, 2, 3])
    )
    scorer = ZDiffScore(fitted_norm_blr_model, reference, subject_id_col="sub_id")
    with pytest.raises(ValueError, match="at most two visits"):
        scorer.score(score_data)


def test_zdiff_caches_result(
    fitted_norm_blr_model,
    blr_predicted_longitudinal_norm_data,
    blr_longitudinal_dataframe_factory,
    blr_predicted_norm_data_factory,
):
    reference = blr_predicted_longitudinal_norm_data
    target = blr_predicted_norm_data_factory(blr_longitudinal_dataframe_factory())
    scorer = ZDiffScore(fitted_norm_blr_model, reference, subject_id_col="sub_id")
    first = scorer.score(target)
    assert scorer.zdiff is first
    second = scorer.score(target)
    assert scorer.zdiff is second
    assert np.allclose(first.values, second.values)


def test_zdiff_requires_reference_at_init(
    fitted_norm_blr_model,
):
    with pytest.raises(ValueError, match="reference"):
        ZDiffScore(
            fitted_norm_blr_model,
            None,  # type: ignore[arg-type]
            subject_id_col="sub_id",
        )
