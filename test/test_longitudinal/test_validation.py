"""Minimal tests for NormData visit handling and LongitudinalScore validation."""

from __future__ import annotations

import pytest

from pcntoolkit.longitudinal_score.longitudinal_score import LongitudinalScore


@pytest.mark.parametrize(
    ("factory_kwargs", "error_match"),
    [
        ({"duplicate_visits": True}, "identical visit labels"),
        ({}, None),
    ],
    ids=["duplicate_visits", "valid"],
)
def test_check_is_longitudinal_visit_rules(
    longitudinal_dataframe_factory,
    predicted_norm_data_factory,
    factory_kwargs,
    error_match,
):
    data = predicted_norm_data_factory(longitudinal_dataframe_factory(**factory_kwargs))
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            LongitudinalScore._check_is_longitudinal(data)
    else:
        LongitudinalScore._check_is_longitudinal(data)


def test_get_visits_requires_numeric_labels(
    longitudinal_dataframe,
    predicted_norm_data_factory,
):
    df = longitudinal_dataframe.copy()
    df["visit"] = df["visit"].map({1: "baseline", 2: "followup"})
    with pytest.raises(ValueError, match="must be numeric"):
        predicted_norm_data_factory(df).get_visits()


def test_score_validation_requires_predictions(predicted_longitudinal_norm_data):
    data = predicted_longitudinal_norm_data.drop_vars("Yhat")
    with pytest.raises(ValueError, match="'Yhat'"):
        LongitudinalScore._check_is_predicted(data)
