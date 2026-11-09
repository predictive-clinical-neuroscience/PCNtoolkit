"""Tests for thriveline utilities, VelocityModel.compute_thrivelines, and
plot_thrivelines."""

from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pcntoolkit.longitudinal.correlation_matrix import CorrelationMatrix
from pcntoolkit.longitudinal.velocity_model import VelocityModel
from pcntoolkit.math_functions.velocity import (
    _resolve_reference_batch_effects,
    compute_thriveline_z,
    thrivelines_to_dataframe,
)
from pcntoolkit.util.plotter import plot_thrivelines


def test_compute_thrivelines_negative_correlation(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=3, offset_correlations={1: -0.5})
    thrive_Z, _ = compute_thriveline_z(
        matrix,
        centiles=[0.5],
        covariate_range=(0, 1),
    )
    seg = thrive_Z.isel(segment=0).sel(response_vars="metric_a", drop=True)
    assert seg.isel(offset=1).item() < seg.isel(offset=0).item()


def test_compute_thrivelines_respects_spacing(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=4, offset_correlations={1: 0.8})
    thrive_Z, _ = compute_thriveline_z(
        matrix,
        thriveline_spacing=2,
        centiles=[0.5],
        covariate_range=(0, 4),
    )
    assert set(thrive_Z.coords["start_age"].values) <= {0, 2, 4}


def test_compute_thrivelines_span_of_two(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=4, offset_correlations={2: 0.7})
    thrive_Z, thrive_X = compute_thriveline_z(
        matrix,
        thriveline_span=2,
        centiles=[0.84],
        covariate_range=(0, 2),
    )
    assert list(thrive_X.coords["offset"].values) == [0, 2]
    assert thrive_Z.attrs.get("thriveline_span") == 2


def test_thrivelines_to_dataframe_multiple_response_vars(
    correlation_matrix_array_factory,
):
    matrix = correlation_matrix_array_factory(
        max_age=2,
        response_vars=["metric_a", "metric_b"],
        offset_correlations={1: 0.8},
    )
    thrive_Z, thrive_X = compute_thriveline_z(
        matrix,
        centiles=[0.5],
        covariate_range=(0, 1),
    )
    thrive_Y = thrive_Z.copy()
    thrive_Y.values = thrive_Z.values + 1.0
    df = thrivelines_to_dataframe(thrive_Z, thrive_X, thrive_Y)
    assert set(df["response_var"]) == {"metric_a", "metric_b"}


def test_zgain_compute_thrivelines_overwrites_cache(
    fitted_norm_blr_model, blr_correlation_matrix
):
    scorer = VelocityModel(fitted_norm_blr_model, blr_correlation_matrix)
    first = scorer.compute_thrivelines(centiles=[0.5], covariate_range=(0, 1))
    second = scorer.compute_thrivelines(centiles=[0.84, 0.98], covariate_range=(0, 2))
    assert scorer.thrivelines is second
    assert len(second) != len(first) or not first.equals(second)


def test_plot_thrivelines_rejects_missing_columns(fitted_norm_blr_model):
    bad_df = pd.DataFrame({"segment": [0], "X": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        plot_thrivelines(
            fitted_norm_blr_model,
            bad_df,
            show_figure=False,
        )


def test_compute_thrivelines_rejects_non_positive_thriveline_span(
    synthetic_correlation_matrix,
):
    with pytest.raises(ValueError, match="thriveline_span"):
        compute_thriveline_z(synthetic_correlation_matrix, thriveline_span=0)


def test_compute_thrivelines_output_structure(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(
        max_age=3,
        response_vars=["metric_a", "metric_b"],
    )
    thrive_Z, thrive_X = compute_thriveline_z(
        matrix,
        thriveline_span=1,
        thriveline_spacing=1,
        centiles=[0.5],
    )

    assert thrive_Z.dims == ("segment", "response_vars", "offset")
    assert thrive_X.dims == ("segment", "response_vars", "offset")
    assert list(thrive_Z.coords["response_vars"].values) == ["metric_a", "metric_b"]
    assert list(thrive_X.coords["offset"].values) == [0, 1]
    assert "start_age" in thrive_Z.coords
    assert "start_z" in thrive_Z.coords


def test_compute_thrivelines_propagates_with_known_correlation(
    correlation_matrix_array_factory,
):
    matrix = correlation_matrix_array_factory(max_age=2, offset_correlations={1: 0.8})
    thrive_Z, thrive_X = compute_thriveline_z(
        matrix,
        thriveline_span=1,
        thriveline_spacing=1,
        centiles=[norm.cdf(1.0)],
        covariate_range=(0, 1),
    )

    mask = (thrive_Z.coords["start_age"] == 0) & np.isclose(
        thrive_Z.coords["start_z"], 1.0
    )
    seg_idx = int(np.where(mask)[0][0])
    seg = thrive_Z.isel(segment=seg_idx).sel(response_vars="metric_a", drop=True)
    expected_next = 1.0 * 0.8 + math.sqrt(1.0 - 0.8**2) * (-1.96)

    assert seg.isel(offset=0).item() == pytest.approx(1.0)
    assert seg.isel(offset=1).item() == pytest.approx(expected_next)

    x_seg = thrive_X.isel(segment=seg_idx).sel(response_vars="metric_a", drop=True)
    assert list(x_seg.values) == [0, 1]


def test_compute_thrivelines_starts_at_given_centiles(
    correlation_matrix_array_factory,
):
    matrix = correlation_matrix_array_factory(max_age=2, offset_correlations={1: 0.8})
    thrive_Z, _ = compute_thriveline_z(
        matrix,
        centiles=[0.05, 0.5],
        covariate_range=(0, 1),
    )

    np.testing.assert_allclose(
        np.unique(thrive_Z.coords["start_z"].values), norm.ppf([0.05, 0.5])
    )


def test_compute_thrivelines_default_centiles(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=2, offset_correlations={1: 0.8})
    thrive_Z, _ = compute_thriveline_z(matrix, covariate_range=(0, 1))

    np.testing.assert_allclose(
        np.unique(thrive_Z.coords["start_z"].values),
        norm.ppf([0.05, 0.25, 0.5, 0.75, 0.95]),
    )


@pytest.mark.parametrize("bad_centile", [0.0, 1.0, 1.5])
def test_compute_thrivelines_rejects_centiles_outside_0_1(
    correlation_matrix_array_factory, bad_centile
):
    matrix = correlation_matrix_array_factory(max_age=2, offset_correlations={1: 0.8})
    with pytest.raises(ValueError, match="centiles"):
        compute_thriveline_z(matrix, centiles=[0.5, bad_centile])


def test_compute_thrivelines_zero_correlation_warns_and_omits_segments(
    correlation_matrix_array_factory,
):
    matrix = correlation_matrix_array_factory(max_age=3, offset_correlations={1: 0.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        thrive_Z, thrive_X = compute_thriveline_z(
            matrix,
            thriveline_span=1,
            centiles=[0.5],
            covariate_range=(0, 2),
        )

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "cannot be completed" in str(caught[0].message).lower()
    # The warning should mention which segments could not be computed.
    assert "0->1, 1->2" in str(caught[0].message)
    assert np.all(np.isnan(thrive_Z.values))
    # NaN lines are left out of the final table. Both lines failed,
    # so the table should be empty.
    assert thrivelines_to_dataframe(thrive_Z, thrive_X, thrive_Z).empty


def test_compute_thrivelines_respects_covariate_range(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=5, offset_correlations={1: 0.8})
    thrive_Z, thrive_X = compute_thriveline_z(
        matrix,
        covariate_range=(1, 2),
        centiles=[0.5],
    )

    assert set(thrive_Z.coords["start_age"].values) <= {1, 2}
    assert np.all(np.isfinite(thrive_X.sel(response_vars="metric_a").values))


def test_compute_thrivelines_default_covariate_range(
    fitted_norm_blr_model, correlation_matrix_array_factory
):
    """Without covariate_range, lines cover the model's training range (0-1),
    not every age in the matrix (0-5)."""
    matrix = correlation_matrix_array_factory(
        max_age=5, response_vars=["response_var_0"], covariate="covariate_0"
    )
    corr = CorrelationMatrix(matrix, covariate="covariate_0")

    thrivelines = VelocityModel(fitted_norm_blr_model, corr).compute_thrivelines(
        centiles=[0.5]
    )

    assert thrivelines["X"].min() == 0
    assert thrivelines["X"].max() == 1


def test_resolve_reference_batch_effects_uses_first_level(batch_effect_model):
    resolved = _resolve_reference_batch_effects(batch_effect_model)
    assert resolved == {"site": "A", "sex": "F"}


def test_resolve_reference_batch_effects_honours_overrides(batch_effect_model):
    resolved = _resolve_reference_batch_effects(
        batch_effect_model,
        batch_effects={"site": "C"},
    )
    assert resolved == {"site": "C", "sex": "F"}


def test_resolve_reference_batch_effects_without_counts():
    model = type(
        "Model",
        (),
        {"unique_batch_effects": {"site": ["A", "B"]}, "batch_effect_counts": None},
    )()
    resolved = _resolve_reference_batch_effects(model)
    assert resolved == {"site": "A"}


def test_thrivelines_to_dataframe_long_format(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=2, offset_correlations={1: 0.8})
    thrive_Z, thrive_X = compute_thriveline_z(
        matrix,
        centiles=[0.5, 0.84],
        covariate_range=(0, 1),
    )
    thrive_Y = thrive_Z.copy()
    thrive_Y.values = thrive_Z.values + 10.0
    thrive_Y.name = "thrive_Y"

    df = thrivelines_to_dataframe(thrive_Z, thrive_X, thrive_Y)

    assert list(df.columns) == [
        "segment",
        "response_var",
        "offset",
        "X",
        "Z",
        "Y",
    ]
    assert len(df) == thrive_Z.sizes["segment"] * thrive_Z.sizes["offset"]
    assert set(df["response_var"]) == {"metric_a"}
    assert set(df["offset"]) == {0, 1}
    assert df.attrs.get("thriveline_span") == 1


def test_zgain_compute_thrivelines_returns_dataframe(
    fitted_norm_blr_model, blr_correlation_matrix
):
    scorer = VelocityModel(fitted_norm_blr_model, blr_correlation_matrix)
    thrivelines = scorer.compute_thrivelines(
        centiles=[0.5, 0.84],
        covariate_range=(0, 2),
    )

    assert list(thrivelines.columns) == [
        "segment",
        "response_var",
        "offset",
        "X",
        "Z",
        "Y",
    ]
    assert scorer.thrivelines is thrivelines
    assert not thrivelines.empty


def test_zgain_compute_thrivelines_saved_csv_matches_returned_dataframe(
    tmp_path, fitted_norm_blr_model, blr_correlation_matrix
):
    scorer = VelocityModel(fitted_norm_blr_model, blr_correlation_matrix)
    save_path = tmp_path / "thrivelines.csv"

    thrivelines = scorer.compute_thrivelines(
        centiles=[0.5, 0.84],
        covariate_range=(0, 2),
        save_path=save_path,
    )

    saved = pd.read_csv(save_path)
    # Compare contents only: ignore column order
    pd.testing.assert_frame_equal(saved, thrivelines, check_like=True)


def test_plot_thrivelines_returns_figures(
    fitted_norm_blr_model,
    blr_correlation_matrix,
):
    scorer = VelocityModel(fitted_norm_blr_model, blr_correlation_matrix)
    thrivelines = scorer.compute_thrivelines(
        centiles=[0.5],
        covariate_range=(0, 2),
    )

    model_ranges = fitted_norm_blr_model.covariate_ranges
    figures = plot_thrivelines(
        fitted_norm_blr_model,
        thrivelines,
        centiles=[0.05, 0.5, 0.95],
        covariate="covariate_0",
        covariate_ranges={
            "covariate_0": (0, 2),
            "covariate_1": (
                model_ranges["covariate_1"]["min"],
                model_ranges["covariate_1"]["max"],
            ),
        },
        response_vars=["response_var_0"],
        show_figure=False,
    )

    assert len(figures) == 1
    for fig in figures:
        plt.close(fig)


def test_plot_thrivelines_uses_thriveline_covariate(
    fitted_norm_blr_model, correlation_matrix_array_factory
):
    """Use the thrivelines' covariate on the x-axis of the plot and raise if 
    this covariate is unknown."""
    # Thrivelines run along covariate_1, the model's second covariate, so the old
    # default (first model covariate) would pick the wrong x-axis.
    matrix = correlation_matrix_array_factory(
        response_vars=["response_var_0"], covariate="covariate_1"
    )
    corr = CorrelationMatrix(matrix, covariate="covariate_1")
    thrivelines = VelocityModel(fitted_norm_blr_model, corr).compute_thrivelines(
        centiles=[0.5], covariate_range=(0, 2)
    )

    figures = plot_thrivelines(
        fitted_norm_blr_model,
        thrivelines,
        response_vars=["response_var_0"],
        show_figure=False,
    )
    assert figures[0].axes[0].get_xlabel() == "covariate_1"
    plt.close(figures[0])

    # A thriveline table read back from CSV has lost its attrs, so the covariate is unknown.
    thrivelines.attrs = {}
    with pytest.raises(ValueError, match="covariate="):
        plot_thrivelines(fitted_norm_blr_model, thrivelines, show_figure=False)
