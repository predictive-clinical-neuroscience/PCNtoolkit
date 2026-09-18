"""Tests for thriveline utilities, ZGainScore.get_thrivelines, and plot_thrivelines."""

from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pcntoolkit.longitudinal_score.zgain_score import ZGainScore
from pcntoolkit.math_functions.velocity import (
    _resolve_reference_batch_effects,
    compute_thrivelines,
    propagate_thriveline_z,
    thrivelines_to_dataframe,
)
from pcntoolkit.util.plotter import plot_thrivelines


def test_propagate_thriveline_z_multi_hop():
    hop = xr.DataArray(
        [0.7, 0.6],
        dims=("hop",),
        coords={"hop": [0, 1]},
        attrs={"timepoint_diff": 1},
    )
    z_path = propagate_thriveline_z(hop, start_z=0.5, z_thrive=-1.96)

    assert list(z_path.coords["offset"].values) == [0, 1, 2]
    assert z_path.isel(offset=0).item() == pytest.approx(0.5)
    assert np.isfinite(z_path.isel(offset=2).item())


def test_compute_thrivelines_negative_correlation(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=3, offset_correlations={1: -0.5})
    thrive_Z, _ = compute_thrivelines(
        matrix,
        z_anchors=[0.0],
        covariate_range=(0, 1),
    )
    seg = thrive_Z.isel(segment=0).sel(response_vars="metric_a", drop=True)
    assert seg.isel(offset=1).item() < seg.isel(offset=0).item()


def test_compute_thrivelines_respects_anchor_step(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=4, offset_correlations={1: 0.8})
    thrive_Z, _ = compute_thrivelines(
        matrix,
        anchor_step=2,
        z_anchor_start=0,
        z_anchor_end=1,
        covariate_range=(0, 4),
    )
    assert set(thrive_Z.coords["start_age"].values) <= {0, 2, 4}


def test_compute_thrivelines_timepoint_diff_two(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=4, offset_correlations={2: 0.7})
    thrive_Z, thrive_X = compute_thrivelines(
        matrix,
        timepoint_diff=2,
        z_anchors=[1.0],
        covariate_range=(0, 2),
    )
    assert list(thrive_X.coords["offset"].values) == [0, 2]
    assert thrive_Z.attrs.get("timepoint_diff") == 2


def test_thrivelines_to_dataframe_multiple_response_vars(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(
        max_age=2,
        response_vars=["metric_a", "metric_b"],
        offset_correlations={1: 0.8},
    )
    thrive_Z, thrive_X = compute_thrivelines(
        matrix,
        z_anchors=[0.0],
        covariate_range=(0, 1),
    )
    thrive_Y = thrive_Z.copy()
    thrive_Y.values = thrive_Z.values + 1.0
    df = thrivelines_to_dataframe(thrive_Z, thrive_X, thrive_Y)
    assert set(df["response_var"]) == {"metric_a", "metric_b"}


def test_zgain_get_thrivelines_overwrites_cache(fitted_norm_blr_model, blr_correlation_matrix):
    scorer = ZGainScore(fitted_norm_blr_model, blr_correlation_matrix)
    first = scorer.get_thrivelines(z_anchors=[0.0], covariate_range=(0, 1))
    second = scorer.get_thrivelines(z_anchors=[1.0, 2.0], covariate_range=(0, 2))
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


def test_propagate_thriveline_z_requires_hop_dimension():
    with pytest.raises(ValueError, match="hop"):
        propagate_thriveline_z(xr.DataArray([0.8]), start_z=0.0)


def test_propagate_thriveline_z_bayer_update():
    hop = xr.DataArray([0.8], dims=("hop",), coords={"hop": [0]}, attrs={"timepoint_diff": 2})
    z_path = propagate_thriveline_z(hop, start_z=1.0, z_thrive=-1.96)

    expected_next = 1.0 * 0.8 + math.sqrt(1.0 - 0.8**2) * (-1.96)
    assert z_path.dims == ("offset",)
    assert list(z_path.coords["offset"].values) == [0, 2]
    assert z_path.values[0] == pytest.approx(1.0)
    assert z_path.values[1] == pytest.approx(expected_next)


def test_compute_thrivelines_rejects_non_positive_timepoint_diff(
    synthetic_correlation_matrix,
):
    with pytest.raises(ValueError, match="timepoint_diff"):
        compute_thrivelines(synthetic_correlation_matrix, timepoint_diff=0)


def test_compute_thrivelines_output_structure(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(
        max_age=3,
        response_vars=["metric_a", "metric_b"],
    )
    thrive_Z, thrive_X = compute_thrivelines(
        matrix,
        timepoint_diff=1,
        anchor_step=1,
        z_anchor_start=0,
        z_anchor_end=1,
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
    thrive_Z, thrive_X = compute_thrivelines(
        matrix,
        timepoint_diff=1,
        anchor_step=1,
        z_anchor_start=1,
        z_anchor_end=2,
        covariate_range=(0, 1),
    )

    mask = (thrive_Z.coords["start_age"] == 0) & (thrive_Z.coords["start_z"] == 1)
    seg_idx = int(np.where(mask)[0][0])
    seg = thrive_Z.isel(segment=seg_idx).sel(response_vars="metric_a", drop=True)
    expected_next = 1.0 * 0.8 + math.sqrt(1.0 - 0.8**2) * (-1.96)

    assert seg.isel(offset=0).item() == pytest.approx(1.0)
    assert seg.isel(offset=1).item() == pytest.approx(expected_next)

    x_seg = thrive_X.isel(segment=seg_idx).sel(response_vars="metric_a", drop=True)
    assert list(x_seg.values) == [0, 1]


def test_compute_thrivelines_uses_explicit_z_anchors(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=2, offset_correlations={1: 0.8})
    z_anchors = np.array([-1.0, 0.5])
    thrive_Z, _ = compute_thrivelines(
        matrix,
        z_anchors=z_anchors,
        covariate_range=(0, 1),
    )

    assert set(np.round(thrive_Z.coords["start_z"].values, 1)) == {-1.0, 0.5}


def test_compute_thrivelines_zero_correlation_warns_and_omits_segments(
    correlation_matrix_array_factory,
):
    matrix = correlation_matrix_array_factory(max_age=3, offset_correlations={1: 0.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        thrive_Z, _ = compute_thrivelines(
            matrix,
            timepoint_diff=1,
            z_anchor_start=0,
            z_anchor_end=1,
            covariate_range=(0, 2),
        )

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "cannot be completed" in str(caught[0].message).lower()
    assert np.all(np.isnan(thrive_Z.values))


def test_compute_thrivelines_respects_covariate_range(correlation_matrix_array_factory):
    matrix = correlation_matrix_array_factory(max_age=5, offset_correlations={1: 0.8})
    thrive_Z, thrive_X = compute_thrivelines(
        matrix,
        covariate_range=(1, 2),
        z_anchor_start=0,
        z_anchor_end=1,
    )

    assert set(thrive_Z.coords["start_age"].values) <= {1, 2}
    assert np.all(np.isfinite(thrive_X.sel(response_vars="metric_a").values))


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
    thrive_Z, thrive_X = compute_thrivelines(
        matrix,
        z_anchors=[0.0, 1.0],
        covariate_range=(0, 1),
    )
    thrive_Y = thrive_Z.copy()
    thrive_Y.values = thrive_Z.values + 10.0
    thrive_Y.name = "thrive_Y"

    df = thrivelines_to_dataframe(thrive_Z, thrive_X, thrive_Y)

    assert list(df.columns) == [
        "segment",
        "start_age",
        "start_z",
        "response_var",
        "offset",
        "X",
        "Z",
        "Y",
    ]
    assert len(df) == thrive_Z.sizes["segment"] * thrive_Z.sizes["offset"]
    assert set(df["response_var"]) == {"metric_a"}
    assert set(df["offset"]) == {0, 1}
    assert df.attrs.get("timepoint_diff") == 1


def test_zgain_get_thrivelines_returns_dataframe(fitted_norm_blr_model, blr_correlation_matrix):
    scorer = ZGainScore(fitted_norm_blr_model, blr_correlation_matrix)
    thrivelines = scorer.get_thrivelines(
        z_anchors=[0.0, 1.0],
        covariate_range=(0, 2),
    )

    assert list(thrivelines.columns) == [
        "segment",
        "start_age",
        "start_z",
        "response_var",
        "offset",
        "X",
        "Z",
        "Y",
    ]
    assert scorer.thrivelines is thrivelines
    assert not thrivelines.empty


def test_plot_thrivelines_returns_figures(
    fitted_norm_blr_model,
    blr_correlation_matrix,
):
    scorer = ZGainScore(fitted_norm_blr_model, blr_correlation_matrix)
    thrivelines = scorer.get_thrivelines(
        z_anchors=[0.0],
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
