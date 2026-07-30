"""Tests for thriveline utilities in velocity.py."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import xarray as xr

from pcntoolkit.math_functions.velocity import (
    _resolve_reference_batch_effects,
    compute_thrivelines,
    thrivelines_to_dataframe,
    propagate_thriveline_z
)


def _make_correlation_matrix(
    max_age: int = 5,
    response_vars: list[str] | None = None,
    offset_correlations: dict[int, float] | None = None,
) -> xr.DataArray:
    """Build a synthetic longitudinal correlation matrix for thriveline tests."""
    if response_vars is None:
        response_vars = ["metric_a"]
    if offset_correlations is None:
        offset_correlations = {1: 0.8}

    ages = np.arange(max_age + 1)
    mats = []
    for _ in response_vars:
        mat = np.eye(max_age + 1)
        for offset, r in offset_correlations.items():
            for age in range(max_age + 1 - offset):
                later = age + offset
                mat[age, later] = mat[later, age] = r
        mats.append(mat)

    return xr.DataArray(
        np.stack(mats),
        dims=("response_vars", "age_1", "age_2"),
        coords={"response_vars": response_vars, "age_1": ages, "age_2": ages},
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


def test_compute_thrivelines_rejects_non_positive_timepoint_diff():
    R = _make_correlation_matrix()
    with pytest.raises(ValueError, match="timepoint_diff"):
        compute_thrivelines(R, timepoint_diff=0)


def test_compute_thrivelines_output_structure():
    R = _make_correlation_matrix(max_age=3, response_vars=["metric_a", "metric_b"])
    thrive_Z, thrive_X = compute_thrivelines(
        R,
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


def test_compute_thrivelines_propagates_with_known_correlation():
    R = _make_correlation_matrix(max_age=2, offset_correlations={1: 0.8})
    thrive_Z, thrive_X = compute_thrivelines(
        R,
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


def test_compute_thrivelines_uses_explicit_z_anchors():
    R = _make_correlation_matrix(max_age=2, offset_correlations={1: 0.8})
    z_anchors = np.array([-1.0, 0.5])
    thrive_Z, _ = compute_thrivelines(
        R,
        z_anchors=z_anchors,
        covariate_range=(0, 1),
    )

    assert set(np.round(thrive_Z.coords["start_z"].values, 1)) == {-1.0, 0.5}


def test_compute_thrivelines_zero_correlation_warns_and_omits_segments():
    R = _make_correlation_matrix(max_age=3, offset_correlations={1: 0.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        thrive_Z, _ = compute_thrivelines(
            R,
            timepoint_diff=1,
            z_anchor_start=0,
            z_anchor_end=1,
            covariate_range=(0, 2),
        )

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "cannot be completed" in str(caught[0].message).lower()
    assert np.all(np.isnan(thrive_Z.values))


def test_compute_thrivelines_respects_covariate_range():
    R = _make_correlation_matrix(max_age=5, offset_correlations={1: 0.8})
    thrive_Z, thrive_X = compute_thrivelines(
        R,
        covariate_range=(1, 2),
        z_anchor_start=0,
        z_anchor_end=1,
    )

    assert set(thrive_Z.coords["start_age"].values) <= {1, 2}
    assert np.all(np.isfinite(thrive_X.sel(response_vars="metric_a").values))


class _BatchEffectModel:
    unique_batch_effects = {"site": ["A", "B", "C"], "sex": ["F", "M"]}
    batch_effect_counts = {
        "site": {"A": 10, "B": 50, "C": 5},
        "sex": {"F": 30, "M": 35},
    }


def test_resolve_reference_batch_effects_uses_first_level():
    resolved = _resolve_reference_batch_effects(_BatchEffectModel())
    assert resolved == {"site": "A", "sex": "F"}


def test_resolve_reference_batch_effects_honours_overrides():
    resolved = _resolve_reference_batch_effects(
        _BatchEffectModel(),
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


def test_thrivelines_to_dataframe_long_format():
    R = _make_correlation_matrix(max_age=2, offset_correlations={1: 0.8})
    thrive_Z, thrive_X = compute_thrivelines(
        R,
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
