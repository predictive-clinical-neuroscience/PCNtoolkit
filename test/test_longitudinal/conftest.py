"""Shared fixtures and builders for longitudinal tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.correlation_matrix import CorrelationMatrix

# Re-export BLR and NormData fixtures without star-imports in test modules.
pytest_plugins = [
    "test.fixtures.data_fixtures",
    "test.fixtures.norm_data_fixtures",
    "test.fixtures.path_fixtures",
    "test.fixtures.blr_model_fixtures",
]


def make_blr_longitudinal_dataframe(
    *,
    duplicate_visits: bool = False,
    n_subjects: int = 2,
    ages: tuple[float, float] = (20.0, 22.0),
    extra_visits: list[int] | None = None,
) -> pd.DataFrame:
    """Build longitudinal data using the same column names as BLR fixtures."""
    rows: list[dict[str, Any]] = []
    for i in range(n_subjects):
        subject = chr(ord("a") + i)
        if extra_visits is not None and i == 0:
            visit_pair = extra_visits
            age_pair = [ages[0] + step for step in range(len(visit_pair))]
        elif duplicate_visits:
            visit_pair = [1, 1]
            age_pair = [ages[0], ages[0]]
        else:
            visit_pair = [1, 2]
            age_pair = list(ages)
        for visit, age in zip(visit_pair, age_pair, strict=True):
            rows.append(
                {
                    "sub_id": subject,
                    "visit": visit,
                    "covariate_0": age + float(i * 10),
                    "covariate_1": 0.5 + 0.1 * i,
                    "response_var_0": 1.0 + 0.1 * visit + float(i),
                    "batch_effect_0": 0,
                    "batch_effect_1": 0,
                }
            )
    return pd.DataFrame(rows)


def make_blr_predicted_norm_data(
    dataframe: pd.DataFrame,
    *,
    z_values: np.ndarray | None = None,
    yhat_values: np.ndarray | None = None,
) -> NormData:
    """Wrap BLR-style longitudinal data as predicted NormData."""
    data = NormData.from_dataframe(
        "longitudinal_blr",
        dataframe,
        covariates=["covariate_0", "covariate_1"],
        batch_effects=["batch_effect_0", "batch_effect_1"],
        response_vars=["response_var_0"],
        subject_ids="sub_id",
        visits="visit",
    )
    if yhat_values is None:
        yhat = data.Y.values.copy()
        yhat[0::2] *= 0.95
        yhat[1::2] *= 1.05
    else:
        yhat = yhat_values
    data["Yhat"] = (["observations", "response_vars"], yhat)
    if z_values is None:
        z_values = np.zeros(data.Y.shape)
    data["Z"] = (["observations", "response_vars"], z_values)
    return data


def set_z_scores(data: NormData, z_values: np.ndarray) -> NormData:
    """Return a copy of NormData with updated z-scores."""
    updated = data.copy(deep=True)
    updated["Z"] = (["observations", "response_vars"], z_values)
    return updated


def expected_zgain(z_prev: float, z_last: float, r: float) -> float:
    """Analytic z-gain for a two-visit subject."""
    return (z_last - r * z_prev) / np.sqrt(1.0 - r**2)


def make_cross_sectional_norm_data(predicted_norm_data_factory) -> NormData:
    """One row per subject — invalid for longitudinal scoring."""
    df = make_longitudinal_dataframe(n_subjects=3)
    df = df.groupby("sub_id", sort=False).head(1).reset_index(drop=True)
    return predicted_norm_data_factory(df)


def make_longitudinal_dataframe(
    *,
    duplicate_visits: bool = False,
    identical_ages: bool = False,
    n_subjects: int = 2,
    ages: tuple[float, float] = (20.0, 22.0),
) -> pd.DataFrame:
    """Build a small tidy longitudinal DataFrame for tests."""
    rows: list[dict[str, Any]] = []
    for i in range(n_subjects):
        subject = chr(ord("a") + i)
        visit_pair = [1, 1] if duplicate_visits else [1, 2]
        age_pair = [ages[0], ages[0]] if identical_ages else list(ages)
        for visit, age in zip(visit_pair, age_pair, strict=True):
            rows.append(
                {
                    "sub_id": subject,
                    "visit": visit,
                    "age": age + float(i * 10),
                    "site": "s1",
                    "sex": "F" if i % 2 == 0 else "M",
                    "metric_a": 1.0 + 0.1 * visit + float(i),
                }
            )
    return pd.DataFrame(rows)


def make_predicted_norm_data(
    dataframe: pd.DataFrame,
    *,
    name: str = "longitudinal",
    response_vars: list[str] | None = None,
) -> NormData:
    """Wrap a DataFrame as predicted longitudinal NormData."""
    if response_vars is None:
        response_vars = ["metric_a"]
    data = NormData.from_dataframe(
        name,
        dataframe,
        covariates=["age"],
        batch_effects=["site", "sex"],
        response_vars=response_vars,
        subject_ids="sub_id",
        visits="visit",
    )
    data["Yhat"] = (["observations", "response_vars"], data.Y.values.copy())
    data["Z"] = (["observations", "response_vars"], np.zeros(data.Y.shape))
    return data


def make_correlation_matrix_array(
    max_age: int = 5,
    response_vars: list[str] | None = None,
    offset_correlations: dict[int, float] | None = None,
    covariate: str = "age",
) -> xr.DataArray:
    """Build a synthetic longitudinal correlation matrix."""
    if response_vars is None:
        response_vars = ["metric_a"]
    if offset_correlations is None:
        offset_correlations = {1: 0.8}

    ages = np.arange(max_age + 1)
    mats = []
    for _ in response_vars:
        mat = np.eye(max_age + 1)
        for offset, corr in offset_correlations.items():
            for age in range(max_age + 1 - offset):
                later = age + offset
                mat[age, later] = mat[later, age] = corr
        mats.append(mat)

    return xr.DataArray(
        np.stack(mats),
        dims=("response_vars", f"{covariate}_1", f"{covariate}_2"),
        coords={
            "response_vars": response_vars,
            f"{covariate}_1": ages,
            f"{covariate}_2": ages,
        },
    )


def make_longitudinal_normdata(
    n_subjects: int = 30,
    ages: tuple[int, ...] = (10, 11),
    within_subject_noise: float = 0.05,
    covariate_name: str = "age",
    seed: int = 0,
) -> NormData:
    """Build longitudinal NormData with z-scores correlated within subject."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for subject_id in range(n_subjects):
        subject_level = rng.normal()
        for age in ages:
            z = subject_level + within_subject_noise * rng.normal()
            rows.append(
                {
                    covariate_name: float(age),
                    "metric_a": rng.normal(),
                    "site": "s1",
                    "sex": "F" if subject_id % 2 == 0 else "M",
                    "sub_id": subject_id,
                    "visit": age,
                    "z_metric_a": z,
                }
            )

    df = pd.DataFrame(rows)
    data = NormData.from_dataframe(
        "longitudinal_cohort",
        df,
        covariates=[covariate_name],
        batch_effects=["site", "sex"],
        response_vars=["metric_a"],
        subject_ids="sub_id",
        visits="visit",
    )
    data["Yhat"] = (["observations", "response_vars"], data.Y.values.copy())
    data["Z"] = (
        ["observations", "response_vars"],
        df[["z_metric_a"]].to_numpy(),
    )
    return data


class BatchEffectModel:
    """Minimal normative-model stub for batch-effect resolution tests."""

    unique_batch_effects = {"site": ["A", "B", "C"], "sex": ["F", "M"]}
    batch_effect_counts = {
        "site": {"A": 10, "B": 50, "C": 5},
        "sex": {"F": 30, "M": 35},
    }


@pytest.fixture
def longitudinal_dataframe() -> pd.DataFrame:
    """Two-subject longitudinal DataFrame with distinct visits."""
    return make_longitudinal_dataframe()


@pytest.fixture
def predicted_longitudinal_norm_data(
    longitudinal_dataframe: pd.DataFrame,
) -> NormData:
    """Small predicted NormData with visit labels."""
    return make_predicted_norm_data(longitudinal_dataframe)


@pytest.fixture
def synthetic_correlation_matrix() -> xr.DataArray:
    """Hand-built correlation matrix for thriveline and z-gain tests."""
    return make_correlation_matrix_array()


@pytest.fixture
def correlation_matrix(synthetic_correlation_matrix: xr.DataArray) -> CorrelationMatrix:
    """CorrelationMatrix wrapper around the synthetic matrix."""
    return CorrelationMatrix(
        synthetic_correlation_matrix,
        covariate="age",
        bandwidth=1,
        estimated_range=(0, 5),
    )


@pytest.fixture
def blr_correlation_matrix(correlation_matrix_array_factory) -> CorrelationMatrix:
    """Correlation matrix whose response vars match norm_data_from_arrays."""
    matrix = correlation_matrix_array_factory(
        response_vars=["response_var_0"],
        covariate="covariate_0",
    )
    return CorrelationMatrix(
        matrix,
        covariate="covariate_0",
        bandwidth=1,
        estimated_range=(0, 5),
    )


@pytest.fixture
def blr_longitudinal_dataframe() -> pd.DataFrame:
    """Longitudinal DataFrame compatible with fitted_norm_blr_model."""
    return make_blr_longitudinal_dataframe()


@pytest.fixture
def blr_predicted_longitudinal_norm_data(
    blr_longitudinal_dataframe: pd.DataFrame,
) -> NormData:
    """Predicted longitudinal NormData for BLR-backed score tests."""
    return make_blr_predicted_norm_data(blr_longitudinal_dataframe)


@pytest.fixture
def longitudinal_cohort_norm_data() -> NormData:
    """Larger longitudinal cohort for CorrelationMatrix.compute."""
    return make_longitudinal_normdata()


@pytest.fixture
def batch_effect_model() -> BatchEffectModel:
    """Stub model exposing batch-effect metadata."""
    return BatchEffectModel()


@pytest.fixture
def longitudinal_dataframe_factory():
    """Return the longitudinal DataFrame builder."""
    return make_longitudinal_dataframe


@pytest.fixture
def predicted_norm_data_factory():
    """Return the predicted NormData builder."""
    return make_predicted_norm_data


@pytest.fixture
def blr_longitudinal_dataframe_factory():
    """Return the BLR-compatible longitudinal DataFrame builder."""
    return make_blr_longitudinal_dataframe


@pytest.fixture
def blr_predicted_norm_data_factory():
    """Return the BLR-compatible predicted NormData builder."""
    return make_blr_predicted_norm_data


@pytest.fixture
def correlation_matrix_array_factory():
    """Return the synthetic correlation-matrix builder."""
    return make_correlation_matrix_array
