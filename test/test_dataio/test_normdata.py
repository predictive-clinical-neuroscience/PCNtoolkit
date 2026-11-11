import copy
import os
from tempfile import gettempdir

import numpy as np
import pytest

from test.fixtures.norm_data_fixtures import *

"""
This file contains tests for the NormData class in the PCNtoolkit.

The tests cover the following aspects:
1. Creating NormData objects from numpy arrays and pandas DataFrames
2. Verifying the structure and properties of created NormData objects
3. Splitting NormData into training and validation sets
4. Checking the stratification of splits

These tests ensure that the NormData class correctly handles data input,
maintains data integrity, and performs proper data splitting and stratification.
"""

# Attributes that summarise the observations: they must change with a subset.
SUMMARY_ATTRS = (
    "unique_batch_effects",
    "batch_effect_counts",
    "covariate_ranges",
    "batch_effect_covariate_ranges",
)


def assert_summaries_describe(data: NormData) -> None:
    """Assert that the counts and covariate ranges match the data itself.

    Parameters
    ----------
    data : NormData
        The data to check, usually a subset of a bigger dataset.
    """
    n_observations = data.sizes["observations"]
    for dim in data.batch_effect_dims.to_numpy():
        present = data.batch_effects.sel(batch_effect_dims=dim).to_numpy()
        assert set(data.unique_batch_effects[dim]) == set(present)
        assert sum(data.batch_effect_counts[dim].values()) == n_observations
        assert set(data.batch_effect_covariate_ranges[dim]) == set(present)
    for cov in data.covariates.to_numpy():
        values = data.X.sel(covariates=cov).to_numpy()
        assert data.covariate_ranges[cov]["min"] == pytest.approx(values.min())
        assert data.covariate_ranges[cov]["max"] == pytest.approx(values.max())


@pytest.mark.parametrize(
    "norm_data_fixture, n_batch_effect_dims",
    [("norm_data_from_arrays", 2), ("norm_data_from_dataframe", 2)],
)
def test_norm_data_creation(
    request,
    norm_data_fixture,
    n_batch_effect_dims,
    n_covariates,
    n_train_subjects,
    batch_effect_values,
    n_response_vars,
):
    norm_data = request.getfixturevalue(norm_data_fixture)

    assert norm_data.X.shape == (n_train_subjects, n_covariates)
    assert norm_data.Y.shape == (n_train_subjects, n_response_vars)
    assert norm_data.batch_effects.shape == (n_train_subjects, len(batch_effect_values))
    assert norm_data.covariates.shape == (n_covariates,)
    assert norm_data.batch_effect_dims.shape == (n_batch_effect_dims,)
    assert norm_data.coords["observations"].shape == (n_train_subjects,)
    assert norm_data.coords["covariates"].shape == (n_covariates,)
    assert norm_data.coords["batch_effect_dims"].shape == (n_batch_effect_dims,)


@pytest.mark.parametrize("split_ratio", [(0.5, 0.5), (0.7, 0.3), (0.8, 0.2)])
def test_split_with_stratify(
    norm_data_from_arrays,
    n_covariates,
    n_train_subjects,
    n_response_vars,
    split_ratio,
    batch_effect_values,
):
    splits = norm_data_from_arrays.train_test_split(
        splits=split_ratio, split_names=("train", "val")
    )

    # Check basic split properties
    assert len(splits) == 2
    assert splits[0].name == "train"
    assert splits[1].name == "val"

    # Check if total samples in splits equal original samples
    assert splits[0].X.shape[0] + splits[1].X.shape[0] == n_train_subjects

    for i, split in enumerate(splits):
        expected_samples = int(n_train_subjects * split_ratio[i])
        assert split.X.shape == (expected_samples, n_covariates)
        assert split.Y.shape == (expected_samples, n_response_vars)
        assert split.batch_effects.shape == (expected_samples, len(batch_effect_values))
        assert split.covariates.shape == (n_covariates,)
        assert split.batch_effect_dims.shape == (len(batch_effect_values),)
        assert split.coords["observations"].shape == (expected_samples,)
        assert split.coords["covariates"].to_numpy().tolist() == [
            f"covariate_{i}" for i in range(n_covariates)
        ]
        assert split.coords["batch_effect_dims"].to_numpy().tolist() == [
            f"batch_effect_{i}" for i in range(len(batch_effect_values))
        ]

    # Check if stratification worked
    original_batch_effects = norm_data_from_arrays.batch_effects.data

    # Check column-wise value distributions
    for col_idx in range(original_batch_effects.shape[1]):
        original_values, original_counts = np.unique(
            original_batch_effects[:, col_idx], return_counts=True
        )
        original_frequencies = original_counts / len(original_batch_effects)

        for split in splits:
            split_values, split_counts = np.unique(
                split.batch_effects.data[:, col_idx], return_counts=True
            )
            split_frequencies = split_counts / len(split.batch_effects.data)

            # Check if all unique values from original exist in split
            assert np.all(np.isin(original_values, split_values)), (
                f"Split missing some values from column {col_idx}"
            )

            # Check if frequencies are approximately equal (within 10% tolerance)
            for val, orig_freq in zip(original_values, original_frequencies):
                split_freq = split_frequencies[split_values == val][0]
                np.testing.assert_allclose(
                    split_freq,
                    orig_freq,
                    atol=0.1,
                    err_msg=f"Frequency mismatch for value {val} in column {col_idx}",
                )

    # Check if all unique batch effects are present in both splits
    original_unique_batch_effects = np.unique(original_batch_effects, axis=0)
    for split in splits:
        split_unique_batch_effects = np.unique(split.batch_effects.data, axis=0)
        assert np.all(
            np.isin(original_unique_batch_effects, split_unique_batch_effects)
        )

        # Check if the data in splits is a subset of the original data
        original_data = np.hstack(
            (
                norm_data_from_arrays.X.data,
                norm_data_from_arrays.Y.data,
                norm_data_from_arrays.batch_effects.data,
            )
        )
        for split in splits:
            split_data = np.hstack(
                (split.X.data, split.Y.data, split.batch_effects.data)
            )
            assert np.all(np.isin(split_data, original_data))

        # Check if the attributes are preserved in the splits. The counts and
        # covariate ranges are the exception: they describe the split itself.
        for split in splits:
            assert split.name in ["train", "val"]
            for key in norm_data_from_arrays.attrs:
                if key not in ("name", *SUMMARY_ATTRS):
                    assert split.attrs[key] == norm_data_from_arrays.attrs[key]
            assert_summaries_describe(split)

        # Check if the coordinate names are preserved
        for split in splits:
            assert list(split.coords) == list(norm_data_from_arrays.coords)

        # Check if the data variable names are preserved
        for split in splits:
            assert list(split.data_vars) == list(norm_data_from_arrays.data_vars)

        # Check if the dimensions are preserved
        for split in splits:
            assert list(split.dims) == list(norm_data_from_arrays.dims)


def test_chunk(norm_data_from_arrays):
    chunks = norm_data_from_arrays.chunk(n_chunks=2)
    for i, chunk in enumerate(chunks):
        assert chunk.response_vars == ["response_var_{}".format(i)]


@pytest.mark.parametrize("how", ["isel", "sel", "where_drop"])
def test_subset_describes_its_own_data(norm_data_from_arrays: NormData, how: str):
    """A subset must report its own counts and covariate ranges.

    Regression test for issue #640: a subset kept the size, sites and
    covariate ranges of the full dataset, so a model fitted on it stored
    those wrong values.

    Parameters
    ----------
    norm_data_from_arrays : NormData
        Dataset with covariates drawn uniformly between 0 and 1.
    how : str
        Which xarray selection to use. All of them must give the same result.
    """
    parent_attrs = copy.deepcopy(norm_data_from_arrays.attrs)
    # Keep the lower part of the first covariate, like keeping subjects under 30.
    mask = (norm_data_from_arrays.X.isel(covariates=0) < 0.3).to_numpy()
    if how == "isel":
        subset = norm_data_from_arrays.isel(observations=mask)
    elif how == "sel":
        labels = norm_data_from_arrays.observations.to_numpy()[mask]
        subset = norm_data_from_arrays.sel(observations=labels)
    else:
        cond = norm_data_from_arrays.X.isel(covariates=0, drop=True) < 0.3
        subset = norm_data_from_arrays.where(cond, drop=True)

    assert subset.sizes["observations"] == mask.sum()
    assert subset.covariate_ranges["covariate_0"]["max"] < 0.3
    assert_summaries_describe(subset)

    # The full dataset must not change when we take a subset or rename it.
    subset.attrs["name"] = "subset"
    assert norm_data_from_arrays.attrs.keys() == parent_attrs.keys()
    for key, value in parent_attrs.items():
        assert norm_data_from_arrays.attrs[key] == value


def test_subset_forgets_dropped_batch_effects(norm_data_from_arrays: NormData):
    """A subset must not list sites it no longer contains.

    Otherwise ``synthesize`` can generate subjects from sites the model was
    never fitted on.

    Parameters
    ----------
    norm_data_from_arrays : NormData
        Dataset whose ``batch_effect_1`` has the levels "0", "1" and "2".
    """
    site = norm_data_from_arrays.batch_effects.sel(batch_effect_dims="batch_effect_1")
    subset = norm_data_from_arrays.isel(observations=(site != "2").to_numpy())

    assert sorted(subset.unique_batch_effects["batch_effect_1"]) == ["0", "1"]
    assert "2" not in subset.batch_effect_counts["batch_effect_1"]
    assert "2" not in subset.batch_effect_covariate_ranges["batch_effect_1"]


def test_selecting_one_observation_does_not_fail(norm_data_from_arrays: NormData):
    """Selecting a single observation drops the observations dimension.

    There is then no table left to summarise, so the selection must still work
    and simply keep the summaries it had.

    Parameters
    ----------
    norm_data_from_arrays : NormData
        Any dataset.
    """
    single = norm_data_from_arrays.isel(observations=0)
    assert "observations" not in single.dims


def test_model_fitted_on_subset_stores_subset_info(norm_data_from_arrays: NormData):
    """A model fitted on a subset must store the subset's counts and ranges.

    This is the scenario of issue #640: before the fix ``n_fit_observations``
    was the size of the full dataset and ``covariate_ranges`` its full range,
    so centile plots extrapolated and ``merge`` over-weighted the model.

    Parameters
    ----------
    norm_data_from_arrays : NormData
        Dataset with covariates drawn uniformly between 0 and 1.
    """
    from pcntoolkit.normative_model import NormativeModel
    from pcntoolkit.regression_model.blr import BLR

    mask = (norm_data_from_arrays.X.isel(covariates=0) < 0.3).to_numpy()
    subset = norm_data_from_arrays.isel(observations=mask)
    subset_max = float(subset.X.sel(covariates="covariate_0").max())

    model = NormativeModel(BLR(), savemodel=False, saveresults=False, saveplots=False)
    model.fit(subset)

    assert model.n_fit_observations == mask.sum()
    assert model.covariate_ranges["covariate_0"]["max"] == pytest.approx(subset_max)


def test_merge(norm_data_from_arrays: NormData):
    datacopy = copy.deepcopy(norm_data_from_arrays)
    merged = datacopy.merge(norm_data_from_arrays)
    assert (
        merged.X.to_numpy().shape[0] == 2 * norm_data_from_arrays.X.to_numpy().shape[0]
    )


def test_merge_preserves_observations(norm_data_from_arrays: NormData):
    """An observation must keep pointing at the same subject after a merge.

    Splitting the data and merging it back must therefore return the
    same observations as the data we started with.
    """
    # Split the data in two, then put the same rows back together.
    train, test = norm_data_from_arrays.train_test_split(0.6)
    merged = train.merge(test)

    # Ask both datasets for the same observations and compare the covariates they
    # hand back. If the observations are not preserved, then the covariates
    # will be different.
    observations = merged.observations
    assert np.allclose(
        merged.X.sel(observations=observations).to_numpy(),
        norm_data_from_arrays.X.sel(observations=observations).to_numpy(),
    )


def test_merge_shifts_duplicate_observations(norm_data_from_arrays: NormData):
    """Merging two datasets that use the same observations must not duplicate
    them. Instead the merged dataset must shift the observations of the second
    dataset to make them unique.
    """
    # Merging the data with a copy of itself: every observation collides.
    merged = copy.deepcopy(norm_data_from_arrays).merge(norm_data_from_arrays)

    # The shifted observations must not repeat any of the originals.
    observations = merged.observations.to_numpy()
    assert len(np.unique(observations)) == observations.size
    merged.to_dataframe()  # raises on duplicate labels


def test_merge_knows_all_batch_effects(
    norm_data_from_arrays: NormData, transfer_norm_data_from_arrays: NormData
):
    """Merging A and B must give data that knows the batch effects of both."""
    # The transfer data was collected at a site the training data never saw.
    merged = norm_data_from_arrays.merge(transfer_norm_data_from_arrays)

    for dim in merged.batch_effect_dims.to_numpy():
        # The merged data must know every site of A and every site of B.
        both = set(norm_data_from_arrays.unique_batch_effects[dim]) | set(
            transfer_norm_data_from_arrays.unique_batch_effects[dim]
        )
        assert set(merged.unique_batch_effects[dim]) == both

        # And it must have counted all of its rows, not just the ones from A.
        assert sum(merged.batch_effect_counts[dim].values()) == merged.observations.size


def test_to_dataframe():
    # create data with duplicate subject IDs
    n_subjects = 100
    n_samples = 150

    subjects = np.random.choice(n_subjects, n_samples, replace=True)
    coefs = np.random.rand(n_subjects, 2)
    covariates = np.random.choice(10, n_samples, replace=True) + 1
    responses1 = (
        covariates
        + 0.1 * covariates * coefs[subjects, 0]
        + np.random.randn(n_samples) * 0.03
    )
    responses2 = (
        covariates
        + 0.1 * covariates * coefs[subjects, 1]
        + np.random.randn(n_samples) * 0.03
    )

    df = pd.DataFrame(
        {
            "subjects": subjects,
            "age": covariates,
            "resp1": responses1,
            "resp2": responses2,
        },
    )
    data = NormData.from_dataframe(
        name="dummy",
        dataframe=df,
        covariates=["age"],
        response_vars=["resp1", "resp2"],
        subject_ids="subjects",
    )

    print(df)
    print(data.to_dataframe())


def test_to_dataframe_preserves_ordered_observations(norm_data_from_arrays: NormData):
    """The contents of to_dataframe come back sorted, so the observations must be sorted too.

    X, Y and the batch effects come back ordered 0,1,2,3,... If the observations
    keep the old unsorted order (3,1,5,10,...), each row is labelled with someone
    else's observation.
    """
    # Splitting shuffles the observations, so they are no longer sorted.
    train, _ = norm_data_from_arrays.train_test_split(0.6)

    # The bug only shows up on unsorted observations, so make sure they are.
    observations = train.observations.to_numpy()
    assert not np.array_equal(observations, np.sort(observations))

    df = train.to_dataframe()

    # Our simulated data here numbers the subjects 0,1,2,..., which on purpose
    # is the same as the observations.
    # So if to_dataframe kept its observations sorted , then
    # the observations MUST equal its subject id.
    assert np.array_equal(
        df.index.to_numpy().astype(int),
        df[("subject_ids", "subject_ids")].to_numpy().astype(int),
    )


def test_netcdf(norm_data_from_arrays: NormData):
    norm_data_from_arrays.to_netcdf(os.path.join(gettempdir(), "test.nc"))
    norm_data_from_netcdf = NormData.from_netcdf(
        "from_arrays", os.path.join(gettempdir(), "test.nc")
    )

    # Check if the two norm data objects are equal
    # Checks for data variables and coordinates.
    assert norm_data_from_arrays.equals(norm_data_from_netcdf)

    # Check if the two norm data objects are identical
    # Additionally checks for object's name and attributes.
    assert norm_data_from_arrays.identical(norm_data_from_netcdf)
