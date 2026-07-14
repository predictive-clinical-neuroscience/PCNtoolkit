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
    splits = norm_data_from_arrays.train_test_split(splits=split_ratio, split_names=("train", "val"))

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
        assert split.coords["covariates"].to_numpy().tolist() == [f"covariate_{i}" for i in range(n_covariates)]
        assert split.coords["batch_effect_dims"].to_numpy().tolist() == [
            f"batch_effect_{i}" for i in range(len(batch_effect_values))
        ]

    # Check if stratification worked
    original_batch_effects = norm_data_from_arrays.batch_effects.data

    # Check column-wise value distributions
    for col_idx in range(original_batch_effects.shape[1]):
        original_values, original_counts = np.unique(original_batch_effects[:, col_idx], return_counts=True)
        original_frequencies = original_counts / len(original_batch_effects)

        for split in splits:
            split_values, split_counts = np.unique(split.batch_effects.data[:, col_idx], return_counts=True)
            split_frequencies = split_counts / len(split.batch_effects.data)

            # Check if all unique values from original exist in split
            assert np.all(np.isin(original_values, split_values)), f"Split missing some values from column {col_idx}"

            # Check if frequencies are approximately equal (within 10% tolerance)
            for val, orig_freq in zip(original_values, original_frequencies):
                split_freq = split_frequencies[split_values == val][0]
                np.testing.assert_allclose(
                    split_freq, orig_freq, atol=0.1, err_msg=f"Frequency mismatch for value {val} in column {col_idx}"
                )

    # Check if all unique batch effects are present in both splits
    original_unique_batch_effects = np.unique(original_batch_effects, axis=0)
    for split in splits:
        split_unique_batch_effects = np.unique(split.batch_effects.data, axis=0)
        assert np.all(np.isin(original_unique_batch_effects, split_unique_batch_effects))

        # Check if the data in splits is a subset of the original data
        original_data = np.hstack(
            (
                norm_data_from_arrays.X.data,
                norm_data_from_arrays.Y.data,
                norm_data_from_arrays.batch_effects.data,
            )
        )
        for split in splits:
            split_data = np.hstack((split.X.data, split.Y.data, split.batch_effects.data))
            assert np.all(np.isin(split_data, original_data))

        # Check if the attributes are preserved in the splits
        for split in splits:
            assert split.name in ["train", "val"]
            for key in norm_data_from_arrays.attrs:
                if key != "name":
                    assert split.attrs[key] == norm_data_from_arrays.attrs[key]

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


def test_merge(norm_data_from_arrays: NormData):
    datacopy = copy.deepcopy(norm_data_from_arrays)
    merged = datacopy.merge(norm_data_from_arrays)
    assert merged.X.to_numpy().shape[0] == 2 * norm_data_from_arrays.X.to_numpy().shape[0]


def test_to_dataframe():
    # create data with duplicate subject IDs
    n_subjects = 100
    n_samples = 150

    subjects = np.random.choice(n_subjects, n_samples, replace=True)
    coefs = np.random.rand(n_subjects, 2)
    covariates = np.random.choice(10, n_samples, replace=True) + 1
    responses1 = covariates + 0.1 * covariates * coefs[subjects, 0] + np.random.randn(n_samples) * 0.03
    responses2 = covariates + 0.1 * covariates * coefs[subjects, 1] + np.random.randn(n_samples) * 0.03

    df = pd.DataFrame(
        {"subjects": subjects, "age": covariates, "resp1": responses1, "resp2": responses2},
    )
    data = NormData.from_dataframe(
        name="dummy", dataframe=df, covariates=["age"], response_vars=["resp1", "resp2"], subject_ids="subjects"
    )

    print(df)
    print(data.to_dataframe())


def traceable_normdata(n: int = 10) -> NormData:
    """Build data where the age encodes the subject: sub-03 always has age 3.

    Any row that pairs a subject with another subject's data is therefore
    directly detectable, without having to track row positions by hand.
    """
    df = pd.DataFrame(
        {
            "sub_id": [f"sub-{i:02d}" for i in range(n)],
            "age": range(n),
            "response_var_0": [100 + i for i in range(n)],
            "site": ["A", "B"] * (n // 2),
        }
    )
    return NormData.from_dataframe(
        name="traceable",
        dataframe=df,
        covariates=["age"],
        response_vars=["response_var_0"],
        batch_effects=["site"],
        subject_ids="sub_id",
    )


def subject_of(observation_label: str) -> int:
    """Recover the subject number from a 'sub-XX' label."""
    return int(observation_label.split("-")[1])


def test_to_dataframe_aligns_subject_ids_with_their_own_data():
    """to_dataframe must not pair a subject with another subject's covariates.

    The pivoted blocks come back sorted by observation, so a block left in the
    dataset's own (unsorted) order would be silently misaligned against them.
    train_test_split produces exactly such an unsorted order.
    """
    train, _ = traceable_normdata().train_test_split(0.6)
    assert not np.array_equal(train.observations.to_numpy(), np.sort(train.observations.to_numpy())), (
        "test is vacuous unless the observations are unsorted"
    )

    df = train.to_dataframe()

    for label, subject, age in zip(df.index, df[("subject_ids", "subject_ids")], df[("X", "age")]):
        assert subject_of(subject) == int(age), f"{subject} was paired with age {age}"
        assert int(label) == subject_of(subject), f"row labelled {label} holds {subject}"


def test_merge_preserves_observation_labels():
    """Splitting a dataset and merging it back must reconstruct the original.

    Regenerating the labels on merge silently reassigns an observation to a
    different subject.
    """
    data = traceable_normdata()
    train, test = data.train_test_split(0.6)

    merged = train.merge(test)

    before = dict(zip(data.observations.to_numpy().tolist(), data.subject_ids.to_numpy().tolist()))
    after = dict(zip(merged.observations.to_numpy().tolist(), merged.subject_ids.to_numpy().tolist()))
    assert after == before


def test_merge_keeps_observation_labels_unique_on_collision():
    """Merging datasets that share labels must not produce duplicates.

    Independently created datasets both start numbering at 0, so their labels
    collide. Duplicates make to_dataframe raise and turn the joins in
    save_zscores into a cartesian product.
    """
    data = traceable_normdata()

    merged = copy.deepcopy(data).merge(data)

    observations = merged.observations.to_numpy()
    assert len(np.unique(observations)) == observations.size
    merged.to_dataframe()  # raises on duplicate labels


def test_netcdf(norm_data_from_arrays: NormData):
    norm_data_from_arrays.to_netcdf(os.path.join(gettempdir(), "test.nc"))
    norm_data_from_netcdf = NormData.from_netcdf("from_arrays", os.path.join(gettempdir(), "test.nc"))

    # Check if the two norm data objects are equal
    # Checks for data variables and coordinates.
    assert norm_data_from_arrays.equals(norm_data_from_netcdf)

    # Check if the two norm data objects are identical
    # Additionally checks for object's name and attributes.
    assert norm_data_from_arrays.identical(norm_data_from_netcdf)
