import numpy as np
import pandas as pd
import pytest

"""
This file contains pytest fixtures for generating and using synthetic data in the PCNtoolkit.

The fixtures defined here include:
1. Number of training, testing, and transfer data points
2. Number of covariates and response variables
3. Functions to generate numpy arrays and pandas DataFrames

These fixtures are used to create consistent and controlled datasets for testing
"""

# ----------- Constants -----------


@pytest.fixture(scope="session")
def n_train_subjects():
    return 1500


@pytest.fixture(scope="session")
def n_test_subjects():
    return 1000


@pytest.fixture(scope="session")
def n_transfer_subjects():
    return 500


@pytest.fixture(scope="session")
def n_covariates():
    return 2


@pytest.fixture(scope="session")
def n_response_vars():
    return 2


@pytest.fixture(scope="session")
def batch_effect_values():
    return [[0, 1], [0, 1, 2]]


# ----------- Generate data-----------
def generate_covariates(n_subjects, n_covariates):
    return np.random.rand(n_subjects, n_covariates)


def generate_response_vars(n_subjects, n_response_vars, X, seed=42):
    out = np.random.randn(n_subjects, n_response_vars)

    noise_coef = np.array([2.3, 1.5])
    slope_coefs = np.array([[1, 0.5, 0.3], [1, 0.5, 0.3]])
    for i in range(n_response_vars):
        out[:, i] = out[:, i] * noise_coef[i] * X[:, 0]

    for i in range(n_response_vars):
        out[:, i] = out[:, i] + slope_coefs[i, 0] + X[:, 0] * \
            slope_coefs[i, 1] + 0.3 * X[:, 0] ** 2 * slope_coefs[i, 2]
    return np.square(out)


def generate_batch_effects(n_subjects, batch_effect_values):
    batch_effects = []
    for batch_effect in batch_effect_values:
        batch_effects.append(np.random.choice(
            batch_effect, (n_subjects, 1)).astype(int))
    return np.concatenate(batch_effects, axis=1)


# ----------- Create Arrays -----------
def np_arrays(
        n_subjects,
        n_covariates,
        n_response_vars,
        batch_effect_values):
    np.random.seed(42)
    X = generate_covariates(n_subjects, n_covariates)
    y = generate_response_vars(n_subjects, n_response_vars, X)
    batch_effects = generate_batch_effects(n_subjects, batch_effect_values)
    return X, y, batch_effects


@pytest.fixture(scope="module")
def train_arrays(
        n_train_subjects,
        n_covariates,
        n_response_vars,
        batch_effect_values):
    X_train, y_train, batch_effects_train = np_arrays(
        n_train_subjects, n_covariates, n_response_vars, batch_effect_values)
    return X_train, y_train, batch_effects_train


@pytest.fixture(scope="module")
def test_arrays(
        n_test_subjects,
        n_covariates,
        n_response_vars,
        batch_effect_values):
    X_test, y_test, batch_effects_test = np_arrays(
        n_test_subjects, n_covariates, n_response_vars, batch_effect_values)
    return X_test, y_test, batch_effects_test


@pytest.fixture(scope="module")
def transfer_arrays(
        n_transfer_subjects: int,
        n_covariates: int,
        n_response_vars: int,
        batch_effect_values: list[list[int]]):
    # Generate transfer data arrays using the sliced batch effects
    X_transfer, y_transfer, batch_effects_transfer = np_arrays(
        n_transfer_subjects,
        n_covariates,
        n_response_vars,
        batch_effect_values,
    )
    return X_transfer, y_transfer, batch_effects_transfer


# ----------- Create Dataframes -----------
def dataframe(n_subjects, n_covariates, n_response_vars, batch_effect_values):
    X, y, batch_effects = np_arrays(
        n_subjects, n_covariates, n_response_vars, batch_effect_values)
    X_columns = [f"covariate_{i}" for i in range(X.shape[1])]
    y_columns = [f"response_var_{i}" for i in range(y.shape[1])]
    batch_effect_columns = [
        f"batch_effect_{i}" for i in range(len(batch_effect_values))]
    all_columns = X_columns + y_columns + batch_effect_columns
    if len(y.shape) == 1:
        y = y[:, None]
    return pd.DataFrame(
        np.concatenate([X, y, batch_effects.astype(object)], axis=1),
        columns=all_columns,
    )


@pytest.fixture(scope="module")
def train_dataframe(n_train_subjects, n_covariates, n_response_vars, batch_effect_values):
    dataframe_train = dataframe(
        n_train_subjects, n_covariates, n_response_vars, batch_effect_values)
    return dataframe_train


@pytest.fixture(scope="module")
def test_dataframe(n_test_subjects, n_covariates, n_response_vars, batch_effect_values):
    dataframe_test = dataframe(
        n_test_subjects, n_covariates, n_response_vars, batch_effect_values)
    return dataframe_test
