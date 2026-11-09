import os
import shutil
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pytest

from test.fixtures.data_fixtures import (
    generate_batch_effects,
    generate_covariates,
    generate_response_vars,
)

"""This file contains pytest fixtures for file paths in the PCNtoolkit.

The fixtures defined here include:
1. Log directory
2. Save directory
3. Response file
4. Mask file
5. Covariate file
6. Test covariate file
7. Test response file
"""


MODEL_PATH = Path("tests/fixtures/model.joblib")
LOCK_PATH = Path("tests/fixtures/model.lock")


@pytest.fixture(scope="session")
def log_dir():
    path = os.path.join(gettempdir(), "pcntoolkit_tests", "log_test")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def fit_files(n_train_subjects, n_covariates, n_response_vars, batch_effect_values):
    source_data_dir = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data"
    )
    os.makedirs(source_data_dir, exist_ok=True)
    cov_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates.csv"
    )
    resp_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "responses.csv"
    )
    trbefile_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects.csv"
    )
    covariates = generate_covariates(n_train_subjects, n_covariates)
    responses = generate_response_vars(n_train_subjects, n_response_vars, covariates)
    batch_effects = generate_batch_effects(n_train_subjects, batch_effect_values)
    np.savetxt(cov_path, covariates)
    np.savetxt(resp_path, responses)
    np.savetxt(trbefile_path, batch_effects)
    yield (cov_path, resp_path, trbefile_path)
    os.remove(cov_path)
    os.remove(resp_path)
    os.remove(trbefile_path)


@pytest.fixture(scope="session")
def test_files(n_test_subjects, n_covariates, n_response_vars, batch_effect_values):
    source_data_dir = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data"
    )
    os.makedirs(source_data_dir, exist_ok=True)
    test_cov_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "covariates_test.csv"
    )
    test_resp_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "responses_test.csv"
    )
    test_trbefile_path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "resources", "data", "batch_effects_test.csv"
    )
    test_covariates = generate_covariates(n_test_subjects, n_covariates)
    test_responses = generate_response_vars(
        n_test_subjects, n_response_vars, test_covariates
    )
    test_batch_effects = generate_batch_effects(n_test_subjects, batch_effect_values)
    np.savetxt(test_cov_path, test_covariates)
    np.savetxt(test_resp_path, test_responses)
    np.savetxt(test_trbefile_path, test_batch_effects)
    yield (test_cov_path, test_resp_path, test_trbefile_path)
    os.remove(test_cov_path)
    os.remove(test_resp_path)
    os.remove(test_trbefile_path)


@pytest.fixture(scope="session")
def save_dir():
    path = os.path.join(gettempdir(), "pcntoolkit_tests", "save_load_test")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def maskfile():
    return None


@pytest.fixture(scope="session")
def save_dir_hbr():
    path = os.path.join(gettempdir(), "pcntoolkit_tests", "save_load_test", "hbr")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def save_dir_blr():
    path = os.path.join(gettempdir(), "pcntoolkit_tests", "save_load_test", "blr")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def save_dir_test_model():
    path = os.path.join(
        gettempdir(), "pcntoolkit_tests", "save_load_test", "test_model"
    )
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path
