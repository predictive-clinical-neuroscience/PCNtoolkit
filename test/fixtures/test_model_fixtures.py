import os
import shutil

import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.test_model import TestModel
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


@pytest.fixture
def test_model_args(save_dir_test_model):
    return {
        "savemodel": False,
        "saveresults": False,
        "evaluate_model": False,
        "saveplots": False,
        "save_dir": save_dir_test_model,
        "inscaler": "standardize",
        "outscaler": "standardize",
        "name": "test_model",
        "alg": "test_model",
        "success_ratio": 1.0,
    }


@pytest.fixture
def test_model():
    return TestModel("test_model")


@pytest.fixture
def new_norm_test_model(test_model, save_dir_test_model):
    if os.path.exists(save_dir_test_model):
        shutil.rmtree(save_dir_test_model)
    os.makedirs(save_dir_test_model, exist_ok=True)
    return NormativeModel(test_model, save_dir=save_dir_test_model)


@pytest.fixture
def fitted_norm_test_model(
    new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    if os.path.exists(new_norm_test_model.save_dir):
        shutil.rmtree(new_norm_test_model.save_dir)
    os.makedirs(new_norm_test_model.save_dir, exist_ok=True)
    new_norm_test_model.fit(norm_data_from_arrays)
    return new_norm_test_model
