import multiprocessing
import shutil
import sys
from multiprocessing.synchronize import Barrier
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcntoolkit.util.runner import Runner
from test.fixtures.test_model_fixtures import *


def cleanup(model, runner):
    shutil.rmtree(os.path.join(model.save_dir))
    shutil.rmtree(os.path.join(runner.log_dir))
    shutil.rmtree(os.path.join(runner.temp_dir))


def test_runner_fit(
    new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=False, parallelize=False)
    runner.fit(new_norm_test_model, norm_data_from_arrays, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(
        os.path.join(new_norm_test_model.save_dir, "model", "normative_model.json")
    )
    cleanup(new_norm_test_model, runner)


def test_runner_fit_kfold(
    new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    runner.fit(new_norm_test_model, norm_data_from_arrays, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_0",
            "model",
            "normative_model.json",
        )
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_1",
            "model",
            "normative_model.json",
        )
    )
    cleanup(new_norm_test_model, runner)


def test_runner_predict(
    fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=False, parallelize=False)
    runner.predict(fitted_norm_test_model, norm_data_from_arrays, observe=True)
    assert fitted_norm_test_model.is_fitted
    assert os.path.exists(
        os.path.join(fitted_norm_test_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            fitted_norm_test_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            fitted_norm_test_model.save_dir,
            "plots",
            f"centiles_{norm_data_from_arrays.response_vars.values[0]}_{norm_data_from_arrays.name}_harmonized.png",
        )
    )
    cleanup(fitted_norm_test_model, runner)


def test_runner_predict_kfold(
    new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    # assert this throws an error
    with pytest.raises(ValueError):
        runner.predict(new_norm_test_model, norm_data_from_arrays, observe=True)


def test_runner_fit_predict(
    new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    runner = Runner(cross_validate=False, parallelize=False)
    runner.fit_predict(new_norm_test_model, train, test, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(
        os.path.join(new_norm_test_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "plots",
            f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png",
        )
    )
    cleanup(new_norm_test_model, runner)


def test_runner_fit_predict_kfold(
    new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    runner.fit_predict(new_norm_test_model, train, test, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_0",
            "model",
            "normative_model.json",
        )
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_1",
            "model",
            "normative_model.json",
        )
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_0",
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_1",
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_0",
            "plots",
            f"centiles_{train.response_vars.values[0]}_{train.name}_fold_0_predict_harmonized.png",
        )
    )
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "folds",
            "fold_1",
            "plots",
            f"centiles_{train.response_vars.values[0]}_{train.name}_fold_1_train_harmonized.png",
        )
    )
    cleanup(new_norm_test_model, runner)


def test_runner_extend(
    fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=False, parallelize=False)
    # Create extend dir:
    extend_dir = os.path.join(fitted_norm_test_model.save_dir, "extend")
    if os.path.exists(extend_dir):
        shutil.rmtree(extend_dir)
    os.makedirs(extend_dir, exist_ok=True)
    extended_model = runner.extend(
        fitted_norm_test_model, norm_data_from_arrays, extend_dir, observe=True
    )
    assert isinstance(extended_model, NormativeModel)
    assert extended_model.is_fitted
    assert os.path.exists(
        os.path.join(extended_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            extended_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            extended_model.save_dir,
            "plots",
            f"centiles_{norm_data_from_arrays.response_vars.values[0]}_from_arrays_+_synthesized_harmonized.png",
        )
    )
    cleanup(extended_model, runner)


def test_runner_extend_predict(
    fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=False, parallelize=False)
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    extend_dir = os.path.join(fitted_norm_test_model.save_dir, "extend_predict")
    if os.path.exists(extend_dir):
        shutil.rmtree(extend_dir)
    os.makedirs(extend_dir, exist_ok=True)
    extended_model = runner.extend_predict(
        fitted_norm_test_model, train, test, extend_dir, observe=True
    )
    assert isinstance(extended_model, NormativeModel)
    assert extended_model.is_fitted
    assert os.path.exists(
        os.path.join(extended_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            extended_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            extended_model.save_dir,
            "plots",
            f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png",
        )
    )
    cleanup(extended_model, runner)


def test_runner_extend_predict_kfold(
    fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    extend_dir = os.path.join(fitted_norm_test_model.save_dir, "extend_predict_kfold")
    if os.path.exists(extend_dir):
        shutil.rmtree(extend_dir)
    os.makedirs(extend_dir, exist_ok=True)
    extended_model = runner.extend_predict(
        fitted_norm_test_model, norm_data_from_arrays, None, extend_dir, observe=True
    )
    assert isinstance(extended_model, NormativeModel)
    assert extended_model.is_fitted
    assert os.path.exists(
        os.path.join(extended_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            extended_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            extended_model.save_dir,
            "plots",
            f"centiles_{norm_data_from_arrays.response_vars.values[0]}_{norm_data_from_arrays.name}_fold_0_predict_harmonized.png",
        )
    )
    cleanup(extended_model, runner)


def test_runner_transfer(
    fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=False, parallelize=False)
    transfer_dir = os.path.join(fitted_norm_test_model.save_dir, "transfer")
    if os.path.exists(transfer_dir):
        shutil.rmtree(transfer_dir)
    os.makedirs(transfer_dir, exist_ok=True)
    transferred_model = runner.transfer(
        fitted_norm_test_model, norm_data_from_arrays, transfer_dir, observe=True
    )
    assert isinstance(transferred_model, NormativeModel)
    assert transferred_model.is_fitted
    assert os.path.exists(
        os.path.join(transferred_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            transferred_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            transferred_model.save_dir,
            "plots",
            f"centiles_{norm_data_from_arrays.response_vars.values[0]}_{norm_data_from_arrays.name}_harmonized.png",
        )
    )
    cleanup(transferred_model, runner)


def test_runner_transfer_predict(
    fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=False, parallelize=False)
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    transferred_model = runner.transfer_predict(
        fitted_norm_test_model, train, test, observe=True
    )
    assert isinstance(transferred_model, NormativeModel)
    assert transferred_model.is_fitted
    assert os.path.exists(
        os.path.join(transferred_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            transferred_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            transferred_model.save_dir,
            "plots",
            f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png",
        )
    )
    cleanup(transferred_model, runner)


def test_runner_transfer_predict_kfold(
    fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData
):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    transferred_model = runner.transfer_predict(
        fitted_norm_test_model, norm_data_from_arrays, None, observe=True
    )
    assert isinstance(transferred_model, NormativeModel)
    assert transferred_model.is_fitted
    assert os.path.exists(
        os.path.join(transferred_model.save_dir, "model", "normative_model.json")
    )
    assert os.path.exists(
        os.path.join(
            transferred_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(
            transferred_model.save_dir,
            "plots",
            f"centiles_{norm_data_from_arrays.response_vars.values[0]}_{norm_data_from_arrays.name}_fold_0_predict_harmonized.png",
        )
    )
    cleanup(transferred_model, runner)


def _save_zscores_one_var(save_dir: str, i: int, n_obs: int, barrier: Barrier) -> None:
    """Save Z-scores of one response variable, like one Runner job does."""
    data = NormData.from_ndarrays(
        "test",
        X=np.zeros((n_obs, 1)),
        Y=np.full((n_obs, 1), float(i)),
        subject_ids=np.array([f"sub-{j}" for j in range(n_obs)]),
    )
    data = data.assign_coords(response_vars=[f"rv_{i}"])
    data["Z"] = data["Y"]
    # The bug is a race, so repeat: with 5 rounds v1.1.2 failed in 7/10 trials,
    # with 50 rounds in 10/10.
    for _ in range(50):
        # All processes write at the same moment. The timeout stops the others
        # from waiting forever if one process crashes.
        barrier.wait(timeout=10)
        data.save_zscores(save_dir)


@pytest.mark.skipif(sys.platform == "win32", reason="fork is not available on Windows")
def test_save_zscores_parallel_no_duplicate_header(tmp_path: Path) -> None:
    """Regression test for #534: parallel Runner jobs saving into one folder must
    give one valid CSV.

    The Runner itself is not used: with parallelize=False its jobs run one after
    another, and parallelize=True needs SLURM. Instead, here we simulate10 processes
    to call save_zscores at the same moment, as parallel jobs do.

    This test fails for pcntoolkit <= 1.1.2.
    """
    n_jobs, n_obs = 10, 50
    # fork instead of spawn, so each process does not re-import pcntoolkit
    # (makes the test faster)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(n_jobs)
    procs = [
        ctx.Process(
            target=_save_zscores_one_var, args=(str(tmp_path), i, n_obs, barrier)
        )
        for i in range(n_jobs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert all(p.exitcode == 0 for p in procs)

    path = os.path.join(tmp_path, "Z_test.csv")
    with open(path) as f:
        n_headers = sum(line.startswith("observations") for line in f)
    assert n_headers == 1
    df = pd.read_csv(path)
    assert len(df) == n_obs
    saved_vars = sorted(c for c in df.columns if c.startswith("rv_"))
    assert saved_vars == sorted(f"rv_{i}" for i in range(n_jobs))
