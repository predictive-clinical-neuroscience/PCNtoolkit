import shutil

from pcntoolkit.util.runner import Runner
from test.fixtures.test_model_fixtures import *


def cleanup(model, runner):
    shutil.rmtree(os.path.join(model.save_dir))
    shutil.rmtree(os.path.join(runner.log_dir))
    shutil.rmtree(os.path.join(runner.temp_dir))


def test_runner_fit(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    runner.fit(new_norm_test_model, norm_data_from_arrays, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "model", "normative_model.json"))
    cleanup(new_norm_test_model, runner)


def test_runner_fit_kfold(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    runner.fit(new_norm_test_model, norm_data_from_arrays, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_0", "model", "normative_model.json"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_1", "model", "normative_model.json"))
    cleanup(new_norm_test_model, runner)


def test_runner_predict(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    runner.predict(fitted_norm_test_model, norm_data_from_arrays, observe=True)
    assert fitted_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(fitted_norm_test_model.save_dir, "model", "normative_model.json"))
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


def test_runner_predict_kfold(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    # assert this throws an error
    with pytest.raises(ValueError):
        runner.predict(new_norm_test_model, norm_data_from_arrays, observe=True)


def test_runner_fit_predict(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    runner = Runner(cross_validate=False, parallelize=False)
    runner.fit_predict(new_norm_test_model, train, test, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "model", "normative_model.json"))
    assert os.path.exists(
        os.path.join(
            new_norm_test_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(new_norm_test_model.save_dir, "plots", f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png")
    )
    cleanup(new_norm_test_model, runner)


def test_runner_fit_predict_kfold(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    runner.fit_predict(new_norm_test_model, train, test, observe=True)
    assert new_norm_test_model.is_fitted
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_0", "model", "normative_model.json"))
    assert os.path.exists(os.path.join(new_norm_test_model.save_dir, "folds", "fold_1", "model", "normative_model.json"))
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


def test_runner_extend(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    # Create extend dir:
    extend_dir = os.path.join(fitted_norm_test_model.save_dir, "extend")
    if os.path.exists(extend_dir):
        shutil.rmtree(extend_dir)
    os.makedirs(extend_dir, exist_ok=True)
    extended_model = runner.extend(fitted_norm_test_model, norm_data_from_arrays, extend_dir, observe=True)
    assert isinstance(extended_model, NormativeModel)
    assert extended_model.is_fitted
    assert os.path.exists(os.path.join(extended_model.save_dir, "model", "normative_model.json"))
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


def test_runner_extend_predict(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    extend_dir = os.path.join(fitted_norm_test_model.save_dir, "extend_predict")
    if os.path.exists(extend_dir):
        shutil.rmtree(extend_dir)
    os.makedirs(extend_dir, exist_ok=True)
    extended_model = runner.extend_predict(fitted_norm_test_model, train, test, extend_dir, observe=True)
    assert isinstance(extended_model, NormativeModel)
    assert extended_model.is_fitted
    assert os.path.exists(os.path.join(extended_model.save_dir, "model", "normative_model.json"))
    assert os.path.exists(
        os.path.join(
            extended_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(extended_model.save_dir, "plots", f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png")
    )
    cleanup(extended_model, runner)


def test_runner_extend_predict_kfold(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    extend_dir = os.path.join(fitted_norm_test_model.save_dir, "extend_predict_kfold")
    if os.path.exists(extend_dir):
        shutil.rmtree(extend_dir)
    os.makedirs(extend_dir, exist_ok=True)
    extended_model = runner.extend_predict(fitted_norm_test_model, norm_data_from_arrays, None, extend_dir, observe=True)
    assert isinstance(extended_model, NormativeModel)
    assert extended_model.is_fitted
    assert os.path.exists(os.path.join(extended_model.save_dir, "model", "normative_model.json"))
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


def test_runner_transfer(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    transfer_dir = os.path.join(fitted_norm_test_model.save_dir, "transfer")
    if os.path.exists(transfer_dir):
        shutil.rmtree(transfer_dir)
    os.makedirs(transfer_dir, exist_ok=True)
    transferred_model = runner.transfer(fitted_norm_test_model, norm_data_from_arrays, transfer_dir, observe=True)
    assert isinstance(transferred_model, NormativeModel)
    assert transferred_model.is_fitted
    assert os.path.exists(os.path.join(transferred_model.save_dir, "model", "normative_model.json"))
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


def test_runner_transfer_predict(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=False, parallelize=False)
    train, test = norm_data_from_arrays.train_test_split(splits=[0.2, 0.8])
    transferred_model = runner.transfer_predict(fitted_norm_test_model, train, test, observe=True)
    assert isinstance(transferred_model, NormativeModel)
    assert transferred_model.is_fitted
    assert os.path.exists(os.path.join(transferred_model.save_dir, "model", "normative_model.json"))
    assert os.path.exists(
        os.path.join(
            transferred_model.save_dir,
            "results",
        )
    )
    assert os.path.exists(
        os.path.join(transferred_model.save_dir, "plots", f"centiles_{test.response_vars.values[0]}_{test.name}_harmonized.png")
    )
    cleanup(transferred_model, runner)


def test_runner_transfer_predict_kfold(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    runner = Runner(cross_validate=True, cv_folds=2, parallelize=False)
    transferred_model = runner.transfer_predict(fitted_norm_test_model, norm_data_from_arrays, None, observe=True)
    assert isinstance(transferred_model, NormativeModel)
    assert transferred_model.is_fitted
    assert os.path.exists(os.path.join(transferred_model.save_dir, "model", "normative_model.json"))
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
