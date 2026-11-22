Command line interface
======================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/09_command_line_interface.ipynb>`

The PCNtoolkit is a python package, but it can also be used from the
command line.

Here we show how to use the PCNtoolkit from the command line.

Furthermore, you can use this script to generate commands for the
command line interface. (Although if you are able to run this notebook,
why not just use it as a python package?)

.. code:: ipython3

    import os
    import pickle
    import sys
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.model_selection import train_test_split

BLR Example
-----------

Data preparation
~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Download and split data first
    # If you are running this notebook for the first time, you need to download the dataset from github.
    # If you have already downloaded the dataset, you can comment out the following line
    os.makedirs("resources/data", exist_ok=True)
    pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    ).to_csv("resources/data/fcon1000.csv", index=False)

.. code:: ipython3

    data = pd.read_csv("resources/data/fcon1000.csv")

.. code:: ipython3

    # Inspect the data
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(
        data=data, x=("age"), y=("rh_MeanThickness_thickness"), hue=("site"), ax=ax[1]
    )
    ax[1].legend([], [])
    ax[1].set_title("Scatter plot of age vs rh_MeanThickness_thickness")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("rh_MeanThickness_thickness")
    sns.countplot(data=data, y="site", hue="sex", ax=ax[0], orient="h")
    ax[0].legend(title="Sex")
    ax[0].set_title("Count of sites")
    ax[0].set_xlabel("Site")
    ax[0].set_ylabel("Count")
    plt.show()



.. image:: 09_command_line_interface_files/09_command_line_interface_6_0.png


.. code:: ipython3

    # Split into X, y, and batch effects
    covariate_columns = ["age"]
    batch_effect_columns = ["sex", "site"]
    response_columns = ["rh_MeanThickness_thickness", "WM-hypointensities"]
    
    X = data[covariate_columns]
    Y = data[response_columns]
    batch_effects = data[batch_effect_columns]
    
    batch_effects_strings = [str(b[0]) + " " + str(b[1]) for b in batch_effects.values]
    
    # Split into train and test set
    trainidx, testidx = train_test_split(
        data.index, test_size=0.2, random_state=42, stratify=batch_effects_strings
    )
    train_X = X.loc[trainidx]
    train_Y = Y.loc[trainidx]
    train_batch_effects = batch_effects.loc[trainidx]
    
    test_X = X.loc[testidx]
    test_Y = Y.loc[testidx]
    test_batch_effects = batch_effects.loc[testidx]

.. code:: ipython3

    # Save stuff
    root_dir = os.path.join("resources", "cli_example")
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    resp = os.path.abspath(os.path.join(data_dir, "responses.csv"))
    cov = os.path.abspath(os.path.join(data_dir, "covariates.csv"))
    be = os.path.abspath(os.path.join(data_dir, "batch_effects.csv"))
    
    t_resp = os.path.abspath(os.path.join(data_dir, "test_responses.csv"))
    t_cov = os.path.abspath(os.path.join(data_dir, "test_covariates.csv"))
    t_be = os.path.abspath(os.path.join(data_dir, "test_batch_effects.csv"))
    
    
    with open(cov, "wb") as f:
        pickle.dump(train_X, f)
    with open(resp, "wb") as f:
        pickle.dump(train_Y, f)
    with open(be, "wb") as f:
        pickle.dump(train_batch_effects, f)
    with open(t_cov, "wb") as f:
        pickle.dump(test_X, f)
    with open(t_resp, "wb") as f:
        pickle.dump(test_Y, f)
    with open(t_be, "wb") as f:
        pickle.dump(test_batch_effects, f)

BLR configuration
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    alg = "blr"
    func = "fit_predict"
    
    # normative model configuration
    save_dir = os.path.join(root_dir, "blr_cli", "save_dir")
    savemodel = True
    saveresults = True
    basis_function = "linear"
    inscaler = "standardize"
    outscaler = "standardize"
    
    # Regression model configuration
    optimizer = "l-bfgs-b"
    n_iter = 200
    heteroskedastic = True
    fixed_effect = True
    warp = "WarpSinhArcsinh"
    warp_reparam = True
    
    # runner configuration
    cross_validate = True
    cv_folds = 5
    parallelize = False
    job_type = "local"
    n_jobs = 2
    temp_dir = os.path.join(root_dir, "temp")
    log_dir = os.path.join(root_dir, "log")
    python_env = os.path.join(os.path.dirname(os.path.dirname(sys.executable)))

Constructing command
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    command = "normative"
    args = f"-a {alg} -f {func} -c {cov} -r {resp} -t {t_resp} -e {t_cov} -k {cv_folds}"
    kwargs = f"be={be} t_be={t_be}"
    normative_model_kwargs = f"save_dir={save_dir} savemodel={savemodel} saveresults={saveresults} basis_function={basis_function} inscaler={inscaler} outscaler={outscaler}"
    runner_kwargs = f"cross_validate={cross_validate} parallelize={parallelize} job_type={job_type} n_jobs={n_jobs} temp_dir={temp_dir} log_dir={log_dir} environment={python_env}"
    blr_kwargs = f"optimizer={optimizer} n_iter={n_iter} heteroskedastic={heteroskedastic} fixed_effect={fixed_effect} warp={warp} warp_reparam={warp_reparam}"
    full_command = (
        f"{command} {args} {kwargs} {runner_kwargs} {normative_model_kwargs} {blr_kwargs}"
    )

.. code:: ipython3

    print(full_command)


.. code:: text

    normative -a blr -f fit_predict -c /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/covariates.csv -r /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/responses.csv -t /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_responses.csv -e /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_covariates.csv -k 5 be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/batch_effects.csv t_be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_batch_effects.csv cross_validate=True parallelize=False job_type=local n_jobs=2 temp_dir=resources/cli_example/temp log_dir=resources/cli_example/log environment=/opt/hostedtoolcache/Python/3.13.16/x64 save_dir=resources/cli_example/blr_cli/save_dir savemodel=True saveresults=True basis_function=linear inscaler=standardize outscaler=standardize optimizer=l-bfgs-b n_iter=200 heteroskedastic=True fixed_effect=True warp=WarpSinhArcsinh warp_reparam=True


Running command
~~~~~~~~~~~~~~~

.. code:: ipython3

    !{full_command}


.. code:: text

    Process: 3155 - 2026-10-27 14:43:17 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3155 - 2026-10-27 14:43:17 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3155 - 2026-10-27 14:43:17 - Task ID created: fit_predict_fit_data__2026-10-27_14-43-17_964.687012
    Process: 3155 - 2026-10-27 14:43:17 - Temporary directory created:
    	/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/temp/fit_predict_fit_data__2026-10-27_14-43-17_964.687012
    Process: 3155 - 2026-10-27 14:43:17 - Log directory created:
    	/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/log/fit_predict_fit_data__2026-10-27_14-43-17_964.687012
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:342: UserWarning: Process: 3155 - 2026-10-27 14:43:17 - Predict data not used in k-fold cross-validation
      warnings.warn(message, category)
    /opt/hostedtoolcache/Python/3.13.16/x64/lib/python3.13/site-packages/sklearn/model_selection/_split.py:812: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
      warnings.warn(
    Process: 3155 - 2026-10-27 14:43:17 - Fitting models on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:17 - Fitting model for response_var_0.
    Process: 3155 - 2026-10-27 14:43:18 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.523723946631791e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:342: UserWarning: Process: 3155 - 2026-10-27 14:43:18 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.375496563279125e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.9980721138137064e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.746691663731469e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.104659065295494e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 3155 - 2026-10-27 14:43:18 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_0.
    Process: 3155 - 2026-10-27 14:43:18 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:18 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:18 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:18 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:18 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:18 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:18 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:18 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:18 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:18 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:18 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:18 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:18 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:18 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:19 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:19 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:19 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:19 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:19 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:19 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:19 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:20 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:20 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:20 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:20 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:20 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:20 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:20 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:20 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:20 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:20 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:20 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:20 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:20 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:20 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:21 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:21 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:21 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:21 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:21 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:21 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:21 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:21 - Fitting models on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:21 - Fitting model for response_var_0.
    Process: 3155 - 2026-10-27 14:43:22 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.2455126268504203e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:342: UserWarning: Process: 3155 - 2026-10-27 14:43:22 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.910057133212752e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.8415638861183265e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.185259614266013e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.691929377086917e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/hostedtoolcache/Python/3.13.16/x64/lib/python3.13/site-packages/scipy/optimize/_numdiff.py:711: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    Process: 3155 - 2026-10-27 14:43:22 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_1.
    Process: 3155 - 2026-10-27 14:43:22 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:22 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:22 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:22 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:22 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:22 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:22 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:22 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:22 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:22 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:22 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:22 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:22 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:22 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:23 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:23 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:23 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:23 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:23 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:23 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:23 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:23 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:23 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:23 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:23 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:23 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:23 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:23 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:24 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:24 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:24 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:24 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:24 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:24 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:24 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:24 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:24 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:24 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:24 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:24 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:24 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:24 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:25 - Fitting models on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:25 - Fitting model for response_var_0.
    Process: 3155 - 2026-10-27 14:43:25 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.701875421761123e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:342: UserWarning: Process: 3155 - 2026-10-27 14:43:25 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.6521548448675865e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.3656485817707845e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.502124577440806e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 3155 - 2026-10-27 14:43:25 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_2.
    Process: 3155 - 2026-10-27 14:43:25 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:25 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:25 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:25 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:25 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:25 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:25 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:25 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:25 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:25 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:25 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:26 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:26 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:26 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:26 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:26 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:26 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:26 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:26 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:26 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:26 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:27 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:27 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:27 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:27 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:27 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:27 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:27 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:27 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:27 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:27 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:27 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:27 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:27 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:27 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:28 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:28 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:28 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:28 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:28 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:28 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:28 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:29 - Fitting models on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:29 - Fitting model for response_var_0.
    Process: 3155 - 2026-10-27 14:43:29 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.345402012863007e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:342: UserWarning: Process: 3155 - 2026-10-27 14:43:29 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.416128457233547e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.01776121710389e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.4551614365857112e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/hostedtoolcache/Python/3.13.16/x64/lib/python3.13/site-packages/scipy/optimize/_numdiff.py:711: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    Process: 3155 - 2026-10-27 14:43:29 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_3.
    Process: 3155 - 2026-10-27 14:43:29 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:29 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:29 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:29 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:29 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:29 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:29 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:29 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:29 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:29 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:29 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:29 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:29 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:29 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:30 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:30 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:30 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:30 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:30 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:30 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:30 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:31 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:31 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:31 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:31 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:31 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:31 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:31 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:31 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:31 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:31 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:31 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:31 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:31 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:31 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:32 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:32 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:32 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:32 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:32 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:32 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:32 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:32 - Fitting models on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:32 - Fitting model for response_var_0.
    Process: 3155 - 2026-10-27 14:43:33 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204862148507314e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:342: UserWarning: Process: 3155 - 2026-10-27 14:43:33 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.1618189276041684e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204835921010135e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204861445939724e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 3155 - 2026-10-27 14:43:33 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_4.
    Process: 3155 - 2026-10-27 14:43:33 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:33 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:33 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:33 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:33 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:33 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:33 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:33 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:33 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:33 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:33 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:33 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:33 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:33 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:34 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:34 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:34 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:34 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:34 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:34 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:34 - Harmonizing data for response_var_0.
    Process: 3155 - 2026-10-27 14:43:35 - Making predictions on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Computing z-scores for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Computing z-scores for response_var_1.
    Process: 3155 - 2026-10-27 14:43:35 - Computing z-scores for response_var_0.
    Process: 3155 - 2026-10-27 14:43:35 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:35 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:35 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Computing log-probabilities for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Computing log-probabilities for response_var_1.
    Process: 3155 - 2026-10-27 14:43:35 - Computing log-probabilities for response_var_0.
    Process: 3155 - 2026-10-27 14:43:35 - Computing yhat for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Computing yhat for response_var_1.
    Process: 3155 - 2026-10-27 14:43:35 - Computing yhat for response_var_0.
    Process: 3155 - 2026-10-27 14:43:35 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3155 - 2026-10-27 14:43:35 - Computing centiles for 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Computing centiles for response_var_1.
    Process: 3155 - 2026-10-27 14:43:35 - Computing centiles for response_var_0.
    Process: 3155 - 2026-10-27 14:43:35 - Harmonizing data on 2 response variables.
    Process: 3155 - 2026-10-27 14:43:35 - Harmonizing data for response_var_1.
    Process: 3155 - 2026-10-27 14:43:35 - Harmonizing data for response_var_0.


You can find the results in the
``resources/cli_example/blr_cli/save_dir`` folder.

.. code:: ipython3

    results_path = os.path.join(
        save_dir,
        "folds",
        "fold_1",
        "results",
        "statistics_fit_data_fold_1_predict.csv",
    )
    a = pd.read_csv(results_path, index_col=0)
    display(a)



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>response_var_0</th>
          <th>response_var_1</th>
        </tr>
        <tr>
          <th>statistic</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>EXPV</th>
          <td>4.679810e-01</td>
          <td>2.030427e-01</td>
        </tr>
        <tr>
          <th>Kurtosis</th>
          <td>-1.876473e-01</td>
          <td>-7.910999e-02</td>
        </tr>
        <tr>
          <th>MACE</th>
          <td>1.646634e-01</td>
          <td>1.748506e-01</td>
        </tr>
        <tr>
          <th>MAPE</th>
          <td>2.360966e-02</td>
          <td>3.066030e-01</td>
        </tr>
        <tr>
          <th>MLL</th>
          <td>1.128169e+00</td>
          <td>8.333058e-01</td>
        </tr>
        <tr>
          <th>MSLL</th>
          <td>-3.154758e-01</td>
          <td>-8.654640e-01</td>
        </tr>
        <tr>
          <th>R2</th>
          <td>4.679601e-01</td>
          <td>2.011754e-01</td>
        </tr>
        <tr>
          <th>RMSE</th>
          <td>7.322627e-02</td>
          <td>9.000878e+02</td>
        </tr>
        <tr>
          <th>Rho</th>
          <td>6.433635e-01</td>
          <td>4.877740e-01</td>
        </tr>
        <tr>
          <th>Rho_p</th>
          <td>1.359551e-21</td>
          <td>9.994549e-12</td>
        </tr>
        <tr>
          <th>SMSE</th>
          <td>5.320399e-01</td>
          <td>7.988246e-01</td>
        </tr>
        <tr>
          <th>ShapiroW</th>
          <td>9.841289e-01</td>
          <td>9.565320e-01</td>
        </tr>
        <tr>
          <th>Skewness</th>
          <td>-2.071004e-01</td>
          <td>6.880513e-01</td>
        </tr>
      </tbody>
    </table>
    </div>


HBR example
-----------

.. code:: ipython3

    alg = "hbr"
    func = "fit_predict"
    
    # normative model configuration
    save_dir = os.path.join(root_dir, "hbr", "save_dir")
    savemodel = True
    saveresults = True
    basis_function = "bspline"
    inscaler = "standardize"
    outscaler = "standardize"
    
    
    # Regression model configuration
    draws = 1000
    tune = 500
    chains = 4
    nuts_sampler = "nutpie"
    # The progress bar cannot be shown when the CLI runs from a notebook cell
    progressbar = "False"
    
    likelihood = "Normal"
    linear_mu = "True"
    random_intercept_mu = "True"
    random_slope_mu = "False"
    linear_sigma = "True"
    random_intercept_sigma = "False"
    random_slope_sigma = "False"

Constructing command
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    command = "normative"
    args = f"-a {alg} -f {func} -c {cov} -r {resp} -t {t_resp} -e {t_cov}"
    kwargs = f"be={be} t_be={t_be}"
    normative_model_kwargs = f"save_dir={save_dir} savemodel={savemodel} saveresults={saveresults} basis_function={basis_function} inscaler={inscaler} outscaler={outscaler}"
    hbr_kwargs = f"draws={draws} tune={tune} chains={chains} nuts_sampler={nuts_sampler} progressbar={progressbar} likelihood={likelihood} linear_mu={linear_mu} random_intercept_mu={random_intercept_mu} random_slope_mu={random_slope_mu} linear_sigma={linear_sigma} random_intercept_sigma={random_intercept_sigma} random_slope_sigma={random_slope_sigma}"
    full_command = f"{command} {args} {kwargs} {normative_model_kwargs} {hbr_kwargs}"
    print(full_command)


.. code:: text

    normative -a hbr -f fit_predict -c /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/covariates.csv -r /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/responses.csv -t /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_responses.csv -e /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_covariates.csv be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/batch_effects.csv t_be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_batch_effects.csv save_dir=resources/cli_example/hbr/save_dir savemodel=True saveresults=True basis_function=bspline inscaler=standardize outscaler=standardize draws=1000 tune=500 chains=4 nuts_sampler=nutpie progressbar=False likelihood=Normal linear_mu=True random_intercept_mu=True random_slope_mu=False linear_sigma=True random_intercept_sigma=False random_slope_sigma=False


Running command
~~~~~~~~~~~~~~~

.. code:: ipython3

    !{full_command}


.. code:: text

    Process: 3167 - 2026-10-27 14:43:39 - No log directory specified. Using default log directory: /home/runner/.pcntoolkit/logs
    Process: 3167 - 2026-10-27 14:43:39 - No temporary directory specified. Using default temporary directory: /home/runner/.pcntoolkit/temp
    Process: 3167 - 2026-10-27 14:43:39 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3167 - 2026-10-27 14:43:39 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3167 - 2026-10-27 14:43:39 - Task ID created: fit_predict_fit_data__2026-10-27_14-43-39_903.841797
    Process: 3167 - 2026-10-27 14:43:39 - Temporary directory created:
    	/home/runner/.pcntoolkit/temp/fit_predict_fit_data__2026-10-27_14-43-39_903.841797
    Process: 3167 - 2026-10-27 14:43:39 - Log directory created:
    	/home/runner/.pcntoolkit/logs/fit_predict_fit_data__2026-10-27_14-43-39_903.841797
    Process: 3167 - 2026-10-27 14:43:39 - Fitting models on 2 response variables.
    Process: 3167 - 2026-10-27 14:43:39 - Fitting model for response_var_0.
    NUTS[nutpie]: [slope_mu, mu_intercept_mu, batch_effect_0_sigma_intercept_mu, normalized_batch_effect_0_offset_intercept_mu, batch_effect_1_sigma_intercept_mu, normalized_batch_effect_1_offset_intercept_mu, slope_sigma, intercept_sigma]
    Process: 3167 - 2026-10-27 14:43:58 - Fitting model for response_var_1.
    NUTS[nutpie]: [slope_mu, mu_intercept_mu, batch_effect_0_sigma_intercept_mu, normalized_batch_effect_0_offset_intercept_mu, batch_effect_1_sigma_intercept_mu, normalized_batch_effect_1_offset_intercept_mu, slope_sigma, intercept_sigma]
    Process: 3167 - 2026-10-27 14:44:07 - Saving model to:
    	resources/cli_example/hbr/save_dir.
    Process: 3167 - 2026-10-27 14:44:07 - Making predictions on 2 response variables.
    Process: 3167 - 2026-10-27 14:44:07 - Computing z-scores for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:07 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:08 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:09 - Computing centiles for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:09 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:11 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:12 - Computing log-probabilities for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:12 - Computing log-probabilities for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:12 - Computing log-probabilities for response_var_0.
    Process: 3167 - 2026-10-27 14:44:14 - Computing log-probabilities for response_var_1.
    Process: 3167 - 2026-10-27 14:44:14 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:16 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3167 - 2026-10-27 14:44:16 - Computing centiles for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:16 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:17 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:19 - Harmonizing data on 2 response variables.
    Process: 3167 - 2026-10-27 14:44:19 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:20 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:21 - Making predictions on 2 response variables.
    Process: 3167 - 2026-10-27 14:44:21 - Computing z-scores for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:21 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:21 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:22 - Computing centiles for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:22 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:23 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:25 - Computing log-probabilities for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:25 - Computing log-probabilities for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:25 - Computing log-probabilities for response_var_0.
    Process: 3167 - 2026-10-27 14:44:25 - Computing log-probabilities for response_var_1.
    Process: 3167 - 2026-10-27 14:44:25 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:27 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3167 - 2026-10-27 14:44:27 - Computing centiles for 2 response variables.
    Process: 3167 - 2026-10-27 14:44:27 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:28 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:30 - Harmonizing data on 2 response variables.
    Process: 3167 - 2026-10-27 14:44:30 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    Process: 3167 - 2026-10-27 14:44:30 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []

