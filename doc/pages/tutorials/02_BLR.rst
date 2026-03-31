Bayesian Linear Regression (BLR)
================================

Welcome to this tutorial notebook that will go through the fitting and
evaluation of Normative models with Bayesian Linear Regression (BLR).

Let’s jump right in.

Imports
~~~~~~~

.. code:: ipython3

    import warnings
    import logging
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from pcntoolkit import (
        BLR,
        BsplineBasisFunction,
        LinearBasisFunction,
        NormativeModel,
        NormData,
        load_fcon1000,
        plot_centiles_advanced,
        plot_qq,
        plot_ridge,
    )
    
    import pcntoolkit.util.output
    import seaborn as sns
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn'
    pcntoolkit.util.output.Output.set_show_messages(True)

Load data
---------

First we download a small example dataset from github.

.. code:: ipython3

    # Download an example dataset
    norm_data: NormData = load_fcon1000()
    
    # Select only a few features
    features_to_model = [
        "WM-hypointensities",
        "Right-Lateral-Ventricle",
        "Right-Amygdala",
        "CortexVol",
    ]
    norm_data = norm_data.sel({"response_vars": features_to_model})
    
    # Split into train and test sets
    train, test = norm_data.train_test_split()


.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:42 - Removed 0 NANs
    Process: 75222 - 2025-11-20 13:13:42 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 217 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    

.. code:: ipython3

    # Visualize the data
    feature_to_plot = features_to_model[1]
    df = train.to_dataframe()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.countplot(data=df, y=("batch_effects", "site"), hue=("batch_effects", "sex"), ax=ax[0], orient="h")
    ax[0].legend(title="Sex")
    ax[0].set_title("Count of sites")
    ax[0].set_xlabel("Site")
    ax[0].set_ylabel("Count")
    
    
    sns.scatterplot(
        data=df,
        x=("X", "age"),
        y=("Y", feature_to_plot),
        hue=("batch_effects", "site"),
        style=("batch_effects", "sex"),
        ax=ax[1],
    )
    ax[1].legend([], [])
    ax[1].set_title(f"Scatter plot of age vs {feature_to_plot}")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel(feature_to_plot)
    
    plt.show()



.. image:: 02_BLR_files/02_BLR_6_0.png


Creating a Normative model
--------------------------

A normative model has a regression model for each response variable. We
provide a template regression model which is copied for each response
variable.

A template regression model can be anything that extends the
``RegressionModel``. We provide a number of built-in regression models,
but you can also create your own.

Here we use the ``BLR`` class, which implements a Bayesian Linear
Regression model.

The ``BLR`` class has a number of parameters that can be set,
determining whether and how batch effects are modeled, which basis
expansion to use, and more.

.. code:: ipython3

    template_blr = BLR(
        name="template",
        basis_function_mean=BsplineBasisFunction(
            degree=3, nknots=5
        ),  # We use a B-spline basis expansion for the mean, so the predicted mean is a smooth function of the covariates
        fixed_effect=True,  # By setting fixed_effect=True, we \model offsets in the mean for each individual batch effect,
        fixed_effect_slope=True,  # We also model a fixed effect in the slope of the mean for each individual batch effect
        fixed_effect_var_slope=True,
        heteroskedastic=True,  # We want the variance to be a function of the covariates too
        warp_name="warpsinharcsinh",  # We configure a sinh-arcsinh warp, so we can model flexible non-gaussian distributions
    )

After specifying the regression model, we can configure a normative
model.

A normative model has a number of configuration options: -
``savemodel``: Whether to save the model after fitting. -
``evaluate_model``: Whether to evaluate the model after fitting. -
``saveresults``: Whether to save the results after evaluation. -
``saveplots``: Whether to save the plots after fitting. - ``save_dir``:
The directory to save the model, results, and plots. - ``inscaler``: The
scaler to use for the input data. - ``outscaler``: The scaler to use for
the output data.

.. code:: ipython3

    model = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_blr,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/blr/save_dir",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

Fit the model
-------------

With all that configured, we can fit the model.

The ``fit_predict`` function will fit the model, evaluate it, and save
the results and plots (if so configured).

After that, it will compute Z-scores and centiles for the test set.

All results can be found in the save directory.

.. code:: ipython3

    model.fit_predict(train, test)


.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:43 - Fitting models on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:43 - Fitting model for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:44 - Fitting model for Right-Lateral-Ventricle.
    

.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.32781e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.99801e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=3.59377e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.57933e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=4.97275e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.13681e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.24046e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.33402e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.53589e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.55948e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=3.91748e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.33937e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.99954e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.32878e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.5002e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.26581e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=4.69477e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=6.90536e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.34843e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=4.95844e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=4.68778e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.45059e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.32535e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.48238e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=5.44743e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=6.36754e-17): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    

.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:46 - Fitting model for Right-Amygdala.
    

.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.16328e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 75222 - 2025-11-20 13:13:46 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.95673e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.3908e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=4.14393e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=4.37e-42): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.16329e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=1.71025e-35): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.16368e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=2.97928e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/scipy/_lib/_util.py:1226: LinAlgWarning: Ill-conditioned matrix (rcond=1.95743e-41): result may not be accurate.
      return f(*arrays, *other_args, **kwargs)
    

.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:46 - Fitting model for CortexVol.
    Process: 75222 - 2025-11-20 13:13:47 - Making predictions on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:47 - Computing z-scores for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:47 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:47 - Computing z-scores for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:47 - Computing z-scores for CortexVol.
    Process: 75222 - 2025-11-20 13:13:47 - Computing z-scores for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:47 - Computing centiles for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:47 - Computing centiles for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:47 - Computing centiles for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:47 - Computing centiles for CortexVol.
    Process: 75222 - 2025-11-20 13:13:47 - Computing centiles for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:47 - Computing log-probabilities for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:47 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:47 - Computing log-probabilities for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:47 - Computing log-probabilities for CortexVol.
    Process: 75222 - 2025-11-20 13:13:47 - Computing log-probabilities for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:47 - Computing yhat for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:47 - Computing yhat for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for CortexVol.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:48 - Saving model to:
    	resources/blr/save_dir.
    Process: 75222 - 2025-11-20 13:13:48 - Making predictions on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:48 - Computing z-scores for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:48 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:48 - Computing z-scores for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:48 - Computing z-scores for CortexVol.
    Process: 75222 - 2025-11-20 13:13:48 - Computing z-scores for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for CortexVol.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:48 - Computing log-probabilities for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:48 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:48 - Computing log-probabilities for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:48 - Computing log-probabilities for CortexVol.
    Process: 75222 - 2025-11-20 13:13:48 - Computing log-probabilities for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for CortexVol.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:48 - Saving model to:
    	resources/blr/save_dir.
    



.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base rgba(0, 0, 0, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, white)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
      );
    }
    
    html[theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    body.vscode-dark {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base, rgba(255, 255, 255, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, #111111)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
      );
    }
    
    .xr-wrap {
      display: block !important;
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: inline-block;
      opacity: 0;
      height: 0;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
      border: 2px solid transparent !important;
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:focus + label {
      border: 2px solid var(--xr-font-color0) !important;
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: "►";
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: "▼";
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: "(";
    }
    
    .xr-dim-list:after {
      content: ")";
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ",";
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      border-color: var(--xr-background-color-row-odd);
      margin-bottom: 0;
      padding-top: 2px;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
      border-color: var(--xr-background-color-row-even);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-index-preview {
      grid-column: 2 / 5;
      color: var(--xr-font-color2);
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      display: none;
      border-top: 2px dotted var(--xr-background-color);
      padding-bottom: 20px !important;
      padding-top: 10px !important;
    }
    
    .xr-var-attrs-in + label,
    .xr-var-data-in + label,
    .xr-index-data-in + label {
      padding: 0 1px;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data,
    .xr-index-data-in:checked ~ .xr-index-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-data > pre,
    .xr-index-data > pre,
    .xr-var-data > table > tbody > tr {
      background-color: transparent !important;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-index-name div,
    .xr-index-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2,
    .xr-no-icon {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    
    .xr-var-attrs-in:checked + label > .xr-icon-file-text2,
    .xr-var-data-in:checked + label > .xr-icon-database,
    .xr-index-data-in:checked + label > .xr-icon-database {
      color: var(--xr-font-color0);
      filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
      stroke-width: 0.8px;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 98kB
    Dimensions:            (observations: 216, response_vars: 4, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 11)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 368B &#x27;WM-hypointensities&#x27; ... &#x27;Co...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 352B &#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
    Data variables:
        subject_ids        (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 7kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 7kB 0.991 ... -1...
        centiles           (centile, observations, response_vars) float64 35kB 57...
        logp               (observations, response_vars) float64 7kB -1.869 ... -...
        Yhat               (observations, response_vars) float64 7kB 1.91e+03 ......
        statistics         (response_vars, statistic) float64 352B -2.475 ... 0.995
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.2...
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-ee5d490d-3177-4c4f-a557-b451cfb403c7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-ee5d490d-3177-4c4f-a557-b451cfb403c7' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 4</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 11</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-feae479c-1e0b-4c1f-a23c-7f894206c264' class='xr-section-summary-in' type='checkbox'  checked><label for='section-feae479c-1e0b-4c1f-a23c-7f894206c264' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-c7602038-e446-4df4-8172-564b86b9181c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c7602038-e446-4df4-8172-564b86b9181c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-486a45db-1810-4445-a659-5bf7536f772c' class='xr-var-data-in' type='checkbox'><label for='data-486a45db-1810-4445-a659-5bf7536f772c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Cortex...</div><input id='attrs-79fed74f-1fa4-491a-9612-a501d9c13278' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-79fed74f-1fa4-491a-9612-a501d9c13278' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5165888f-0fc6-4623-8d04-dc58df6cbecc' class='xr-var-data-in' type='checkbox'><label for='data-5165888f-0fc6-4623-8d04-dc58df6cbecc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-4d17eb3e-4c12-4951-9df5-6e5ad020386f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4d17eb3e-4c12-4951-9df5-6e5ad020386f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-14230cac-332c-4f31-ac27-1d1499b6416f' class='xr-var-data-in' type='checkbox'><label for='data-14230cac-332c-4f31-ac27-1d1499b6416f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-ed52fa10-0c2c-47b8-8d9d-228848787f9c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ed52fa10-0c2c-47b8-8d9d-228848787f9c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1ab364e5-ceb2-4827-849f-0089bdec8159' class='xr-var-data-in' type='checkbox'><label for='data-1ab364e5-ceb2-4827-849f-0089bdec8159' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-f85e1af8-ec72-4738-a534-ec4dd0a542eb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f85e1af8-ec72-4738-a534-ec4dd0a542eb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fe5498db-0789-4266-8fb6-e85610a89ef0' class='xr-var-data-in' type='checkbox'><label for='data-fe5498db-0789-4266-8fb6-e85610a89ef0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-4629cf4a-ab94-471d-b5c2-ee4069f277bc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4629cf4a-ab94-471d-b5c2-ee4069f277bc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eda31c7a-cfa0-4393-8eeb-ea9db6a12e9c' class='xr-var-data-in' type='checkbox'><label for='data-eda31c7a-cfa0-4393-8eeb-ea9db6a12e9c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7844e918-0774-4753-8655-7a0139351448' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7844e918-0774-4753-8655-7a0139351448' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-27bbd14a-0dcc-40b4-894e-525ab59f15a4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-27bbd14a-0dcc-40b4-894e-525ab59f15a4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a03a8193-dace-4aaa-a440-27a5b4ca10f5' class='xr-var-data-in' type='checkbox'><label for='data-a03a8193-dace-4aaa-a440-27a5b4ca10f5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
           &#x27;ICBM_sub47658&#x27;, &#x27;AnnArbor_b_sub45569&#x27;, &#x27;Beijing_Zang_sub18960&#x27;,
           &#x27;Leiden_2200_sub18456&#x27;, &#x27;Berlin_Margulies_sub27711&#x27;,
           &#x27;Beijing_Zang_sub87776&#x27;, &#x27;Milwaukee_b_sub63196&#x27;,
           &#x27;Beijing_Zang_sub07144&#x27;, &#x27;Atlanta_sub76280&#x27;,
           &#x27;Beijing_Zang_sub40037&#x27;, &#x27;Cambridge_Buckner_sub17737&#x27;,
           &#x27;ICBM_sub89049&#x27;, &#x27;ICBM_sub55656&#x27;, &#x27;Oulu_sub45566&#x27;,
           &#x27;Beijing_Zang_sub89088&#x27;, &#x27;Atlanta_sub16563&#x27;,
           &#x27;Cambridge_Buckner_sub51172&#x27;, &#x27;Oulu_sub98739&#x27;,
           &#x27;Queensland_sub49845&#x27;, &#x27;Cambridge_Buckner_sub84256&#x27;,
           &#x27;Cleveland_sub80263&#x27;, &#x27;ICBM_sub16607&#x27;, &#x27;Newark_sub46570&#x27;,
           &#x27;NewYork_a_sub88286&#x27;, &#x27;Cambridge_Buckner_sub02591&#x27;,
           &#x27;Oulu_sub66467&#x27;, &#x27;Beijing_Zang_sub74386&#x27;, &#x27;Newark_sub55760&#x27;,
           &#x27;ICBM_sub30623&#x27;, &#x27;Oulu_sub68752&#x27;, &#x27;Leiden_2180_sub19281&#x27;,
           &#x27;Beijing_Zang_sub50972&#x27;, &#x27;Beijing_Zang_sub85030&#x27;,
           &#x27;Milwaukee_b_sub36386&#x27;, &#x27;Baltimore_sub31837&#x27;, &#x27;PaloAlto_sub84978&#x27;,
           &#x27;Oulu_sub01077&#x27;, &#x27;NewYork_a_ADHD_sub54828&#x27;, &#x27;PaloAlto_sub96705&#x27;,
           &#x27;Cambridge_Buckner_sub40635&#x27;, &#x27;ICBM_sub66794&#x27;,
           &#x27;Beijing_Zang_sub46541&#x27;, &#x27;Beijing_Zang_sub87089&#x27;,
           &#x27;Pittsburgh_sub97823&#x27;, &#x27;Beijing_Zang_sub98617&#x27;, &#x27;ICBM_sub92028&#x27;,
    ...
           &#x27;Leiden_2200_sub04484&#x27;, &#x27;Beijing_Zang_sub80163&#x27;, &#x27;ICBM_sub02382&#x27;,
           &#x27;Cambridge_Buckner_sub77435&#x27;, &#x27;NewYork_a_sub54887&#x27;,
           &#x27;Oulu_sub85532&#x27;, &#x27;Baltimore_sub73823&#x27;, &#x27;Beijing_Zang_sub29590&#x27;,
           &#x27;Oulu_sub99718&#x27;, &#x27;Beijing_Zang_sub08455&#x27;, &#x27;Beijing_Zang_sub85543&#x27;,
           &#x27;Cambridge_Buckner_sub45354&#x27;, &#x27;Beijing_Zang_sub07717&#x27;,
           &#x27;Baltimore_sub76160&#x27;, &#x27;Beijing_Zang_sub17093&#x27;,
           &#x27;AnnArbor_b_sub90127&#x27;, &#x27;SaintLouis_sub73002&#x27;,
           &#x27;Queensland_sub93238&#x27;, &#x27;Cleveland_sub34189&#x27;,
           &#x27;Cambridge_Buckner_sub89107&#x27;, &#x27;Atlanta_sub75153&#x27;,
           &#x27;NewYork_a_ADHD_sub73035&#x27;, &#x27;Cambridge_Buckner_sub59434&#x27;,
           &#x27;Milwaukee_b_sub44912&#x27;, &#x27;Cleveland_sub46739&#x27;, &#x27;Oulu_sub20495&#x27;,
           &#x27;SaintLouis_sub28304&#x27;, &#x27;Cambridge_Buckner_sub35430&#x27;,
           &#x27;Oulu_sub86362&#x27;, &#x27;Newark_sub58526&#x27;, &#x27;Leiden_2180_sub12255&#x27;,
           &#x27;ICBM_sub48210&#x27;, &#x27;Cambridge_Buckner_sub77989&#x27;,
           &#x27;Berlin_Margulies_sub75506&#x27;, &#x27;NewYork_a_sub29216&#x27;,
           &#x27;Beijing_Zang_sub05267&#x27;, &#x27;AnnArbor_b_sub18546&#x27;, &#x27;Oulu_sub75620&#x27;,
           &#x27;AnnArbor_b_sub30250&#x27;, &#x27;Berlin_Margulies_sub86111&#x27;,
           &#x27;Beijing_Zang_sub89592&#x27;, &#x27;Beijing_Zang_sub68012&#x27;,
           &#x27;NewYork_a_sub50559&#x27;, &#x27;Munchen_sub66933&#x27;,
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 5.035e+05</div><input id='attrs-2dea6f8a-baa3-456f-9868-a0f25d42d3c9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2dea6f8a-baa3-456f-9868-a0f25d42d3c9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-39c7b7d9-eba0-473b-b232-e28c83d0f916' class='xr-var-data-in' type='checkbox'><label for='data-39c7b7d9-eba0-473b-b232-e28c83d0f916' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.72140000e+03, 1.28916000e+04, 1.43940000e+03, 4.57858328e+05],
           [1.14310000e+03, 9.91910000e+03, 1.64970000e+03, 5.26780362e+05],
           [9.55800000e+02, 7.47730000e+03, 1.83850000e+03, 4.95744471e+05],
           [1.47390000e+03, 1.43021000e+04, 1.86770000e+03, 5.85303839e+05],
           [7.57800000e+02, 4.11930000e+03, 1.32500000e+03, 3.33111552e+05],
           [8.71100000e+02, 5.03090000e+03, 1.90730000e+03, 5.10794940e+05],
           [1.20730000e+03, 1.78664000e+04, 2.02220000e+03, 5.50533325e+05],
           [5.95000000e+02, 5.00790000e+03, 2.01070000e+03, 4.67673977e+05],
           [6.82400000e+02, 7.28660000e+03, 1.45630000e+03, 4.60129533e+05],
           [4.45100000e+02, 5.74290000e+03, 1.47450000e+03, 4.44494817e+05],
           [1.62000000e+03, 3.71370000e+03, 2.00110000e+03, 5.59424624e+05],
           [6.02800000e+02, 5.30120000e+03, 1.36100000e+03, 4.21551234e+05],
           [1.43250000e+03, 4.42970000e+03, 1.65080000e+03, 5.19842763e+05],
           [1.90820000e+03, 3.57810000e+03, 1.88370000e+03, 5.06679262e+05],
           [1.83400000e+03, 3.27190000e+03, 2.05120000e+03, 5.35569987e+05],
           [4.59600000e+02, 3.98580000e+03, 1.45470000e+03, 4.67607555e+05],
           [1.21000000e+03, 8.72130000e+03, 1.71430000e+03, 5.30904612e+05],
           [8.45900000e+02, 6.59310000e+03, 1.61900000e+03, 5.09371867e+05],
           [9.95200000e+02, 7.04020000e+03, 1.99490000e+03, 4.60068379e+05],
           [1.73470000e+03, 4.01480000e+03, 1.51620000e+03, 4.87269373e+05],
    ...
           [7.85800000e+02, 5.70900000e+03, 1.47480000e+03, 4.53982166e+05],
           [2.24010000e+03, 4.36660000e+03, 2.04210000e+03, 5.58453123e+05],
           [7.58100000e+02, 6.52980000e+03, 1.56730000e+03, 4.73575183e+05],
           [1.44050000e+03, 6.70530000e+03, 1.20540000e+03, 3.82788491e+05],
           [8.18600000e+02, 9.38330000e+03, 1.96740000e+03, 5.02713911e+05],
           [3.76990000e+03, 1.58644000e+04, 1.79170000e+03, 5.12490348e+05],
           [8.80200000e+02, 4.37020000e+03, 1.75520000e+03, 4.37300069e+05],
           [8.23900000e+02, 6.37900000e+03, 1.57650000e+03, 5.67331908e+05],
           [2.11390000e+03, 1.07225000e+04, 1.84380000e+03, 5.12273764e+05],
           [7.41900000e+02, 8.80170000e+03, 1.60640000e+03, 4.91973562e+05],
           [1.33390000e+03, 6.98000000e+03, 1.74850000e+03, 4.78907154e+05],
           [7.07300000e+02, 5.68070000e+03, 1.53450000e+03, 4.74077083e+05],
           [1.13410000e+03, 5.59220000e+03, 1.62620000e+03, 4.54163909e+05],
           [4.38600000e+02, 6.33000000e+03, 1.59670000e+03, 4.68067037e+05],
           [9.66300000e+02, 9.21550000e+03, 1.78250000e+03, 5.09199708e+05],
           [4.24300000e+02, 4.51110000e+03, 1.70200000e+03, 5.26635258e+05],
           [6.04700000e+02, 7.59080000e+03, 1.69930000e+03, 5.20499663e+05],
           [2.34320000e+03, 1.71923000e+04, 1.79380000e+03, 4.86680791e+05],
           [2.72170000e+03, 6.08600000e+03, 2.32470000e+03, 6.10402006e+05],
           [7.03500000e+02, 1.07003000e+04, 1.67620000e+03, 5.03535771e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-acb3c7b3-aaa2-4606-a30c-beabadfe7c50' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-acb3c7b3-aaa2-4606-a30c-beabadfe7c50' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c7317883-0b23-4914-9bfe-97cd4431ae4d' class='xr-var-data-in' type='checkbox'><label for='data-c7317883-0b23-4914-9bfe-97cd4431ae4d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
           [23.27],
           [22.  ],
           [42.  ],
           [63.  ],
           [23.  ],
           [21.  ],
           [26.  ],
           [21.  ],
           [49.  ],
           [20.  ],
           [23.  ],
           [20.  ],
           [26.  ],
           [35.  ],
           [21.  ],
           [22.  ],
           [19.  ],
           [34.  ],
           [18.  ],
    ...
           [21.  ],
           [20.  ],
           [22.  ],
           [25.  ],
           [25.  ],
           [73.  ],
           [22.  ],
           [28.  ],
           [29.06],
           [19.  ],
           [20.  ],
           [22.  ],
           [19.  ],
           [24.  ],
           [21.  ],
           [24.  ],
           [22.79],
           [72.  ],
           [23.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-33fb4e80-f7f8-4fba-9f73-ff5e78ec8211' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-33fb4e80-f7f8-4fba-9f73-ff5e78ec8211' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4abdaaca-a99a-4acf-b8cd-65b7e4058051' class='xr-var-data-in' type='checkbox'><label for='data-4abdaaca-a99a-4acf-b8cd-65b7e4058051' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Milwaukee_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
    ...
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Newark&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2180&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;M&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.991 0.8183 ... -1.358 -1.272</div><input id='attrs-b8bc485c-7f00-4792-b02a-440f51ac667b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b8bc485c-7f00-4792-b02a-440f51ac667b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-64786103-7b90-4092-aa2e-65a288252aee' class='xr-var-data-in' type='checkbox'><label for='data-64786103-7b90-4092-aa2e-65a288252aee' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.9910344 ,  0.81830048, -0.55018133,  0.13955562],
           [-0.01085574,  1.38811541, -1.28944887, -0.22201834],
           [ 0.17713812,  0.33628493,  0.61264106,  0.29792913],
           [-0.01677579,  1.60929134, -0.07031814,  1.26890505],
           [-1.15537626, -1.38013383, -0.26679411, -1.58011453],
           [-0.83705405, -0.6925975 ,  0.45511936, -0.51539529],
           [ 0.62253858,  2.23320074,  0.27847909,  0.31344571],
           [ 0.21661963, -0.6753277 ,  1.98622612, -0.63117465],
           [-1.02395187,  0.81704349, -0.95300482, -0.84426157],
           [-0.66455908, -0.40326532, -0.73652474,  0.11213451],
           [ 1.49555376, -1.21726499,  0.77389599,  0.71203431],
           [-0.66321267, -0.33552826, -1.12904778, -1.03938937],
           [ 1.28383024, -0.49977552,  0.05769295,  1.18720052],
           [ 1.90999261, -0.69351978,  1.29145737,  1.80165049],
           [ 0.98592445, -1.58890315,  0.62572662,  0.09924965],
           [-1.12495195, -0.79240985, -0.98508816, -1.08859152],
           [-0.59763476,  0.62704661, -0.65512582, -0.36102778],
           [-0.62407676,  0.56323037, -0.16281503,  0.76715475],
           [-0.55957317, -0.13658049,  0.85962834, -1.06197019],
           [ 2.13100646, -0.65057486, -1.05001494,  0.87785052],
    ...
           [ 0.10620058, -0.11162761, -0.90209223, -1.33267893],
           [ 3.04842362, -0.88711349,  0.89548308,  1.74547361],
           [-1.39484774,  0.16743464, -0.27712282, -0.68632353],
           [-0.46595139,  0.14880783, -1.2583806 , -0.56745362],
           [-0.87906482,  0.6308172 ,  0.64504465, -0.51910144],
           [ 0.73592004,  0.11971089,  0.32803703,  0.86277602],
           [-1.13954502, -0.45868536,  0.57624067, -0.48693253],
           [ 0.09872182, -0.50135757, -1.51486826,  0.62392987],
           [ 1.98539968,  1.08076884, -0.31257137, -0.35782521],
           [-0.92818113,  1.59376006, -0.23560175,  0.11906651],
           [-0.17402345,  0.32419477,  0.32459488, -0.33556416],
           [-1.50267681, -0.17571033, -0.45561085, -0.66991848],
           [-0.42778381,  0.14953608,  0.71983899,  0.19909464],
           [ 0.09656318, -0.20009939, -0.03069303, -0.66919995],
           [-0.56383703,  1.19393477, -0.34018551, -0.76691625],
           [-1.47972475, -0.50524055,  0.53224766,  1.53950791],
           [-1.27217678,  0.51002814, -0.96896359, -0.40012592],
           [-0.70954373,  0.38006276,  0.30542294, -0.03362917],
           [ 3.51471532, -0.12046593,  1.78738181,  2.73068326],
           [-1.23766073,  1.11498484, -1.35814472, -1.2716136 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>571.2 4.966e+03 ... 6.834e+05</div><input id='attrs-1a115261-8548-46de-8631-1c22999bbffc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1a115261-8548-46de-8631-1c22999bbffc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-be37b42d-14ca-48ee-a89c-8f6e64c2f631' class='xr-var-data-in' type='checkbox'><label for='data-be37b42d-14ca-48ee-a89c-8f6e64c2f631' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[5.71205994e+02, 4.96564655e+03, 1.06316403e+03, 3.39817113e+05],
            [4.13035081e+02, 2.75772941e+03, 1.58879166e+03, 4.75739503e+05],
            [1.88662575e+01, 2.25430203e+03, 1.39579259e+03, 4.20080014e+05],
            ...,
            [1.61029512e+03, 8.09315163e+03, 8.95839362e+02, 3.28867059e+05],
            [8.11465978e+02, 2.24573665e+03, 1.64014264e+03, 4.51301006e+05],
            [4.87567573e+02, 2.43188395e+03, 1.62722821e+03, 4.91292413e+05]],
    
           [[1.28549504e+03, 7.46789216e+03, 1.40492724e+03, 4.16057424e+05],
            [8.78625548e+02, 5.12157892e+03, 1.75109764e+03, 5.09359965e+05],
            [5.81509378e+02, 5.00577241e+03, 1.60414586e+03, 4.62472246e+05],
            ...,
            [2.37761128e+03, 1.11499357e+04, 1.45777998e+03, 4.43279505e+05],
            [1.12683865e+03, 4.81720018e+03, 1.77644124e+03, 4.80360983e+05],
            [9.63991646e+02, 5.35635311e+03, 1.79273273e+03, 5.24834515e+05]],
    
           [[1.75209940e+03, 9.43308712e+03, 1.57379132e+03, 4.51639529e+05],
            [1.14724224e+03, 6.57372795e+03, 1.87190499e+03, 5.36352593e+05],
            [8.83889821e+02, 6.65314858e+03, 1.72490051e+03, 4.85862620e+05],
            ...,
            [3.19798573e+03, 1.44942779e+04, 1.69495484e+03, 4.88734754e+05],
            [1.33087243e+03, 6.35496078e+03, 1.88133716e+03, 4.98844602e+05],
            [1.24359390e+03, 7.13708261e+03, 1.92604343e+03, 5.54585499e+05]],
    
           [[2.35080063e+03, 1.21452946e+04, 1.71430561e+03, 4.79508558e+05],
            [1.40141325e+03, 8.05825203e+03, 2.02563162e+03, 5.71923846e+05],
            [1.14745824e+03, 8.35762438e+03, 1.85098455e+03, 5.08550678e+05],
            ...,
            [4.44734100e+03, 1.98704866e+04, 1.93281869e+03, 5.33654753e+05],
            [1.53695703e+03, 7.89723294e+03, 2.01110742e+03, 5.18370399e+05],
            [1.51726721e+03, 9.09267425e+03, 2.10455268e+03, 5.94956459e+05]],
    
           [[3.78210177e+03, 1.91696618e+04, 1.93448631e+03, 5.17064669e+05],
            [1.79054689e+03, 1.07315286e+04, 2.36031918e+03, 6.50440908e+05],
            [1.50816251e+03, 1.16498333e+04, 2.09934481e+03, 5.48052958e+05],
            ...,
            [7.51942906e+03, 3.44512783e+04, 2.62653940e+03, 6.53779222e+05],
            [1.86198242e+03, 1.06574501e+04, 2.27537187e+03, 5.53054505e+05],
            [1.96040026e+03, 1.33254385e+04, 2.50306886e+03, 6.83443625e+05]]],
          shape=(5, 216, 4))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.869 -1.603 ... -1.467 -1.283</div><input id='attrs-644a912c-9e75-47d6-b28e-3710d4f0606e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-644a912c-9e75-47d6-b28e-3710d4f0606e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5ccd9371-7ef1-49a7-9ea9-891feb42c4e1' class='xr-var-data-in' type='checkbox'><label for='data-5ccd9371-7ef1-49a7-9ea9-891feb42c4e1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.86906501e+00, -1.60266085e+00, -1.16872443e+00,
            -7.47478268e-01],
           [-1.58469803e-01, -1.62580449e+00, -1.36632897e+00,
            -7.11132196e-01],
           [-2.19181521e-01, -5.45926189e-01, -9.02398690e-01,
            -5.15012893e-01],
           [-4.34454374e-01, -2.74765922e+00, -7.26514955e-01,
            -1.96605917e+00],
           [-1.67122763e+00, -1.68727093e+00, -4.79816067e-01,
            -2.21938578e+00],
           [-5.15802252e-01, -6.57001146e-01, -8.20935222e-01,
            -4.70193633e-01],
           [-3.38754242e-01, -4.24650827e+00, -1.04503171e+00,
            -8.48054908e-01],
           [-5.02399231e-01, -7.88898804e-01, -2.98795362e+00,
            -7.62202461e-01],
           [-6.99642072e-01, -6.71339830e-01, -1.23608486e+00,
            -7.61475166e-01],
           [-1.34119838e+00, -9.05503762e-01, -8.85981354e-01,
            -6.66178270e-01],
    ...
           [-1.43832667e-01, -5.66257958e-01, -6.66723409e-01,
            -5.93002770e-01],
           [-1.51256793e+00, -5.02001383e-01, -7.58211952e-01,
            -6.09029430e-01],
           [-1.92262486e-01, -5.14451109e-01, -9.89111305e-01,
            -8.14119623e-01],
           [-5.75832982e-01, -5.45905002e-01, -6.62036576e-01,
            -8.05311515e-01],
           [-2.16911706e-01, -1.27138629e+00, -5.49341186e-01,
            -5.49459895e-01],
           [-1.53837639e+00, -5.98702577e-01, -6.60941081e-01,
            -1.72892590e+00],
           [-1.21627533e+00, -4.87960579e-01, -9.86555788e-01,
            -7.17502315e-01],
           [-1.34493012e+00, -1.73880392e+00, -1.28542246e+00,
            -1.07871724e+00],
           [-6.76604624e+00, -3.82760420e-01, -2.89324088e+00,
            -4.88027805e+00],
           [-1.18852124e+00, -1.60949381e+00, -1.46708031e+00,
            -1.28328036e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.91e+03 1.043e+04 ... 5.667e+05</div><input id='attrs-a8b5f024-5e88-4c55-a920-16e680b2d9a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a8b5f024-5e88-4c55-a920-16e680b2d9a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0fb038fa-4c96-4f49-8239-defb8e31883a' class='xr-var-data-in' type='checkbox'><label for='data-0fb038fa-4c96-4f49-8239-defb8e31883a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.91041133e+03,  1.04341585e+04,  1.54646809e+03,
             4.43073070e+05],
           [ 1.13032022e+03,  6.64197321e+03,  1.90995371e+03,
             5.46278301e+05],
           [ 8.39178018e+02,  6.77321275e+03,  1.73348173e+03,
             4.85216353e+05],
           [ 1.51626904e+03,  8.19404627e+03,  1.92483788e+03,
             5.28245435e+05],
           [ 1.81972723e+03,  8.98371634e+03,  1.35008245e+03,
             3.99081670e+05],
           [ 1.16175434e+03,  6.67434652e+03,  1.84788893e+03,
             5.31647015e+05],
           [ 9.28966969e+02,  7.97159280e+03,  2.00838678e+03,
             5.44982930e+05],
           [ 3.71682343e+02,  6.93649825e+03,  1.60830885e+03,
             4.89690114e+05],
           [ 1.02365924e+03,  5.48565741e+03,  1.62548221e+03,
             4.83940391e+05],
           [ 8.76457633e+02,  7.84997789e+03,  1.58394865e+03,
             4.32454963e+05],
    ...
           [ 1.40881234e+03,  6.18307347e+03,  1.69142305e+03,
             4.91022683e+05],
           [ 1.33485770e+03,  6.13138087e+03,  1.60315150e+03,
             4.94641280e+05],
           [ 1.28497416e+03,  5.05437662e+03,  1.42063754e+03,
             4.34115945e+05],
           [ 2.65544556e+02,  7.05756249e+03,  1.58564808e+03,
             4.92284380e+05],
           [ 1.14296630e+03,  6.51001714e+03,  1.86556449e+03,
             5.36309774e+05],
           [ 1.02322696e+03,  5.64189952e+03,  1.59986772e+03,
             4.77011978e+05],
           [ 1.12489000e+03,  6.55760649e+03,  1.90531392e+03,
             5.47638612e+05],
           [ 3.70760320e+03,  1.70380342e+04,  1.72029246e+03,
             4.89972976e+05],
           [ 1.33301797e+03,  6.39661753e+03,  1.90963880e+03,
             5.00086720e+05],
           [ 1.23623232e+03,  7.42824756e+03,  1.97755494e+03,
             5.66741543e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.475 0.03796 ... 0.6244 0.995</div><input id='attrs-1e9dd963-8def-4323-852b-c3d69ea76aef' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1e9dd963-8def-4323-852b-c3d69ea76aef' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f7d8d7f1-26c2-41aa-8f20-256ede5e8f9a' class='xr-var-data-in' type='checkbox'><label for='data-f7d8d7f1-26c2-41aa-8f20-256ede5e8f9a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-2.47487797e+00,  3.79629630e-02,  3.90576314e-01,
            -6.97874850e+00,  8.45801650e-01, -2.48539139e+00,
             1.12991963e+03,  4.69792686e-01,  2.94834937e-13,
             3.48539139e+00,  9.37601263e-01],
           [ 1.85347407e-01,  4.90740741e-02,  4.20342721e-01,
            -8.40210129e+00,  1.28840444e+00,  1.84189959e-01,
             3.53256787e+03,  2.46607074e-01,  2.52188374e-04,
             8.15810041e-01,  9.37045362e-01],
           [ 2.85233438e-01,  1.85185185e-02,  9.41622245e-02,
            -5.63777308e+00,  1.24475296e+00,  2.84777724e-01,
             1.99538370e+02,  5.12632996e-01,  7.12929722e-16,
             7.15222276e-01,  9.88757003e-01],
           [ 3.87191352e-01,  1.48148148e-02,  6.51500829e-02,
            -1.10844441e+01,  1.13311599e+00,  3.75602928e-01,
             3.86822921e+04,  6.35331704e-01,  8.11366151e-26,
             6.24397072e-01,  9.95039182e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-35fd1535-eb52-40f4-87c5-db5fd7c0fabc' class='xr-section-summary-in' type='checkbox'  ><label for='section-35fd1535-eb52-40f4-87c5-db5fd7c0fabc' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-09d32fd0-e5de-4c45-9633-288a71a1a3b8' class='xr-index-data-in' type='checkbox'/><label for='index-09d32fd0-e5de-4c45-9633-288a71a1a3b8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,
           ...
             27,  959,   29,  346,  304,  264,  798,  751,  470, 1043],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=216))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-a9a29dcc-f4df-4a25-85e3-b8aa30d13792' class='xr-index-data-in' type='checkbox'/><label for='index-a9a29dcc-f4df-4a25-85e3-b8aa30d13792' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-3bcffd49-7393-41ef-822e-061dd38c4410' class='xr-index-data-in' type='checkbox'/><label for='index-3bcffd49-7393-41ef-822e-061dd38c4410' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-2dc16003-0f53-45e4-afc7-79e3c045fe6e' class='xr-index-data-in' type='checkbox'/><label for='index-2dc16003-0f53-45e4-afc7-79e3c045fe6e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-cc3b8a59-0c13-453b-abbf-eff53c056c2e' class='xr-index-data-in' type='checkbox'/><label for='index-cc3b8a59-0c13-453b-abbf-eff53c056c2e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-9d5a87d9-9a28-4ce9-9df0-6c7febede7e6' class='xr-index-data-in' type='checkbox'/><label for='index-9d5a87d9-9a28-4ce9-9df0-6c7febede7e6' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4d079e1f-6f06-4dd4-a7b7-0d01ed10e3c1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4d079e1f-6f06-4dd4-a7b7-0d01ed10e3c1' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;M&#x27;)], np.str_(&#x27;site&#x27;): [np.str_(&#x27;AnnArbor_a&#x27;), np.str_(&#x27;AnnArbor_b&#x27;), np.str_(&#x27;Atlanta&#x27;), np.str_(&#x27;Baltimore&#x27;), np.str_(&#x27;Bangor&#x27;), np.str_(&#x27;Beijing_Zang&#x27;), np.str_(&#x27;Berlin_Margulies&#x27;), np.str_(&#x27;Cambridge_Buckner&#x27;), np.str_(&#x27;Cleveland&#x27;), np.str_(&#x27;ICBM&#x27;), np.str_(&#x27;Leiden_2180&#x27;), np.str_(&#x27;Leiden_2200&#x27;), np.str_(&#x27;Milwaukee_b&#x27;), np.str_(&#x27;Munchen&#x27;), np.str_(&#x27;NewYork_a&#x27;), np.str_(&#x27;NewYork_a_ADHD&#x27;), np.str_(&#x27;Newark&#x27;), np.str_(&#x27;Oulu&#x27;), np.str_(&#x27;Oxford&#x27;), np.str_(&#x27;PaloAlto&#x27;), np.str_(&#x27;Pittsburgh&#x27;), np.str_(&#x27;Queensland&#x27;), np.str_(&#x27;SaintLouis&#x27;)]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x17e771ee0&gt;, {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): 589, np.str_(&#x27;M&#x27;): 489}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): 24, np.str_(&#x27;AnnArbor_b&#x27;): 32, np.str_(&#x27;Atlanta&#x27;): 28, np.str_(&#x27;Baltimore&#x27;): 23, np.str_(&#x27;Bangor&#x27;): 20, np.str_(&#x27;Beijing_Zang&#x27;): 198, np.str_(&#x27;Berlin_Margulies&#x27;): 26, np.str_(&#x27;Cambridge_Buckner&#x27;): 198, np.str_(&#x27;Cleveland&#x27;): 31, np.str_(&#x27;ICBM&#x27;): 85, np.str_(&#x27;Leiden_2180&#x27;): 12, np.str_(&#x27;Leiden_2200&#x27;): 19, np.str_(&#x27;Milwaukee_b&#x27;): 46, np.str_(&#x27;Munchen&#x27;): 15, np.str_(&#x27;NewYork_a&#x27;): 83, np.str_(&#x27;NewYork_a_ADHD&#x27;): 25, np.str_(&#x27;Newark&#x27;): 19, np.str_(&#x27;Oulu&#x27;): 102, np.str_(&#x27;Oxford&#x27;): 22, np.str_(&#x27;PaloAlto&#x27;): 17, np.str_(&#x27;Pittsburgh&#x27;): 3, np.str_(&#x27;Queensland&#x27;): 19, np.str_(&#x27;SaintLouis&#x27;): 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.251224489795916), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.05332767402377), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;M&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.48959100204499), &#x27;min&#x27;: np.float64(9.21), &#x27;max&#x27;: np.float64(78.0)}}}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.28333333333333), &#x27;min&#x27;: np.float64(13.41), &#x27;max&#x27;: np.float64(40.98)}}, np.str_(&#x27;AnnArbor_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.40625), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(79.0)}}, np.str_(&#x27;Atlanta&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(30.892857142857142), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(57.0)}}, np.str_(&#x27;Baltimore&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.26086956521739), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(40.0)}}, np.str_(&#x27;Bangor&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.4), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(38.0)}}, np.str_(&#x27;Beijing_Zang&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.161616161616163), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(26.0)}}, np.str_(&#x27;Berlin_Margulies&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.76923076923077), &#x27;min&#x27;: np.float64(23.0), &#x27;max&#x27;: np.float64(44.0)}}, np.str_(&#x27;Cambridge_Buckner&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.03030303030303), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(30.0)}}, np.str_(&#x27;Cleveland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(43.54838709677419), &#x27;min&#x27;: np.float64(24.0), &#x27;max&#x27;: np.float64(60.0)}}, np.str_(&#x27;ICBM&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.04705882352941), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;Leiden_2180&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(27.0)}}, np.str_(&#x27;Leiden_2200&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.68421052631579), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(28.0)}}, np.str_(&#x27;Milwaukee_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(53.58695652173913), &#x27;min&#x27;: np.float64(44.0), &#x27;max&#x27;: np.float64(65.0)}}, np.str_(&#x27;Munchen&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(68.13333333333334), &#x27;min&#x27;: np.float64(63.0), &#x27;max&#x27;: np.float64(74.0)}}, np.str_(&#x27;NewYork_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.507710843373495), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(49.16)}}, np.str_(&#x27;NewYork_a_ADHD&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(34.9952), &#x27;min&#x27;: np.float64(20.69), &#x27;max&#x27;: np.float64(50.9)}}, np.str_(&#x27;Newark&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.105263157894736), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(39.0)}}, np.str_(&#x27;Oulu&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.519607843137255), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(23.0)}}, np.str_(&#x27;Oxford&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(35.0)}}, np.str_(&#x27;PaloAlto&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.470588235294116), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(46.0)}}, np.str_(&#x27;Pittsburgh&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.333333333333336), &#x27;min&#x27;: np.float64(25.0), &#x27;max&#x27;: np.float64(47.0)}}, np.str_(&#x27;Queensland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.94736842105263), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(34.0)}}, np.str_(&#x27;SaintLouis&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.096774193548388), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(29.0)}}}}</dd></dl></div></li></ul></div></div>



Plot the results
----------------

The PCNtoolkit offers are a number of different plotting functions: 1.
plot_centiles: Plot the predicted centiles for a model 2. plot_qq: Plot
the QQ-plot of the predicted Z-scores 3. plot_ridge: Plot density plots
of the predicted Z-scores

Let’s start with the centiles.

.. code:: ipython3

    plot_centiles_advanced(
        model,
        centiles=[0.05, 0.5, 0.95],  # Plot these centiles, the default is [0.05, 0.25, 0.5, 0.75, 0.95]
        scatter_data=train,  # Scatter this data along with the centiles
        batch_effects={"site": ["Beijing_Zang", "AnnArbor_a"], "sex": ["M"]},  # Highlight these groups
        show_other_data=True,  # scatter data not in those groups as smaller black circles
        harmonize_data=True,  # harmonize the scatterdata, this means that we 'remove' the batch effects from the data, by simulating what the data would have looked like if all data was from the same batch.
        show_yhat=True,
    )


.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:48 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	site (1)
    	sex (1)
        
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for CortexVol.
    Process: 75222 - 2025-11-20 13:13:48 - Computing centiles for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for CortexVol.
    Process: 75222 - 2025-11-20 13:13:48 - Computing yhat for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:49 - Harmonizing data on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:49 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:49 - Harmonizing data for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:49 - Harmonizing data for CortexVol.
    Process: 75222 - 2025-11-20 13:13:49 - Harmonizing data for Right-Amygdala.
    


.. image:: 02_BLR_files/02_BLR_14_1.png



.. image:: 02_BLR_files/02_BLR_14_2.png



.. image:: 02_BLR_files/02_BLR_14_3.png



.. image:: 02_BLR_files/02_BLR_14_4.png


Now let’s see the qq plots

.. code:: ipython3

    plot_qq(test, plot_id_line=True)



.. image:: 02_BLR_files/02_BLR_16_0.png



.. image:: 02_BLR_files/02_BLR_16_1.png



.. image:: 02_BLR_files/02_BLR_16_2.png



.. image:: 02_BLR_files/02_BLR_16_3.png


We can also split the QQ plots by batch effects:

.. code:: ipython3

    plot_qq(test, plot_id_line=True, hue_data="sex", split_data="sex")
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": (0, 0, 0, 0)})



.. image:: 02_BLR_files/02_BLR_18_0.png



.. image:: 02_BLR_files/02_BLR_18_1.png



.. image:: 02_BLR_files/02_BLR_18_2.png



.. image:: 02_BLR_files/02_BLR_18_3.png


And finally the ridge plot:

.. code:: ipython3

    plot_ridge(
        train, "Z", split_by="sex"
    )  # We can also show the 'Y' variable, and that will show the marginal distribution of the response variable, per batch effect.


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:817: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()
    


.. image:: 02_BLR_files/02_BLR_20_1.png


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:817: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()
    


.. image:: 02_BLR_files/02_BLR_20_3.png


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:817: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()
    


.. image:: 02_BLR_files/02_BLR_20_5.png


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/seaborn/axisgrid.py:123: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
      self._figure.tight_layout(*args, **kwargs)
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:817: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
      plt.tight_layout()
    


.. image:: 02_BLR_files/02_BLR_20_7.png


Model evaluation statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluation statistcs are stored in the NormData object:

.. code:: ipython3

    display(train.get_statistics_df())
    display(test.get_statistics_df())



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
          <th>statistic</th>
          <th>EXPV</th>
          <th>MACE</th>
          <th>MAPE</th>
          <th>MSLL</th>
          <th>NLL</th>
          <th>R2</th>
          <th>RMSE</th>
          <th>Rho</th>
          <th>Rho_p</th>
          <th>SMSE</th>
          <th>ShapiroW</th>
        </tr>
        <tr>
          <th>response_vars</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>CortexVol</th>
          <td>0.523230</td>
          <td>0.009791</td>
          <td>0.056926</td>
          <td>-11.207204</td>
          <td>1.073368</td>
          <td>0.523205</td>
          <td>36000.916493</td>
          <td>0.713926</td>
          <td>3.040546e-135</td>
          <td>0.476795</td>
          <td>0.998480</td>
        </tr>
        <tr>
          <th>Right-Amygdala</th>
          <td>0.396912</td>
          <td>0.007007</td>
          <td>0.086440</td>
          <td>-5.747060</td>
          <td>1.173136</td>
          <td>0.396909</td>
          <td>190.264094</td>
          <td>0.611857</td>
          <td>1.158651e-89</td>
          <td>0.603091</td>
          <td>0.997748</td>
        </tr>
        <tr>
          <th>Right-Lateral-Ventricle</th>
          <td>0.233741</td>
          <td>0.036891</td>
          <td>0.373467</td>
          <td>-8.578380</td>
          <td>1.096443</td>
          <td>0.233719</td>
          <td>3370.384251</td>
          <td>0.411791</td>
          <td>1.310378e-36</td>
          <td>0.766281</td>
          <td>0.959359</td>
        </tr>
        <tr>
          <th>WM-hypointensities</th>
          <td>0.383733</td>
          <td>0.040603</td>
          <td>0.282096</td>
          <td>-7.373238</td>
          <td>0.750982</td>
          <td>0.383214</td>
          <td>641.406956</td>
          <td>0.565555</td>
          <td>5.014669e-74</td>
          <td>0.616786</td>
          <td>0.945639</td>
        </tr>
      </tbody>
    </table>
    </div>



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
          <th>statistic</th>
          <th>EXPV</th>
          <th>MACE</th>
          <th>MAPE</th>
          <th>MSLL</th>
          <th>NLL</th>
          <th>R2</th>
          <th>RMSE</th>
          <th>Rho</th>
          <th>Rho_p</th>
          <th>SMSE</th>
          <th>ShapiroW</th>
        </tr>
        <tr>
          <th>response_vars</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>CortexVol</th>
          <td>0.387191</td>
          <td>0.014815</td>
          <td>0.065150</td>
          <td>-11.084444</td>
          <td>1.133116</td>
          <td>0.375603</td>
          <td>38682.292062</td>
          <td>0.635332</td>
          <td>8.113662e-26</td>
          <td>0.624397</td>
          <td>0.995039</td>
        </tr>
        <tr>
          <th>Right-Amygdala</th>
          <td>0.285233</td>
          <td>0.018519</td>
          <td>0.094162</td>
          <td>-5.637773</td>
          <td>1.244753</td>
          <td>0.284778</td>
          <td>199.538370</td>
          <td>0.512633</td>
          <td>7.129297e-16</td>
          <td>0.715222</td>
          <td>0.988757</td>
        </tr>
        <tr>
          <th>Right-Lateral-Ventricle</th>
          <td>0.185347</td>
          <td>0.049074</td>
          <td>0.420343</td>
          <td>-8.402101</td>
          <td>1.288404</td>
          <td>0.184190</td>
          <td>3532.567866</td>
          <td>0.246607</td>
          <td>2.521884e-04</td>
          <td>0.815810</td>
          <td>0.937045</td>
        </tr>
        <tr>
          <th>WM-hypointensities</th>
          <td>-2.474878</td>
          <td>0.037963</td>
          <td>0.390576</td>
          <td>-6.978748</td>
          <td>0.845802</td>
          <td>-2.485391</td>
          <td>1129.919632</td>
          <td>0.469793</td>
          <td>2.948349e-13</td>
          <td>3.485391</td>
          <td>0.937601</td>
        </tr>
      </tbody>
    </table>
    </div>


What’s next?
------------

Now we have a normative Bayesian linear regression model, we can use it
to:

- Make predictions on new data
- Harmonize data, this means that we ‘remove’ the batch effects from the
  data, by simulating what the data would have looked like if all data
  was from the same batch.
- Synthesize new data
- Extend the model using data from new batches

Predicting
~~~~~~~~~~

.. code:: ipython3

    model.predict(test)


.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:50 - Making predictions on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:50 - Computing z-scores for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:50 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:50 - Computing z-scores for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:50 - Computing z-scores for CortexVol.
    Process: 75222 - 2025-11-20 13:13:50 - Computing z-scores for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:50 - Computing centiles for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:50 - Computing centiles for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:50 - Computing centiles for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:50 - Computing centiles for CortexVol.
    Process: 75222 - 2025-11-20 13:13:50 - Computing centiles for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:50 - Computing log-probabilities for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:50 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:50 - Computing log-probabilities for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:50 - Computing log-probabilities for CortexVol.
    Process: 75222 - 2025-11-20 13:13:50 - Computing log-probabilities for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:50 - Computing yhat for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:50 - Computing yhat for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:50 - Computing yhat for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:50 - Computing yhat for CortexVol.
    Process: 75222 - 2025-11-20 13:13:50 - Computing yhat for Right-Amygdala.
    



.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base rgba(0, 0, 0, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, white)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
      );
    }
    
    html[theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    body.vscode-dark {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base, rgba(255, 255, 255, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, #111111)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
      );
    }
    
    .xr-wrap {
      display: block !important;
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: inline-block;
      opacity: 0;
      height: 0;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
      border: 2px solid transparent !important;
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:focus + label {
      border: 2px solid var(--xr-font-color0) !important;
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: "►";
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: "▼";
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: "(";
    }
    
    .xr-dim-list:after {
      content: ")";
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ",";
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      border-color: var(--xr-background-color-row-odd);
      margin-bottom: 0;
      padding-top: 2px;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
      border-color: var(--xr-background-color-row-even);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-index-preview {
      grid-column: 2 / 5;
      color: var(--xr-font-color2);
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      display: none;
      border-top: 2px dotted var(--xr-background-color);
      padding-bottom: 20px !important;
      padding-top: 10px !important;
    }
    
    .xr-var-attrs-in + label,
    .xr-var-data-in + label,
    .xr-index-data-in + label {
      padding: 0 1px;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data,
    .xr-index-data-in:checked ~ .xr-index-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-data > pre,
    .xr-index-data > pre,
    .xr-var-data > table > tbody > tr {
      background-color: transparent !important;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-index-name div,
    .xr-index-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2,
    .xr-no-icon {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    
    .xr-var-attrs-in:checked + label > .xr-icon-file-text2,
    .xr-var-data-in:checked + label > .xr-icon-database,
    .xr-index-data-in:checked + label > .xr-icon-database {
      color: var(--xr-font-color0);
      filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
      stroke-width: 0.8px;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 98kB
    Dimensions:            (observations: 216, response_vars: 4, covariates: 1,
                            batch_effect_dims: 2, statistic: 11, centile: 5)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 368B &#x27;WM-hypointensities&#x27; ... &#x27;Co...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * statistic          (statistic) &lt;U8 352B &#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
    Data variables:
        subject_ids        (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 7kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 7kB 0.991 ... -1...
        logp               (observations, response_vars) float64 7kB -1.869 ... -...
        Yhat               (observations, response_vars) float64 7kB 1.91e+03 ......
        statistics         (response_vars, statistic) float64 352B -2.475 ... 0.995
        centiles           (centile, observations, response_vars) float64 35kB 57...
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.2...
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-f975b6cf-04e4-47ea-af2a-710c79ab9d20' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-f975b6cf-04e4-47ea-af2a-710c79ab9d20' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 4</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>statistic</span>: 11</li><li><span class='xr-has-index'>centile</span>: 5</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-2611b637-1112-4b0c-822f-c468e72b8399' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2611b637-1112-4b0c-822f-c468e72b8399' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-2f297443-622e-4ae7-8e8a-689f67eea701' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2f297443-622e-4ae7-8e8a-689f67eea701' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3db15c51-2a38-431c-b8ad-a680b1299f2c' class='xr-var-data-in' type='checkbox'><label for='data-3db15c51-2a38-431c-b8ad-a680b1299f2c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Cortex...</div><input id='attrs-4c230603-459d-4202-8e01-30149da4ec95' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4c230603-459d-4202-8e01-30149da4ec95' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-24f9bd3b-5890-4a3d-9ed1-512935fcee01' class='xr-var-data-in' type='checkbox'><label for='data-24f9bd3b-5890-4a3d-9ed1-512935fcee01' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-aa2f62bf-97a2-4ad0-b1cb-5db53b57a5e9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-aa2f62bf-97a2-4ad0-b1cb-5db53b57a5e9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5f07cc5c-5293-4663-8dcc-70d2e0e1a57f' class='xr-var-data-in' type='checkbox'><label for='data-5f07cc5c-5293-4663-8dcc-70d2e0e1a57f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-8047680c-e15b-4614-9ac3-49a77578c1b2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8047680c-e15b-4614-9ac3-49a77578c1b2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-96226fae-1db7-4c0e-ad48-944c198d9b2d' class='xr-var-data-in' type='checkbox'><label for='data-96226fae-1db7-4c0e-ad48-944c198d9b2d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;MACE&#x27; ... &#x27;SMSE&#x27; &#x27;ShapiroW&#x27;</div><input id='attrs-494454d4-5a14-4d39-8485-03a5b5164e4e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-494454d4-5a14-4d39-8485-03a5b5164e4e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b32f66c3-a92c-4e33-b8ca-ab9394215ed7' class='xr-var-data-in' type='checkbox'><label for='data-b32f66c3-a92c-4e33-b8ca-ab9394215ed7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-76df7af8-95e6-4093-bac2-a2d0320e5be0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-76df7af8-95e6-4093-bac2-a2d0320e5be0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ab6dc74c-96a5-4506-a7a5-51f4abbc7e9e' class='xr-var-data-in' type='checkbox'><label for='data-ab6dc74c-96a5-4506-a7a5-51f4abbc7e9e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-da057b9a-fb34-45fa-ba72-dc188052c608' class='xr-section-summary-in' type='checkbox'  checked><label for='section-da057b9a-fb34-45fa-ba72-dc188052c608' class='xr-section-summary' >Data variables: <span>(9)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-d600787e-fc29-49c4-856d-4b822be09eb1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d600787e-fc29-49c4-856d-4b822be09eb1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8c78dbf8-40dd-4d79-bd25-776220462afd' class='xr-var-data-in' type='checkbox'><label for='data-8c78dbf8-40dd-4d79-bd25-776220462afd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
           &#x27;ICBM_sub47658&#x27;, &#x27;AnnArbor_b_sub45569&#x27;, &#x27;Beijing_Zang_sub18960&#x27;,
           &#x27;Leiden_2200_sub18456&#x27;, &#x27;Berlin_Margulies_sub27711&#x27;,
           &#x27;Beijing_Zang_sub87776&#x27;, &#x27;Milwaukee_b_sub63196&#x27;,
           &#x27;Beijing_Zang_sub07144&#x27;, &#x27;Atlanta_sub76280&#x27;,
           &#x27;Beijing_Zang_sub40037&#x27;, &#x27;Cambridge_Buckner_sub17737&#x27;,
           &#x27;ICBM_sub89049&#x27;, &#x27;ICBM_sub55656&#x27;, &#x27;Oulu_sub45566&#x27;,
           &#x27;Beijing_Zang_sub89088&#x27;, &#x27;Atlanta_sub16563&#x27;,
           &#x27;Cambridge_Buckner_sub51172&#x27;, &#x27;Oulu_sub98739&#x27;,
           &#x27;Queensland_sub49845&#x27;, &#x27;Cambridge_Buckner_sub84256&#x27;,
           &#x27;Cleveland_sub80263&#x27;, &#x27;ICBM_sub16607&#x27;, &#x27;Newark_sub46570&#x27;,
           &#x27;NewYork_a_sub88286&#x27;, &#x27;Cambridge_Buckner_sub02591&#x27;,
           &#x27;Oulu_sub66467&#x27;, &#x27;Beijing_Zang_sub74386&#x27;, &#x27;Newark_sub55760&#x27;,
           &#x27;ICBM_sub30623&#x27;, &#x27;Oulu_sub68752&#x27;, &#x27;Leiden_2180_sub19281&#x27;,
           &#x27;Beijing_Zang_sub50972&#x27;, &#x27;Beijing_Zang_sub85030&#x27;,
           &#x27;Milwaukee_b_sub36386&#x27;, &#x27;Baltimore_sub31837&#x27;, &#x27;PaloAlto_sub84978&#x27;,
           &#x27;Oulu_sub01077&#x27;, &#x27;NewYork_a_ADHD_sub54828&#x27;, &#x27;PaloAlto_sub96705&#x27;,
           &#x27;Cambridge_Buckner_sub40635&#x27;, &#x27;ICBM_sub66794&#x27;,
           &#x27;Beijing_Zang_sub46541&#x27;, &#x27;Beijing_Zang_sub87089&#x27;,
           &#x27;Pittsburgh_sub97823&#x27;, &#x27;Beijing_Zang_sub98617&#x27;, &#x27;ICBM_sub92028&#x27;,
    ...
           &#x27;Leiden_2200_sub04484&#x27;, &#x27;Beijing_Zang_sub80163&#x27;, &#x27;ICBM_sub02382&#x27;,
           &#x27;Cambridge_Buckner_sub77435&#x27;, &#x27;NewYork_a_sub54887&#x27;,
           &#x27;Oulu_sub85532&#x27;, &#x27;Baltimore_sub73823&#x27;, &#x27;Beijing_Zang_sub29590&#x27;,
           &#x27;Oulu_sub99718&#x27;, &#x27;Beijing_Zang_sub08455&#x27;, &#x27;Beijing_Zang_sub85543&#x27;,
           &#x27;Cambridge_Buckner_sub45354&#x27;, &#x27;Beijing_Zang_sub07717&#x27;,
           &#x27;Baltimore_sub76160&#x27;, &#x27;Beijing_Zang_sub17093&#x27;,
           &#x27;AnnArbor_b_sub90127&#x27;, &#x27;SaintLouis_sub73002&#x27;,
           &#x27;Queensland_sub93238&#x27;, &#x27;Cleveland_sub34189&#x27;,
           &#x27;Cambridge_Buckner_sub89107&#x27;, &#x27;Atlanta_sub75153&#x27;,
           &#x27;NewYork_a_ADHD_sub73035&#x27;, &#x27;Cambridge_Buckner_sub59434&#x27;,
           &#x27;Milwaukee_b_sub44912&#x27;, &#x27;Cleveland_sub46739&#x27;, &#x27;Oulu_sub20495&#x27;,
           &#x27;SaintLouis_sub28304&#x27;, &#x27;Cambridge_Buckner_sub35430&#x27;,
           &#x27;Oulu_sub86362&#x27;, &#x27;Newark_sub58526&#x27;, &#x27;Leiden_2180_sub12255&#x27;,
           &#x27;ICBM_sub48210&#x27;, &#x27;Cambridge_Buckner_sub77989&#x27;,
           &#x27;Berlin_Margulies_sub75506&#x27;, &#x27;NewYork_a_sub29216&#x27;,
           &#x27;Beijing_Zang_sub05267&#x27;, &#x27;AnnArbor_b_sub18546&#x27;, &#x27;Oulu_sub75620&#x27;,
           &#x27;AnnArbor_b_sub30250&#x27;, &#x27;Berlin_Margulies_sub86111&#x27;,
           &#x27;Beijing_Zang_sub89592&#x27;, &#x27;Beijing_Zang_sub68012&#x27;,
           &#x27;NewYork_a_sub50559&#x27;, &#x27;Munchen_sub66933&#x27;,
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 5.035e+05</div><input id='attrs-cce327ad-a450-4dc4-a33e-28acc2b16cfb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cce327ad-a450-4dc4-a33e-28acc2b16cfb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-20323f07-7871-4a9d-b531-2af8e6716b24' class='xr-var-data-in' type='checkbox'><label for='data-20323f07-7871-4a9d-b531-2af8e6716b24' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.72140000e+03, 1.28916000e+04, 1.43940000e+03, 4.57858328e+05],
           [1.14310000e+03, 9.91910000e+03, 1.64970000e+03, 5.26780362e+05],
           [9.55800000e+02, 7.47730000e+03, 1.83850000e+03, 4.95744471e+05],
           [1.47390000e+03, 1.43021000e+04, 1.86770000e+03, 5.85303839e+05],
           [7.57800000e+02, 4.11930000e+03, 1.32500000e+03, 3.33111552e+05],
           [8.71100000e+02, 5.03090000e+03, 1.90730000e+03, 5.10794940e+05],
           [1.20730000e+03, 1.78664000e+04, 2.02220000e+03, 5.50533325e+05],
           [5.95000000e+02, 5.00790000e+03, 2.01070000e+03, 4.67673977e+05],
           [6.82400000e+02, 7.28660000e+03, 1.45630000e+03, 4.60129533e+05],
           [4.45100000e+02, 5.74290000e+03, 1.47450000e+03, 4.44494817e+05],
           [1.62000000e+03, 3.71370000e+03, 2.00110000e+03, 5.59424624e+05],
           [6.02800000e+02, 5.30120000e+03, 1.36100000e+03, 4.21551234e+05],
           [1.43250000e+03, 4.42970000e+03, 1.65080000e+03, 5.19842763e+05],
           [1.90820000e+03, 3.57810000e+03, 1.88370000e+03, 5.06679262e+05],
           [1.83400000e+03, 3.27190000e+03, 2.05120000e+03, 5.35569987e+05],
           [4.59600000e+02, 3.98580000e+03, 1.45470000e+03, 4.67607555e+05],
           [1.21000000e+03, 8.72130000e+03, 1.71430000e+03, 5.30904612e+05],
           [8.45900000e+02, 6.59310000e+03, 1.61900000e+03, 5.09371867e+05],
           [9.95200000e+02, 7.04020000e+03, 1.99490000e+03, 4.60068379e+05],
           [1.73470000e+03, 4.01480000e+03, 1.51620000e+03, 4.87269373e+05],
    ...
           [7.85800000e+02, 5.70900000e+03, 1.47480000e+03, 4.53982166e+05],
           [2.24010000e+03, 4.36660000e+03, 2.04210000e+03, 5.58453123e+05],
           [7.58100000e+02, 6.52980000e+03, 1.56730000e+03, 4.73575183e+05],
           [1.44050000e+03, 6.70530000e+03, 1.20540000e+03, 3.82788491e+05],
           [8.18600000e+02, 9.38330000e+03, 1.96740000e+03, 5.02713911e+05],
           [3.76990000e+03, 1.58644000e+04, 1.79170000e+03, 5.12490348e+05],
           [8.80200000e+02, 4.37020000e+03, 1.75520000e+03, 4.37300069e+05],
           [8.23900000e+02, 6.37900000e+03, 1.57650000e+03, 5.67331908e+05],
           [2.11390000e+03, 1.07225000e+04, 1.84380000e+03, 5.12273764e+05],
           [7.41900000e+02, 8.80170000e+03, 1.60640000e+03, 4.91973562e+05],
           [1.33390000e+03, 6.98000000e+03, 1.74850000e+03, 4.78907154e+05],
           [7.07300000e+02, 5.68070000e+03, 1.53450000e+03, 4.74077083e+05],
           [1.13410000e+03, 5.59220000e+03, 1.62620000e+03, 4.54163909e+05],
           [4.38600000e+02, 6.33000000e+03, 1.59670000e+03, 4.68067037e+05],
           [9.66300000e+02, 9.21550000e+03, 1.78250000e+03, 5.09199708e+05],
           [4.24300000e+02, 4.51110000e+03, 1.70200000e+03, 5.26635258e+05],
           [6.04700000e+02, 7.59080000e+03, 1.69930000e+03, 5.20499663e+05],
           [2.34320000e+03, 1.71923000e+04, 1.79380000e+03, 4.86680791e+05],
           [2.72170000e+03, 6.08600000e+03, 2.32470000e+03, 6.10402006e+05],
           [7.03500000e+02, 1.07003000e+04, 1.67620000e+03, 5.03535771e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-e906c8ed-0f76-49c1-9fb9-5bbd81feaa9b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e906c8ed-0f76-49c1-9fb9-5bbd81feaa9b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-57de0e79-082b-4732-989f-bbf9bcd93787' class='xr-var-data-in' type='checkbox'><label for='data-57de0e79-082b-4732-989f-bbf9bcd93787' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
           [23.27],
           [22.  ],
           [42.  ],
           [63.  ],
           [23.  ],
           [21.  ],
           [26.  ],
           [21.  ],
           [49.  ],
           [20.  ],
           [23.  ],
           [20.  ],
           [26.  ],
           [35.  ],
           [21.  ],
           [22.  ],
           [19.  ],
           [34.  ],
           [18.  ],
    ...
           [21.  ],
           [20.  ],
           [22.  ],
           [25.  ],
           [25.  ],
           [73.  ],
           [22.  ],
           [28.  ],
           [29.06],
           [19.  ],
           [20.  ],
           [22.  ],
           [19.  ],
           [24.  ],
           [21.  ],
           [24.  ],
           [22.79],
           [72.  ],
           [23.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-79ca3058-573d-4196-b667-67ec6397ccaa' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-79ca3058-573d-4196-b667-67ec6397ccaa' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cae25e21-d8d1-4dd1-a828-dc382cfaf457' class='xr-var-data-in' type='checkbox'><label for='data-cae25e21-d8d1-4dd1-a828-dc382cfaf457' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Milwaukee_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
    ...
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Newark&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2180&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;M&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.991 0.8183 ... -1.358 -1.272</div><input id='attrs-e9803996-b255-4e94-b60c-e5f8e94a8efc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e9803996-b255-4e94-b60c-e5f8e94a8efc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-56728e8b-f419-420d-a109-a6f762c80e75' class='xr-var-data-in' type='checkbox'><label for='data-56728e8b-f419-420d-a109-a6f762c80e75' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.9910344 ,  0.81830048, -0.55018133,  0.13955562],
           [-0.01085574,  1.38811541, -1.28944887, -0.22201834],
           [ 0.17713812,  0.33628493,  0.61264106,  0.29792913],
           [-0.01677579,  1.60929134, -0.07031814,  1.26890505],
           [-1.15537626, -1.38013383, -0.26679411, -1.58011453],
           [-0.83705405, -0.6925975 ,  0.45511936, -0.51539529],
           [ 0.62253858,  2.23320074,  0.27847909,  0.31344571],
           [ 0.21661963, -0.6753277 ,  1.98622612, -0.63117465],
           [-1.02395187,  0.81704349, -0.95300482, -0.84426157],
           [-0.66455908, -0.40326532, -0.73652474,  0.11213451],
           [ 1.49555376, -1.21726499,  0.77389599,  0.71203431],
           [-0.66321267, -0.33552826, -1.12904778, -1.03938937],
           [ 1.28383024, -0.49977552,  0.05769295,  1.18720052],
           [ 1.90999261, -0.69351978,  1.29145737,  1.80165049],
           [ 0.98592445, -1.58890315,  0.62572662,  0.09924965],
           [-1.12495195, -0.79240985, -0.98508816, -1.08859152],
           [-0.59763476,  0.62704661, -0.65512582, -0.36102778],
           [-0.62407676,  0.56323037, -0.16281503,  0.76715475],
           [-0.55957317, -0.13658049,  0.85962834, -1.06197019],
           [ 2.13100646, -0.65057486, -1.05001494,  0.87785052],
    ...
           [ 0.10620058, -0.11162761, -0.90209223, -1.33267893],
           [ 3.04842362, -0.88711349,  0.89548308,  1.74547361],
           [-1.39484774,  0.16743464, -0.27712282, -0.68632353],
           [-0.46595139,  0.14880783, -1.2583806 , -0.56745362],
           [-0.87906482,  0.6308172 ,  0.64504465, -0.51910144],
           [ 0.73592004,  0.11971089,  0.32803703,  0.86277602],
           [-1.13954502, -0.45868536,  0.57624067, -0.48693253],
           [ 0.09872182, -0.50135757, -1.51486826,  0.62392987],
           [ 1.98539968,  1.08076884, -0.31257137, -0.35782521],
           [-0.92818113,  1.59376006, -0.23560175,  0.11906651],
           [-0.17402345,  0.32419477,  0.32459488, -0.33556416],
           [-1.50267681, -0.17571033, -0.45561085, -0.66991848],
           [-0.42778381,  0.14953608,  0.71983899,  0.19909464],
           [ 0.09656318, -0.20009939, -0.03069303, -0.66919995],
           [-0.56383703,  1.19393477, -0.34018551, -0.76691625],
           [-1.47972475, -0.50524055,  0.53224766,  1.53950791],
           [-1.27217678,  0.51002814, -0.96896359, -0.40012592],
           [-0.70954373,  0.38006276,  0.30542294, -0.03362917],
           [ 3.51471532, -0.12046593,  1.78738181,  2.73068326],
           [-1.23766073,  1.11498484, -1.35814472, -1.2716136 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.869 -1.603 ... -1.467 -1.283</div><input id='attrs-0b91a0eb-8b9d-4140-9ddf-378d4c3b3c34' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0b91a0eb-8b9d-4140-9ddf-378d4c3b3c34' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b3eb168f-ae35-4c6a-9b96-3b956c75b577' class='xr-var-data-in' type='checkbox'><label for='data-b3eb168f-ae35-4c6a-9b96-3b956c75b577' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.86906501e+00, -1.60266085e+00, -1.16872443e+00,
            -7.47478268e-01],
           [-1.58469803e-01, -1.62580449e+00, -1.36632897e+00,
            -7.11132196e-01],
           [-2.19181521e-01, -5.45926189e-01, -9.02398690e-01,
            -5.15012893e-01],
           [-4.34454374e-01, -2.74765922e+00, -7.26514955e-01,
            -1.96605917e+00],
           [-1.67122763e+00, -1.68727093e+00, -4.79816067e-01,
            -2.21938578e+00],
           [-5.15802252e-01, -6.57001146e-01, -8.20935222e-01,
            -4.70193633e-01],
           [-3.38754242e-01, -4.24650827e+00, -1.04503171e+00,
            -8.48054908e-01],
           [-5.02399231e-01, -7.88898804e-01, -2.98795362e+00,
            -7.62202461e-01],
           [-6.99642072e-01, -6.71339830e-01, -1.23608486e+00,
            -7.61475166e-01],
           [-1.34119838e+00, -9.05503762e-01, -8.85981354e-01,
            -6.66178270e-01],
    ...
           [-1.43832667e-01, -5.66257958e-01, -6.66723409e-01,
            -5.93002770e-01],
           [-1.51256793e+00, -5.02001383e-01, -7.58211952e-01,
            -6.09029430e-01],
           [-1.92262486e-01, -5.14451109e-01, -9.89111305e-01,
            -8.14119623e-01],
           [-5.75832982e-01, -5.45905002e-01, -6.62036576e-01,
            -8.05311515e-01],
           [-2.16911706e-01, -1.27138629e+00, -5.49341186e-01,
            -5.49459895e-01],
           [-1.53837639e+00, -5.98702577e-01, -6.60941081e-01,
            -1.72892590e+00],
           [-1.21627533e+00, -4.87960579e-01, -9.86555788e-01,
            -7.17502315e-01],
           [-1.34493012e+00, -1.73880392e+00, -1.28542246e+00,
            -1.07871724e+00],
           [-6.76604624e+00, -3.82760420e-01, -2.89324088e+00,
            -4.88027805e+00],
           [-1.18852124e+00, -1.60949381e+00, -1.46708031e+00,
            -1.28328036e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.91e+03 1.043e+04 ... 5.667e+05</div><input id='attrs-3bdc7000-56a3-4e81-a0c4-18fd05601cc0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3bdc7000-56a3-4e81-a0c4-18fd05601cc0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eb6cd3a5-5d80-446a-ae40-40acd5e3bbcf' class='xr-var-data-in' type='checkbox'><label for='data-eb6cd3a5-5d80-446a-ae40-40acd5e3bbcf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.91041133e+03,  1.04341585e+04,  1.54646809e+03,
             4.43073070e+05],
           [ 1.13032022e+03,  6.64197321e+03,  1.90995371e+03,
             5.46278301e+05],
           [ 8.39178018e+02,  6.77321275e+03,  1.73348173e+03,
             4.85216353e+05],
           [ 1.51626904e+03,  8.19404627e+03,  1.92483788e+03,
             5.28245435e+05],
           [ 1.81972723e+03,  8.98371634e+03,  1.35008245e+03,
             3.99081670e+05],
           [ 1.16175434e+03,  6.67434652e+03,  1.84788893e+03,
             5.31647015e+05],
           [ 9.28966969e+02,  7.97159280e+03,  2.00838678e+03,
             5.44982930e+05],
           [ 3.71682343e+02,  6.93649825e+03,  1.60830885e+03,
             4.89690114e+05],
           [ 1.02365924e+03,  5.48565741e+03,  1.62548221e+03,
             4.83940391e+05],
           [ 8.76457633e+02,  7.84997789e+03,  1.58394865e+03,
             4.32454963e+05],
    ...
           [ 1.40881234e+03,  6.18307347e+03,  1.69142305e+03,
             4.91022683e+05],
           [ 1.33485770e+03,  6.13138087e+03,  1.60315150e+03,
             4.94641280e+05],
           [ 1.28497416e+03,  5.05437662e+03,  1.42063754e+03,
             4.34115945e+05],
           [ 2.65544556e+02,  7.05756249e+03,  1.58564808e+03,
             4.92284380e+05],
           [ 1.14296630e+03,  6.51001714e+03,  1.86556449e+03,
             5.36309774e+05],
           [ 1.02322696e+03,  5.64189952e+03,  1.59986772e+03,
             4.77011978e+05],
           [ 1.12489000e+03,  6.55760649e+03,  1.90531392e+03,
             5.47638612e+05],
           [ 3.70760320e+03,  1.70380342e+04,  1.72029246e+03,
             4.89972976e+05],
           [ 1.33301797e+03,  6.39661753e+03,  1.90963880e+03,
             5.00086720e+05],
           [ 1.23623232e+03,  7.42824756e+03,  1.97755494e+03,
             5.66741543e+05]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.475 0.03796 ... 0.6244 0.995</div><input id='attrs-3d10783a-af10-4fff-96cf-f1f2de6841c2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3d10783a-af10-4fff-96cf-f1f2de6841c2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-629f50bd-4efe-4415-a9ab-d89fad4bf83b' class='xr-var-data-in' type='checkbox'><label for='data-629f50bd-4efe-4415-a9ab-d89fad4bf83b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-2.47487797e+00,  3.79629630e-02,  3.90576314e-01,
            -6.97874850e+00,  8.45801650e-01, -2.48539139e+00,
             1.12991963e+03,  4.69792686e-01,  2.94834937e-13,
             3.48539139e+00,  9.37601263e-01],
           [ 1.85347407e-01,  4.90740741e-02,  4.20342721e-01,
            -8.40210129e+00,  1.28840444e+00,  1.84189959e-01,
             3.53256787e+03,  2.46607074e-01,  2.52188374e-04,
             8.15810041e-01,  9.37045362e-01],
           [ 2.85233438e-01,  1.85185185e-02,  9.41622245e-02,
            -5.63777308e+00,  1.24475296e+00,  2.84777724e-01,
             1.99538370e+02,  5.12632996e-01,  7.12929722e-16,
             7.15222276e-01,  9.88757003e-01],
           [ 3.87191352e-01,  1.48148148e-02,  6.51500829e-02,
            -1.10844441e+01,  1.13311599e+00,  3.75602928e-01,
             3.86822921e+04,  6.35331704e-01,  8.11366151e-26,
             6.24397072e-01,  9.95039182e-01]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>571.2 4.966e+03 ... 6.834e+05</div><input id='attrs-c5731517-a570-41ff-a98d-9ed3cd1dd29d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c5731517-a570-41ff-a98d-9ed3cd1dd29d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5ddeefbb-4046-468f-8fd4-1a2609d58fdd' class='xr-var-data-in' type='checkbox'><label for='data-5ddeefbb-4046-468f-8fd4-1a2609d58fdd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[5.71205994e+02, 4.96564655e+03, 1.06316403e+03, 3.39817113e+05],
            [4.13035081e+02, 2.75772941e+03, 1.58879166e+03, 4.75739503e+05],
            [1.88662575e+01, 2.25430203e+03, 1.39579259e+03, 4.20080014e+05],
            ...,
            [1.61029512e+03, 8.09315163e+03, 8.95839362e+02, 3.28867059e+05],
            [8.11465978e+02, 2.24573665e+03, 1.64014264e+03, 4.51301006e+05],
            [4.87567573e+02, 2.43188395e+03, 1.62722821e+03, 4.91292413e+05]],
    
           [[1.28549504e+03, 7.46789216e+03, 1.40492724e+03, 4.16057424e+05],
            [8.78625548e+02, 5.12157892e+03, 1.75109764e+03, 5.09359965e+05],
            [5.81509378e+02, 5.00577241e+03, 1.60414586e+03, 4.62472246e+05],
            ...,
            [2.37761128e+03, 1.11499357e+04, 1.45777998e+03, 4.43279505e+05],
            [1.12683865e+03, 4.81720018e+03, 1.77644124e+03, 4.80360983e+05],
            [9.63991646e+02, 5.35635311e+03, 1.79273273e+03, 5.24834515e+05]],
    
           [[1.75209940e+03, 9.43308712e+03, 1.57379132e+03, 4.51639529e+05],
            [1.14724224e+03, 6.57372795e+03, 1.87190499e+03, 5.36352593e+05],
            [8.83889821e+02, 6.65314858e+03, 1.72490051e+03, 4.85862620e+05],
            ...,
            [3.19798573e+03, 1.44942779e+04, 1.69495484e+03, 4.88734754e+05],
            [1.33087243e+03, 6.35496078e+03, 1.88133716e+03, 4.98844602e+05],
            [1.24359390e+03, 7.13708261e+03, 1.92604343e+03, 5.54585499e+05]],
    
           [[2.35080063e+03, 1.21452946e+04, 1.71430561e+03, 4.79508558e+05],
            [1.40141325e+03, 8.05825203e+03, 2.02563162e+03, 5.71923846e+05],
            [1.14745824e+03, 8.35762438e+03, 1.85098455e+03, 5.08550678e+05],
            ...,
            [4.44734100e+03, 1.98704866e+04, 1.93281869e+03, 5.33654753e+05],
            [1.53695703e+03, 7.89723294e+03, 2.01110742e+03, 5.18370399e+05],
            [1.51726721e+03, 9.09267425e+03, 2.10455268e+03, 5.94956459e+05]],
    
           [[3.78210177e+03, 1.91696618e+04, 1.93448631e+03, 5.17064669e+05],
            [1.79054689e+03, 1.07315286e+04, 2.36031918e+03, 6.50440908e+05],
            [1.50816251e+03, 1.16498333e+04, 2.09934481e+03, 5.48052958e+05],
            ...,
            [7.51942906e+03, 3.44512783e+04, 2.62653940e+03, 6.53779222e+05],
            [1.86198242e+03, 1.06574501e+04, 2.27537187e+03, 5.53054505e+05],
            [1.96040026e+03, 1.33254385e+04, 2.50306886e+03, 6.83443625e+05]]],
          shape=(5, 216, 4))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-50e32238-75c8-4905-9981-37639036808f' class='xr-section-summary-in' type='checkbox'  ><label for='section-50e32238-75c8-4905-9981-37639036808f' class='xr-section-summary' >Indexes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>observations</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-f31ed156-bca1-496b-8e6f-96e27969cfe4' class='xr-index-data-in' type='checkbox'/><label for='index-f31ed156-bca1-496b-8e6f-96e27969cfe4' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,
           ...
             27,  959,   29,  346,  304,  264,  798,  751,  470, 1043],
          dtype=&#x27;int64&#x27;, name=&#x27;observations&#x27;, length=216))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>response_vars</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-62f4d7cf-1140-46b9-9a99-91079bf0d003' class='xr-index-data-in' type='checkbox'/><label for='index-62f4d7cf-1140-46b9-9a99-91079bf0d003' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;, &#x27;Right-Amygdala&#x27;,
           &#x27;CortexVol&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;response_vars&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>covariates</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-7b8687c2-5905-4735-8822-6e0d6285205f' class='xr-index-data-in' type='checkbox'/><label for='index-7b8687c2-5905-4735-8822-6e0d6285205f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;age&#x27;], dtype=&#x27;object&#x27;, name=&#x27;covariates&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>batch_effect_dims</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-4001cf47-3051-4631-b45b-1c430aa4869d' class='xr-index-data-in' type='checkbox'/><label for='index-4001cf47-3051-4631-b45b-1c430aa4869d' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;object&#x27;, name=&#x27;batch_effect_dims&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>statistic</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-9284cc5f-3131-4a50-841d-cad7af668d2a' class='xr-index-data-in' type='checkbox'/><label for='index-9284cc5f-3131-4a50-841d-cad7af668d2a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;EXPV&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MSLL&#x27;, &#x27;NLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;, &#x27;Rho_p&#x27;,
           &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;],
          dtype=&#x27;object&#x27;, name=&#x27;statistic&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>centile</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-77dc1005-51ba-42a7-b86f-b77d49ba4dcf' class='xr-index-data-in' type='checkbox'/><label for='index-77dc1005-51ba-42a7-b86f-b77d49ba4dcf' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0.05, 0.25, 0.5, 0.75, 0.95], dtype=&#x27;float64&#x27;, name=&#x27;centile&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a87b3517-836f-41ec-8b0b-4e29f2e9d2f7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a87b3517-836f-41ec-8b0b-4e29f2e9d2f7' class='xr-section-summary' >Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [np.str_(&#x27;F&#x27;), np.str_(&#x27;M&#x27;)], np.str_(&#x27;site&#x27;): [np.str_(&#x27;AnnArbor_a&#x27;), np.str_(&#x27;AnnArbor_b&#x27;), np.str_(&#x27;Atlanta&#x27;), np.str_(&#x27;Baltimore&#x27;), np.str_(&#x27;Bangor&#x27;), np.str_(&#x27;Beijing_Zang&#x27;), np.str_(&#x27;Berlin_Margulies&#x27;), np.str_(&#x27;Cambridge_Buckner&#x27;), np.str_(&#x27;Cleveland&#x27;), np.str_(&#x27;ICBM&#x27;), np.str_(&#x27;Leiden_2180&#x27;), np.str_(&#x27;Leiden_2200&#x27;), np.str_(&#x27;Milwaukee_b&#x27;), np.str_(&#x27;Munchen&#x27;), np.str_(&#x27;NewYork_a&#x27;), np.str_(&#x27;NewYork_a_ADHD&#x27;), np.str_(&#x27;Newark&#x27;), np.str_(&#x27;Oulu&#x27;), np.str_(&#x27;Oxford&#x27;), np.str_(&#x27;PaloAlto&#x27;), np.str_(&#x27;Pittsburgh&#x27;), np.str_(&#x27;Queensland&#x27;), np.str_(&#x27;SaintLouis&#x27;)]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x17e771ee0&gt;, {np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): 589, np.str_(&#x27;M&#x27;): 489}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): 24, np.str_(&#x27;AnnArbor_b&#x27;): 32, np.str_(&#x27;Atlanta&#x27;): 28, np.str_(&#x27;Baltimore&#x27;): 23, np.str_(&#x27;Bangor&#x27;): 20, np.str_(&#x27;Beijing_Zang&#x27;): 198, np.str_(&#x27;Berlin_Margulies&#x27;): 26, np.str_(&#x27;Cambridge_Buckner&#x27;): 198, np.str_(&#x27;Cleveland&#x27;): 31, np.str_(&#x27;ICBM&#x27;): 85, np.str_(&#x27;Leiden_2180&#x27;): 12, np.str_(&#x27;Leiden_2200&#x27;): 19, np.str_(&#x27;Milwaukee_b&#x27;): 46, np.str_(&#x27;Munchen&#x27;): 15, np.str_(&#x27;NewYork_a&#x27;): 83, np.str_(&#x27;NewYork_a_ADHD&#x27;): 25, np.str_(&#x27;Newark&#x27;): 19, np.str_(&#x27;Oulu&#x27;): 102, np.str_(&#x27;Oxford&#x27;): 22, np.str_(&#x27;PaloAlto&#x27;): 17, np.str_(&#x27;Pittsburgh&#x27;): 3, np.str_(&#x27;Queensland&#x27;): 19, np.str_(&#x27;SaintLouis&#x27;): 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.251224489795916), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {np.str_(&#x27;F&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.05332767402377), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;M&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(28.48959100204499), &#x27;min&#x27;: np.float64(9.21), &#x27;max&#x27;: np.float64(78.0)}}}, np.str_(&#x27;site&#x27;): {np.str_(&#x27;AnnArbor_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.28333333333333), &#x27;min&#x27;: np.float64(13.41), &#x27;max&#x27;: np.float64(40.98)}}, np.str_(&#x27;AnnArbor_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.40625), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(79.0)}}, np.str_(&#x27;Atlanta&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(30.892857142857142), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(57.0)}}, np.str_(&#x27;Baltimore&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.26086956521739), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(40.0)}}, np.str_(&#x27;Bangor&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.4), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(38.0)}}, np.str_(&#x27;Beijing_Zang&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.161616161616163), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(26.0)}}, np.str_(&#x27;Berlin_Margulies&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.76923076923077), &#x27;min&#x27;: np.float64(23.0), &#x27;max&#x27;: np.float64(44.0)}}, np.str_(&#x27;Cambridge_Buckner&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.03030303030303), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(30.0)}}, np.str_(&#x27;Cleveland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(43.54838709677419), &#x27;min&#x27;: np.float64(24.0), &#x27;max&#x27;: np.float64(60.0)}}, np.str_(&#x27;ICBM&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(44.04705882352941), &#x27;min&#x27;: np.float64(19.0), &#x27;max&#x27;: np.float64(85.0)}}, np.str_(&#x27;Leiden_2180&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(23.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(27.0)}}, np.str_(&#x27;Leiden_2200&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.68421052631579), &#x27;min&#x27;: np.float64(18.0), &#x27;max&#x27;: np.float64(28.0)}}, np.str_(&#x27;Milwaukee_b&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(53.58695652173913), &#x27;min&#x27;: np.float64(44.0), &#x27;max&#x27;: np.float64(65.0)}}, np.str_(&#x27;Munchen&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(68.13333333333334), &#x27;min&#x27;: np.float64(63.0), &#x27;max&#x27;: np.float64(74.0)}}, np.str_(&#x27;NewYork_a&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.507710843373495), &#x27;min&#x27;: np.float64(7.88), &#x27;max&#x27;: np.float64(49.16)}}, np.str_(&#x27;NewYork_a_ADHD&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(34.9952), &#x27;min&#x27;: np.float64(20.69), &#x27;max&#x27;: np.float64(50.9)}}, np.str_(&#x27;Newark&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(24.105263157894736), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(39.0)}}, np.str_(&#x27;Oulu&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(21.519607843137255), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(23.0)}}, np.str_(&#x27;Oxford&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(29.0), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(35.0)}}, np.str_(&#x27;PaloAlto&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.470588235294116), &#x27;min&#x27;: np.float64(22.0), &#x27;max&#x27;: np.float64(46.0)}}, np.str_(&#x27;Pittsburgh&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(32.333333333333336), &#x27;min&#x27;: np.float64(25.0), &#x27;max&#x27;: np.float64(47.0)}}, np.str_(&#x27;Queensland&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.94736842105263), &#x27;min&#x27;: np.float64(20.0), &#x27;max&#x27;: np.float64(34.0)}}, np.str_(&#x27;SaintLouis&#x27;): {np.str_(&#x27;age&#x27;): {&#x27;mean&#x27;: np.float64(25.096774193548388), &#x27;min&#x27;: np.float64(21.0), &#x27;max&#x27;: np.float64(29.0)}}}}</dd></dl></div></li></ul></div></div>



Harmonize
~~~~~~~~~

.. code:: ipython3

    # Harmonizing is also easy:
    reference_batch_effect = {
        "site": "Beijing_Zang",
        "sex": "M",
    }  # Set a pseudo-batch effect. I.e., this means 'pretend that all data was from this site and sex'
    
    model.harmonize(test, reference_batch_effect=reference_batch_effect)  # <- easy
    
    plt.style.use("seaborn-v0_8")
    df = test.to_dataframe()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sns.scatterplot(data=df, x=("X", "age"), y=("Y", feature_to_plot), hue=("batch_effects", "site"), ax=ax[0])
    sns.scatterplot(data=df, x=("X", "age"), y=("Y_harmonized", feature_to_plot), hue=("batch_effects", "site"), ax=ax[1])
    ax[0].title.set_text("Unharmonized")
    ax[1].title.set_text("Harmonized")
    ax[0].legend([], [])
    ax[1].legend([], [])
    ax[0].set_xlabel("Age")
    ax[0].set_ylabel(feature_to_plot)
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel(feature_to_plot)
    plt.tight_layout()
    plt.show()


.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:51 - Harmonizing data on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:51 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:51 - Harmonizing data for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:51 - Harmonizing data for CortexVol.
    Process: 75222 - 2025-11-20 13:13:51 - Harmonizing data for Right-Amygdala.
    


.. image:: 02_BLR_files/02_BLR_27_1.png


Synthesize
~~~~~~~~~~

Our models can synthesize new data that follows the learned
distribution.

Not only the distribution of the response variables given a covariate is
learned, but also the ranges of the covariates *within* each batch
effect. So if we have fitted a model on a number of sites, and subjects
from A have an age between 10 and 20, then the synthesized
pseudo-subjects from site A will also have an age between 10 and 20.

Not only that, but we also sample the batch effects in the frequency of
the batch effects in the original data. So if the train data contained
twice as many subjects from site A as site B, then the synthesized
pseudo-subjects will also have twice as many subjects from site A as
site B.

.. code:: ipython3

    # Generate 10000 synthetic datapoints from scratch
    synthetic_data = model.synthesize(covariate_range_per_batch_effect=True, n_samples=10000)  # <- also easy
    plot_centiles_advanced(
        model,
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=synthetic_data,
        show_other_data=True,
        harmonize_data=True,
        show_legend=True,
    )


.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:53 - Dataset "synthesized" created.
        - 10000 observations
        - 10000 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for CortexVol.
    Process: 75222 - 2025-11-20 13:13:53 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for CortexVol.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for CortexVol.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for Right-Amygdala.
    


.. image:: 02_BLR_files/02_BLR_29_1.png



.. image:: 02_BLR_files/02_BLR_29_2.png



.. image:: 02_BLR_files/02_BLR_29_3.png



.. image:: 02_BLR_files/02_BLR_29_4.png


.. code:: ipython3

    # Synthesize new Y data for existing X data
    new_test_data = test.copy()
    
    # Remove the Y data, this way we will synthesize new Y data for the existing X data
    if hasattr(new_test_data, "Y"):
        del new_test_data["Y"]
    
    synthetic = model.synthesize(new_test_data)  # <- will fill in the missing Y data
    plot_centiles_advanced(
        model,
        centiles=[0.05, 0.5, 0.95],  # Plot arbitrary centiles
        covariate="age",  # Which covariate to plot on the x-axis
        scatter_data=train,  # Scatter the train data points
        batch_effects="all",  # You can set this to "all" to show all batch effects
        show_other_data=True,  # Show data points that do not match any batch effects
        harmonize_data=True,  # Set this to False to see the difference
        show_legend=False,  # Don't show the legend because it crowds the plot
    )


.. parsed-literal::

    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:53 - Synthesizing data for CortexVol.
    Process: 75222 - 2025-11-20 13:13:53 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 4 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for 4 response variables.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for CortexVol.
    Process: 75222 - 2025-11-20 13:13:53 - Computing centiles for Right-Amygdala.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data on 4 response variables.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for Right-Lateral-Ventricle.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for WM-hypointensities.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for CortexVol.
    Process: 75222 - 2025-11-20 13:13:53 - Harmonizing data for Right-Amygdala.
    


.. image:: 02_BLR_files/02_BLR_30_1.png



.. image:: 02_BLR_files/02_BLR_30_2.png



.. image:: 02_BLR_files/02_BLR_30_3.png



.. image:: 02_BLR_files/02_BLR_30_4.png


Next steps
----------

Please see the other tutorials for more examples, and we also recommend
you to read the documentation! As this toolkit is still in development,
the documentation may not be up to date. If you find any issues, please
let us know!

Also, feel free to contact us on Github if you have any questions or
suggestions.

Have fun modeling!
