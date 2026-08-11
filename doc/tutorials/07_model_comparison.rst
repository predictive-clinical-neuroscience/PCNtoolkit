Compare normative models
========================

.. code:: ipython3

    import logging
    import warnings
    
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import seaborn as sns
    
    import pcntoolkit.util.output
    from pcntoolkit import (
        HBR,
        BsplineBasisFunction,
        NormalLikelihood,
        NormativeModel,
        NormData,
        load_fcon1000,
        make_prior,
    )
    from pcntoolkit.util.model_comparison import compare_hbr_models
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn'
    pcntoolkit.util.output.Output.set_show_messages(True)

.. code:: ipython3

    # Download an example dataset
    norm_data: NormData = load_fcon1000()
    
    # Select only a few features
    features_to_model = [
        "WM-hypointensities",
        "Right-Lateral-Ventricle",
        # "Right-Amygdala",
        # "CortexVol",
    ]
    norm_data = norm_data.sel({"response_vars": features_to_model})
    
    # Split into train and test sets
    train, test = norm_data.train_test_split()


.. parsed-literal::

    Process: 3884 - 2026-08-11 19:25:32 - Removed 0 NANs
    Process: 3884 - 2026-08-11 19:25:32 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 217 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    

.. code:: ipython3

    mu1 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is not random, because we want to compare to a model with random intercept
        intercept=make_prior(
            dist_name="Normal",
            dist_params=(0.0, 2.0),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma1 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        # We use a B-spline basis function to allow for non-linearity in the standard deviation.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        # We use a softplus mapping to ensure that sigma is strictly positive.
        mapping="softplus",
        # We scale the softplus mapping by a factor of 3, to avoid spikes in the resulting density.
        # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
        # This basically provides an affine transformation of the softplus function.
        # a -> horizontal shift
        # b -> scaling
        # c -> vertical shift
        # You can leave c out, and it will default to 0.
        mapping_params=(0.0, 3.0),
    )
    # Set the likelihood with the priors we just created.
    likelihood1 = NormalLikelihood(mu1, sigma1)
    
    template_hbr_1 = HBR(
        name="template",
        # The number of cores to use for sampling.
        cores=16,
        # Whether to show a progress bar during the model fitting.
        progressbar=False,
        # The number of draws to sample from the posterior per chain.
        draws=1500,
        # The number of tuning steps to run.
        tune=500,
        # The number of MCMC chains to run.
        chains=4,
        # The sampler to use for the model.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=likelihood1,
    )
    model1 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_1,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model1",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    mu2 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
        intercept=make_prior(
            random=True,
            # Mu is the mean of the intercept, which is normally distributed with a mean of 0 and a standard deviation of 1.
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
            # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we have to map it to the positive domain.
            sigma=make_prior(dist_name="Normal", dist_params=(1.0, 0.5), mapping="softplus", mapping_params=(0.0, 2.0)),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma2 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        # We use a B-spline basis function to allow for non-linearity in the standard deviation.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        # We use a softplus mapping to ensure that sigma is strictly positive.
        mapping="softplus",
        # We scale the softplus mapping by a factor of 3, to avoid spikes in the resulting density.
        # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
        # This basically provides an affine transformation of the softplus function.
        # a -> horizontal shift
        # b -> scaling
        # c -> vertical shift
        # You can leave c out, and it will default to 0.
        mapping_params=(0.0, 3.0),
    )
    # Set the likelihood with the priors we just created.
    likelihood2 = NormalLikelihood(mu2, sigma2)
    
    template_hbr_2 = HBR(
        name="template",
        # The number of cores to use for sampling.
        cores=16,
        # Whether to show a progress bar during the model fitting.
        progressbar=False,
        # The number of draws to sample from the posterior per chain.
        draws=1500,
        # The number of tuning steps to run.
        tune=500,
        # The number of MCMC chains to run.
        chains=4,
        # The sampler to use for the model.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=likelihood2,
    )
    model2 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_2,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model2",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    model1.fit_predict(train, test)
    model2.fit_predict(train, test)


.. parsed-literal::

    Process: 3884 - 2026-08-11 19:25:32 - Fitting models on 2 response variables.
    Process: 3884 - 2026-08-11 19:25:32 - Fitting model for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:26:00 - Fitting model for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:26:33 - Saving model to:
    	resources/compare_hbr/model1.
    Process: 3884 - 2026-08-11 19:26:33 - Making predictions on 2 response variables.
    Process: 3884 - 2026-08-11 19:26:33 - Computing z-scores for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:33 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:26:34 - Computing z-scores for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:26:35 - Computing centiles for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:35 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:26:38 - Computing centiles for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:26:41 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:41 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:41 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:26:42 - Computing log-probabilities for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:26:42 - Computing yhat for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:44 - Making predictions on 2 response variables.
    Process: 3884 - 2026-08-11 19:26:44 - Computing z-scores for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:44 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:26:45 - Computing z-scores for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:26:45 - Computing centiles for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:45 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:26:47 - Computing centiles for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:26:50 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:50 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:50 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:26:50 - Computing log-probabilities for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:26:51 - Computing yhat for 2 response variables.
    Process: 3884 - 2026-08-11 19:26:52 - Fitting models on 2 response variables.
    Process: 3884 - 2026-08-11 19:26:52 - Fitting model for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:27:30 - Fitting model for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:27:55 - Saving model to:
    	resources/compare_hbr/model2.
    Process: 3884 - 2026-08-11 19:27:55 - Making predictions on 2 response variables.
    Process: 3884 - 2026-08-11 19:27:55 - Computing z-scores for 2 response variables.
    Process: 3884 - 2026-08-11 19:27:55 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:27:56 - Computing z-scores for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:27:56 - Computing centiles for 2 response variables.
    Process: 3884 - 2026-08-11 19:27:56 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:00 - Computing centiles for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:03 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:03 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:03 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:04 - Computing log-probabilities for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:05 - Computing yhat for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:07 - Making predictions on 2 response variables.
    Process: 3884 - 2026-08-11 19:28:07 - Computing z-scores for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:07 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:07 - Computing z-scores for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:08 - Computing centiles for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:08 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:11 - Computing centiles for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:14 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:14 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:14 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:14 - Computing log-probabilities for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:15 - Computing yhat for 2 response variables.
    



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
    <style>/* CSS stylesheet for displaying xarray objects in notebooks */
    
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
      line-height: 1.6;
      padding-bottom: 4px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
    }
    
    .xr-header {
      border-bottom: solid 1px var(--xr-border-color);
      margin-bottom: 4px;
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-obj-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type,
    .xr-group-box-contents > label {
      color: var(--xr-font-color2);
      display: block;
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
      margin-block-start: 0;
      margin-block-end: 0;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item > input,
    .xr-group-box-contents > input,
    .xr-array-wrap > input {
      display: block;
      opacity: 0;
      height: 0;
      margin: 0;
    }
    
    .xr-section-item > input + label,
    .xr-var-item > input + label {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-item > input:enabled + label,
    .xr-var-item > input:enabled + label,
    .xr-array-wrap > input:enabled + label,
    .xr-group-box-contents > input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item > input:focus-visible + label,
    .xr-var-item > input:focus-visible + label,
    .xr-array-wrap > input:focus-visible + label,
    .xr-group-box-contents > input:focus-visible + label {
      outline: auto;
    }
    
    .xr-section-item > input:enabled + label:hover,
    .xr-var-item > input:enabled + label:hover,
    .xr-array-wrap > input:enabled + label:hover,
    .xr-group-box-contents > input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
      white-space: nowrap;
    }
    
    .xr-section-summary > em {
      font-weight: normal;
    }
    
    .xr-span-grid {
      grid-column-end: -1;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.3em;
    }
    
    .xr-group-box-contents > input:checked + label > span {
      display: inline-block;
      padding-left: 0.6em;
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
    .xr-section-inline-details,
    .xr-group-box-contents > label {
      padding-top: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      grid-column: 1 / -1;
      margin-top: 4px;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in ~ .xr-section-details {
      display: none;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-children {
      display: inline-grid;
      grid-template-columns: 100%;
      grid-column: 1 / -1;
      padding-top: 4px;
    }
    
    .xr-group-box {
      display: inline-grid;
      grid-template-columns: 0px 30px auto;
    }
    
    .xr-group-box-vline {
      grid-column-start: 1;
      border-right: 0.2em solid;
      border-color: var(--xr-border-color);
      width: 0px;
    }
    
    .xr-group-box-hline {
      grid-column-start: 2;
      grid-row-start: 1;
      height: 1em;
      width: 26px;
      border-bottom: 0.2em solid;
      border-color: var(--xr-border-color);
    }
    
    .xr-group-box-contents {
      grid-column-start: 3;
      padding-bottom: 4px;
    }
    
    .xr-group-box-contents > label::before {
      content: "📂";
      padding-right: 0.3em;
    }
    
    .xr-group-box-contents > input:checked + label::before {
      content: "📁";
    }
    
    .xr-group-box-contents > input:checked + label {
      padding-bottom: 0px;
    }
    
    .xr-group-box-contents > input:checked ~ .xr-sections {
      display: none;
    }
    
    .xr-group-box-contents > input + label > span {
      display: none;
    }
    
    .xr-group-box-ellipsis {
      font-size: 1.4em;
      font-weight: 900;
      color: var(--xr-font-color2);
      letter-spacing: 0.15em;
      cursor: default;
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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 70kB
    Dimensions:            (observations: 216, response_vars: 2, covariates: 1,
                            batch_effect_dims: 2, statistic: 13, centile: 5)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 184B &#x27;WM-hypointensities&#x27; &#x27;Right-...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * statistic          (statistic) &lt;U8 416B &#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
    Data variables:
        subject_ids        (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 3kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 3kB 0.5342 ... 1...
        baseline_logp      (observations, response_vars) float64 3kB -3.66 ... -1...
        logp               (observations, response_vars) float64 3kB -1.709 ... -...
        Yhat               (observations, response_vars) float64 3kB 1.934e+03 .....
        statistics         (response_vars, statistic) float64 208B 0.361 ... 1.446
        centiles           (centile, observations, response_vars) float64 17kB -5...
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-a49406b6-1c33-4eb7-83a7-2de499c81037' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-a49406b6-1c33-4eb7-83a7-2de499c81037' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 2</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>statistic</span>: 13</li><li><span class='xr-has-index'>centile</span>: 5</li></ul></div></li><li class='xr-section-item'><input id='section-2fad12f7-3f97-4f36-a6e1-00458378d563' class='xr-section-summary-in' type='checkbox' checked /><label for='section-2fad12f7-3f97-4f36-a6e1-00458378d563' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-8ab3769f-1cf8-4cc0-80db-24689c7ea30a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8ab3769f-1cf8-4cc0-80db-24689c7ea30a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-aaf97217-70fd-4912-9e4c-af2506dc8d1d' class='xr-var-data-in' type='checkbox'><label for='data-aaf97217-70fd-4912-9e4c-af2506dc8d1d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; &#x27;Right-Late...</div><input id='attrs-f31a228c-8801-46ab-81b5-7d6b1383b7c4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f31a228c-8801-46ab-81b5-7d6b1383b7c4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-18ffdc24-268f-42de-b26a-8c654f7ea60a' class='xr-var-data-in' type='checkbox'><label for='data-18ffdc24-268f-42de-b26a-8c654f7ea60a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-4fbb03bd-91cb-4285-8123-d3a1f5db4e67' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4fbb03bd-91cb-4285-8123-d3a1f5db4e67' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5b1b1168-40cb-414c-a63b-4aa42aef3c5e' class='xr-var-data-in' type='checkbox'><label for='data-5b1b1168-40cb-414c-a63b-4aa42aef3c5e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-af4af8a4-000e-4223-bf50-ace017ee08f2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-af4af8a4-000e-4223-bf50-ace017ee08f2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3bfb07cc-9097-4d5a-9ce0-327de58634d1' class='xr-var-data-in' type='checkbox'><label for='data-3bfb07cc-9097-4d5a-9ce0-327de58634d1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-6542aeb6-7daf-459c-a599-d657fcc67422' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6542aeb6-7daf-459c-a599-d657fcc67422' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b24d1c47-56cb-426d-9d2f-713576abf45c' class='xr-var-data-in' type='checkbox'><label for='data-b24d1c47-56cb-426d-9d2f-713576abf45c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-be657dce-285e-4560-b5b5-0be7c3a03905' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-be657dce-285e-4560-b5b5-0be7c3a03905' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1e33d1b5-0223-4d1e-98ee-784c0a4b9272' class='xr-var-data-in' type='checkbox'><label for='data-1e33d1b5-0223-4d1e-98ee-784c0a4b9272' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-0a322d1b-ac2f-474f-ae08-efea7349c8b8' class='xr-section-summary-in' type='checkbox' checked /><label for='section-0a322d1b-ac2f-474f-ae08-efea7349c8b8' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-48b620d0-2a64-4055-8e47-7a2311a25c9a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-48b620d0-2a64-4055-8e47-7a2311a25c9a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78bf5777-a381-47d0-81a2-0df381249ec7' class='xr-var-data-in' type='checkbox'><label for='data-78bf5777-a381-47d0-81a2-0df381249ec7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
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
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 1.07e+04</div><input id='attrs-256c979d-f39a-4f25-ab29-603fdf8f1ac9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-256c979d-f39a-4f25-ab29-603fdf8f1ac9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2ae977e2-88d3-4ece-8a0a-73ae9cf2b6f8' class='xr-var-data-in' type='checkbox'><label for='data-2ae977e2-88d3-4ece-8a0a-73ae9cf2b6f8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 2721.4, 12891.6],
           [ 1143.1,  9919.1],
           [  955.8,  7477.3],
           [ 1473.9, 14302.1],
           [  757.8,  4119.3],
           [  871.1,  5030.9],
           [ 1207.3, 17866.4],
           [  595. ,  5007.9],
           [  682.4,  7286.6],
           [  445.1,  5742.9],
           [ 1620. ,  3713.7],
           [  602.8,  5301.2],
           [ 1432.5,  4429.7],
           [ 1908.2,  3578.1],
           [ 1834. ,  3271.9],
           [  459.6,  3985.8],
           [ 1210. ,  8721.3],
           [  845.9,  6593.1],
           [  995.2,  7040.2],
           [ 1734.7,  4014.8],
    ...
           [  785.8,  5709. ],
           [ 2240.1,  4366.6],
           [  758.1,  6529.8],
           [ 1440.5,  6705.3],
           [  818.6,  9383.3],
           [ 3769.9, 15864.4],
           [  880.2,  4370.2],
           [  823.9,  6379. ],
           [ 2113.9, 10722.5],
           [  741.9,  8801.7],
           [ 1333.9,  6980. ],
           [  707.3,  5680.7],
           [ 1134.1,  5592.2],
           [  438.6,  6330. ],
           [  966.3,  9215.5],
           [  424.3,  4511.1],
           [  604.7,  7590.8],
           [ 2343.2, 17192.3],
           [ 2721.7,  6086. ],
           [  703.5, 10700.3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-9b56ad9b-09af-40f1-b5b0-a011565d476f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9b56ad9b-09af-40f1-b5b0-a011565d476f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-951586de-5fb3-43ce-a6b1-b00d58ed2224' class='xr-var-data-in' type='checkbox'><label for='data-951586de-5fb3-43ce-a6b1-b00d58ed2224' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
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
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-93499e68-47df-4f8e-a1a6-f4a27b24fc5d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-93499e68-47df-4f8e-a1a6-f4a27b24fc5d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1ddfff22-c7de-4b53-98f3-68dc27fb06a8' class='xr-var-data-in' type='checkbox'><label for='data-1ddfff22-c7de-4b53-98f3-68dc27fb06a8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.5342 0.1856 ... -1.066 1.178</div><input id='attrs-19d73969-a2c9-4c6d-bc32-06bf8d544e69' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-19d73969-a2c9-4c6d-bc32-06bf8d544e69' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f0bdabc1-50b9-4242-a668-56836d9aaaae' class='xr-var-data-in' type='checkbox'><label for='data-f0bdabc1-50b9-4242-a668-56836d9aaaae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.34220032e-01,  1.85567802e-01],
           [ 3.32295667e-02,  9.82830918e-01],
           [ 2.90704054e-01,  2.56813177e-01],
           [ 3.98473460e-02,  1.50304164e+00],
           [-8.63592518e-01, -1.11181240e+00],
           [-8.03559019e-01, -6.46091543e-01],
           [ 5.17477045e-01,  3.58110329e+00],
           [ 6.46138798e-02, -5.88442814e-01],
           [-8.54362431e-01,  6.33353444e-01],
           [-9.49672785e-01, -4.59398731e-01],
           [ 1.12547912e+00, -1.07506237e+00],
           [-8.15511510e-01, -3.35963767e-01],
           [ 1.06050688e+00, -3.59201058e-01],
           [ 1.82213334e+00, -6.83935652e-01],
           [ 1.05209076e+00, -1.18694164e+00],
           [-1.43959277e+00, -7.57491049e-01],
           [-6.34413601e-01,  4.30390989e-01],
           [-4.29809538e-01,  4.95136899e-01],
           [-4.49107540e-01, -2.07505359e-01],
           [ 1.26845104e+00, -3.86894901e-01],
    ...
           [ 2.09900838e-01, -1.40807699e-01],
           [ 2.23556487e+00, -7.85015966e-01],
           [-1.39401640e+00,  9.52764536e-02],
           [-6.53763853e-01,  1.47289248e-01],
           [-9.20715779e-01,  4.19443117e-01],
           [ 2.01300231e-01,  1.00669047e-01],
           [-8.25464981e-01, -3.88039831e-01],
           [ 1.80767819e-01, -5.39211314e-01],
           [ 2.33039342e+00,  1.00883507e+00],
           [-6.87812805e-01,  1.32950334e+00],
           [-3.04439506e-01,  2.78951998e-01],
           [-1.52710469e+00, -2.00453875e-01],
           [-4.01566375e-01,  2.72834035e-01],
           [-3.14572309e-01, -1.46578829e-01],
           [-5.38882606e-01,  8.83975229e-01],
           [-1.56797563e+00, -4.20142962e-01],
           [-1.38323581e+00,  2.16743497e-01],
           [-4.23532828e-01,  2.33598380e-01],
           [ 3.59778924e+00, -2.42954639e-01],
           [-1.06646692e+00,  1.17827038e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-3.66 -2.043 ... -0.9959 -1.366</div><input id='attrs-18a4c93d-1b7a-4b44-b18c-929fe84c5271' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-18a4c93d-1b7a-4b44-b18c-929fe84c5271' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b582a0de-19c1-45aa-bed7-cb90b27bc861' class='xr-var-data-in' type='checkbox'><label for='data-b582a0de-19c1-45aa-bed7-cb90b27bc861' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -3.66025491,  -2.04288507],
           [ -0.62929369,  -1.20018059],
           [ -0.72099829,  -0.94007561],
           [ -0.70127013,  -2.64484182],
           [ -0.9220753 ,  -1.21898282],
           [ -0.79391552,  -1.07037055],
           [ -0.61989989,  -4.7455716 ],
           [ -1.16758184,  -1.07345203],
           [ -1.02678525,  -0.93617147],
           [ -1.45761657,  -0.99208431],
           [ -0.82816414,  -1.30256853],
           [ -1.15416898,  -1.03674779],
           [ -0.67590815,  -1.16228047],
           [ -1.24932613,  -1.33291168],
           [ -1.11921947,  -1.4058518 ],
           [ -1.42688131,  -1.24530702],
           [ -0.61975138,  -1.02388273],
           [ -0.81939019,  -0.94201722],
           [ -0.69375343,  -0.93464746],
           [ -0.96861762,  -1.23948959],
    ...
           [ -0.88714286,  -0.9950603 ],
           [ -2.01527752,  -1.17329719],
           [ -0.92168968,  -0.94411669],
           [ -0.68044429,  -0.93893952],
           [ -0.8489441 ,  -1.10972487],
           [ -9.4332193 ,  -3.46339473],
           [ -0.78514238,  -1.17266166],
           [ -0.84304737,  -0.95017362],
           [ -1.68860199,  -1.37098214],
           [ -0.94286471,  -1.03277979],
           [ -0.63434716,  -0.93487843],
           [ -0.99048957,  -0.99760222],
           [ -0.63150983,  -1.00588922],
           [ -1.47158077,  -0.95246174],
           [ -0.7133234 ,  -1.08525544],
           [ -1.502708  ,  -1.14845319],
           [ -1.1509269 ,  -0.94352785],
           [ -2.31442671,  -4.28458605],
           [ -3.66147746,  -0.96619251],
           [ -0.99591923,  -1.36569557]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.709 -1.346 ... -0.7591 -1.388</div><input id='attrs-c4e81c81-3fa4-40df-b65b-8b0270eb89fd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c4e81c81-3fa4-40df-b65b-8b0270eb89fd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b24e5f06-9531-4ffb-b543-9a2a33817fba' class='xr-var-data-in' type='checkbox'><label for='data-b24e5f06-9531-4ffb-b543-9a2a33817fba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -1.70861959,  -1.34630835],
           [ -0.16464214,  -1.15679158],
           [ -0.23614856,  -0.6917763 ],
           [ -0.70185133,  -2.15805331],
           [ -1.90446714,  -1.92485187],
           [ -0.47977441,  -0.86752214],
           [ -0.33959299,  -7.05020917],
           [ -0.19219658,  -0.93717071],
           [ -0.54060642,  -0.80236776],
           [ -1.45523559,  -1.20976038],
           [ -0.82849955,  -1.15400348],
           [ -0.51364968,  -0.74024452],
           [ -0.75652995,  -0.6394451 ],
           [ -1.82954235,  -0.9745814 ],
           [ -0.96132547,  -1.63380569],
           [ -1.22880232,  -0.90501305],
           [ -0.36764303,  -0.72569174],
           [ -0.31037107,  -0.67245159],
           [ -0.47832981,  -0.94121623],
           [ -1.05405393,  -0.60176072],
    ...
           [ -0.21665334,  -0.63578382],
           [ -2.69633573,  -0.88387849],
           [ -1.13778397,  -0.63648805],
           [ -0.40028959,  -0.75046001],
           [ -0.62803768,  -0.84130036],
           [ -2.0224025 ,  -1.58342586],
           [ -0.50353464,  -0.70493182],
           [ -0.23303669,  -0.95497417],
           [ -2.94256024,  -1.32614091],
           [ -0.45479499,  -1.43482232],
           [ -0.26801812,  -0.64315431],
           [ -1.33234434,  -0.65203758],
           [ -0.32600016,  -0.61767017],
           [ -0.2313022 ,  -0.72519715],
           [ -0.32125477,  -0.99346399],
           [ -1.38514286,  -0.77467621],
           [ -1.12416275,  -0.68373737],
           [ -2.04925005,  -1.58004694],
           [ -6.63509461,  -0.68809621],
           [ -0.75911322,  -1.38813706]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.934e+03 1.187e+04 ... 7.22e+03</div><input id='attrs-f9367f32-d093-4cfd-87c4-e0383ce1e915' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f9367f32-d093-4cfd-87c4-e0383ce1e915' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-038cfbc4-c13b-47cf-a4bb-e082ea03630b' class='xr-var-data-in' type='checkbox'><label for='data-038cfbc4-c13b-47cf-a4bb-e082ea03630b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1933.90082021, 11867.23307829],
           [ 1130.47872113,  6994.46772657],
           [  844.84698627,  6739.75643462],
           [ 1447.73155028,  7974.54333004],
           [ 2031.88832927, 10234.67606665],
           [ 1175.5673836 ,  6938.90804003],
           [ 1007.34041134,  7867.62287281],
           [  570.38023207,  6890.79454448],
           [ 1012.67215985,  5518.15163645],
           [ 1269.92720422,  7823.04256371],
           [ 1177.00359332,  6634.89450558],
           [  911.76572851,  6293.50439269],
           [ 1015.00980064,  5405.50321333],
           [ 1212.90162877,  5768.12475831],
           [ 1326.98865919,  7822.60873448],
           [ 1016.24665059,  6100.91684757],
           [ 1452.17652413,  7485.77898979],
           [ 1019.12276236,  5282.29224887],
           [ 1203.58817349,  7824.29390988],
           [ 1208.06161548,  5013.72036199],
    ...
           [  704.61814913,  6102.35942004],
           [ 1360.11685123,  6499.53509677],
           [ 1290.18273145,  6256.38769754],
           [ 1688.23196917,  6245.90224038],
           [ 1167.47050874,  8074.86677561],
           [ 3303.95142977, 15119.5099344 ],
           [ 1195.28578537,  5484.3170817 ],
           [  752.95797559,  8188.21663258],
           [ 1180.1092155 ,  7254.89163686],
           [ 1019.12276236,  5282.29224887],
           [ 1453.66067479,  6222.36207314],
           [ 1290.18273145,  6256.38769754],
           [ 1295.77984383,  4869.75981643],
           [  557.5298989 ,  6774.42175789],
           [ 1174.66595253,  6747.5429287 ],
           [ 1016.93803768,  5787.11138054],
           [ 1129.31143855,  6954.41751414],
           [ 3289.63830295, 15569.59197515],
           [ 1358.68064151,  6803.54863122],
           [ 1107.52487055,  7220.23033841]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.361 1.114 0.1608 ... 0.8923 1.446</div><input id='attrs-74f885f0-164d-4a0b-9cd7-132ec6788616' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-74f885f0-164d-4a0b-9cd7-132ec6788616' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-08c1bb05-08d5-493c-bc4f-68e760b05a4b' class='xr-var-data-in' type='checkbox'><label for='data-08c1bb05-08d5-493c-bc4f-68e760b05a4b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3.61033556e-01,  1.11358710e+00,  1.60835397e-01,
             3.42412977e-01,  8.00382477e-01, -3.18886563e-01,
             3.57862695e-01,  4.84993287e+02,  4.88973815e-01,
             2.20339554e-14,  6.42137305e-01,  9.68456177e-01,
             7.37250041e-01],
           [ 1.96415759e-01,  2.62927600e+00,  1.77431043e-01,
             4.23471559e-01,  1.35329982e+00, -8.13209146e-02,
             1.95681868e-01,  3.50759886e+03,  2.62482864e-01,
             9.46221103e-05,  8.04318132e-01,  8.92322967e-01,
             1.44566227e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-514.1 2.739e+03 ... 1.208e+04</div><input id='attrs-479a9017-60d5-4875-8329-a05100ddbc98' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-479a9017-60d5-4875-8329-a05100ddbc98' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b6053318-3aa7-4deb-9123-187ca87efc4f' class='xr-var-data-in' type='checkbox'><label for='data-b6053318-3aa7-4deb-9123-187ca87efc4f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ -514.05792878,  2739.42752765],
            [  507.39846406,  2095.85740492],
            [  216.44776868,  2013.23002377],
            ...,
            [ -428.180898  ,  3951.60572939],
            [  734.91474972,  1941.98977837],
            [  483.75897876,  2358.67148556]],
    
           [[  930.0892603 ,  8124.27918573],
            [  874.97802486,  4985.74038572],
            [  587.16519359,  4801.59393438],
            ...,
            [ 1765.10693856, 10805.51296393],
            [ 1102.89879343,  4810.01463981],
            [  851.74302246,  5226.696347  ]],
    
           [[ 1933.90082021, 11867.23307829],
            [ 1130.47872113,  6994.46772657],
            [  844.84698627,  6739.75643462],
            ...,
            [ 3289.63830295, 15569.59197515],
            [ 1358.68064151,  6803.54863122],
            [ 1107.52487055,  7220.23033841]],
    
           [[ 2937.71238011, 15610.18697085],
            [ 1385.97941739,  9003.19506742],
            [ 1102.52877896,  8677.91893487],
            ...,
            [ 4814.16966735, 20333.67098637],
            [ 1614.4624896 ,  8797.08262263],
            [ 1363.30671864,  9213.76432982]],
    
           [[ 4381.85956919, 20995.03862893],
            [ 1753.55897819, 11893.07804822],
            [ 1473.24620387, 11466.28284547],
            ...,
            [ 7007.4575039 , 27187.57822091],
            [ 1982.4465333 , 11665.10748407],
            [ 1731.29076234, 12081.78919125]]], shape=(5, 216, 2))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-89314ddb-9aea-4266-95ff-019101b4e714' class='xr-section-summary-in' type='checkbox' checked /><label for='section-89314ddb-9aea-4266-95ff-019101b4e714' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x000001C5FF9BB100&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: 489, &#x27;F&#x27;: 589}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;F&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

.. code:: ipython3

    dct = {"model1": "resources/compare_hbr/model1", "model2": "resources/compare_hbr/model2"}
    comparison = compare_hbr_models(dct)


.. parsed-literal::

    Process: 3884 - 2026-08-11 19:28:16 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (20)
        
    Process: 3884 - 2026-08-11 19:28:16 - Synthesizing data for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:16 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:17 - Synthesizing data for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:17 - Making predictions on 2 response variables.
    Process: 3884 - 2026-08-11 19:28:17 - Computing z-scores for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:17 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:17 - Computing z-scores for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:18 - Computing centiles for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:18 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:20 - Computing centiles for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:22 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:22 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:22 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:22 - Computing log-probabilities for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:22 - Computing yhat for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:23 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (19)
        
    Process: 3884 - 2026-08-11 19:28:23 - Synthesizing data for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:23 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:23 - Synthesizing data for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:24 - Making predictions on 2 response variables.
    Process: 3884 - 2026-08-11 19:28:24 - Computing z-scores for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:24 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:25 - Computing z-scores for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:25 - Computing centiles for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:25 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:28 - Computing centiles for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:31 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:31 - Computing log-probabilities for 2 response variables.
    Process: 3884 - 2026-08-11 19:28:31 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3884 - 2026-08-11 19:28:32 - Computing log-probabilities for WM-hypointensities.
    Process: 3884 - 2026-08-11 19:28:32 - Computing yhat for 2 response variables.
    


.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    


.. parsed-literal::

    c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\arviz_stats\loo\helper_loo.py:1146: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\arviz_stats\loo\helper_loo.py:1146: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    


.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    


.. parsed-literal::

    c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\arviz_stats\loo\helper_loo.py:1146: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\arviz_stats\loo\helper_loo.py:1146: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    

.. code:: ipython3

    for k, v in comparison.items():
        print(k)
        display(v)


.. parsed-literal::

    Right-Lateral-Ventricle
    


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
          <th>rank</th>
          <th>elpd_diff</th>
          <th>dse</th>
          <th>p_worse</th>
          <th>diag_diff</th>
          <th>diag_elpd</th>
          <th>p</th>
          <th>elpd</th>
          <th>se</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model1</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td>4 k̂ &gt; 0.70</td>
          <td>15.1</td>
          <td>-160.0</td>
          <td>14.0</td>
          <td>0.54</td>
        </tr>
        <tr>
          <th>model2</th>
          <td>1</td>
          <td>-0.0</td>
          <td>20.0</td>
          <td>0.53</td>
          <td>N &lt; 100</td>
          <td>7 k̂ &gt; 0.70</td>
          <td>22.6</td>
          <td>-160.0</td>
          <td>14.0</td>
          <td>0.46</td>
        </tr>
      </tbody>
    </table>
    </div>


.. parsed-literal::

    WM-hypointensities
    


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
          <th>rank</th>
          <th>elpd_diff</th>
          <th>dse</th>
          <th>p_worse</th>
          <th>diag_diff</th>
          <th>diag_elpd</th>
          <th>p</th>
          <th>elpd</th>
          <th>se</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model1</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td>1 k̂ &gt; 0.70</td>
          <td>4.6</td>
          <td>-150.0</td>
          <td>10.0</td>
          <td>0.64</td>
        </tr>
        <tr>
          <th>model2</th>
          <td>1</td>
          <td>-70.0</td>
          <td>38.0</td>
          <td>0.96</td>
          <td>N &lt; 100</td>
          <td>8 k̂ &gt; 0.70</td>
          <td>62.9</td>
          <td>-210.0</td>
          <td>36.0</td>
          <td>0.36</td>
        </tr>
      </tbody>
    </table>
    </div>

