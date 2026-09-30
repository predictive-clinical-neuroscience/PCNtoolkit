Compare normative models
========================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/07_model_comparison.ipynb>`

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


.. code:: text

    Process: 2844 - 2026-09-25 17:54:35 - Removed 0 NANs
    Process: 2844 - 2026-09-25 17:54:35 - Dataset "fcon1000" created.
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


.. code:: text

    Process: 2844 - 2026-09-25 17:54:35 - Fitting models on 2 response variables.
    Process: 2844 - 2026-09-25 17:54:35 - Fitting model for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:00 - Fitting model for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:55:16 - Saving model to:
    	resources/compare_hbr/model1.
    Process: 2844 - 2026-09-25 17:55:16 - Making predictions on 2 response variables.
    Process: 2844 - 2026-09-25 17:55:16 - Computing z-scores for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:16 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:55:17 - Computing z-scores for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:18 - Computing centiles for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:18 - Computing centiles for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:55:19 - Computing centiles for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:21 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:21 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:21 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:55:22 - Computing log-probabilities for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:23 - Computing yhat for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:23 - Making predictions on 2 response variables.
    Process: 2844 - 2026-09-25 17:55:23 - Computing z-scores for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:23 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:55:24 - Computing z-scores for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:24 - Computing centiles for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:24 - Computing centiles for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:55:25 - Computing centiles for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:26 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:26 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:26 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:55:27 - Computing log-probabilities for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:27 - Computing yhat for 2 response variables.
    Process: 2844 - 2026-09-25 17:55:27 - Fitting models on 2 response variables.
    Process: 2844 - 2026-09-25 17:55:27 - Fitting model for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:55:53 - Fitting model for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:08 - Saving model to:
    	resources/compare_hbr/model2.
    Process: 2844 - 2026-09-25 17:56:08 - Making predictions on 2 response variables.
    Process: 2844 - 2026-09-25 17:56:08 - Computing z-scores for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:08 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:09 - Computing z-scores for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:10 - Computing centiles for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:10 - Computing centiles for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:12 - Computing centiles for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:15 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:15 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:15 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:16 - Computing log-probabilities for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:17 - Computing yhat for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:18 - Making predictions on 2 response variables.
    Process: 2844 - 2026-09-25 17:56:18 - Computing z-scores for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:18 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:18 - Computing z-scores for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:19 - Computing centiles for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:19 - Computing centiles for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:20 - Computing centiles for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:22 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:23 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:23 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:23 - Computing log-probabilities for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:23 - Computing yhat for 2 response variables.




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
        Z                  (observations, response_vars) float64 3kB 0.5349 ... 1...
        baseline_logp      (observations, response_vars) float64 3kB -3.66 ... -1...
        logp               (observations, response_vars) float64 3kB -1.699 ... -...
        Yhat               (observations, response_vars) float64 3kB 1.939e+03 .....
        statistics         (response_vars, statistic) float64 208B 0.3669 ... 1.444
        centiles           (centile, observations, response_vars) float64 17kB -4...
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-e6f8c9b1-65b0-4222-9148-a0452c79a87a' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-e6f8c9b1-65b0-4222-9148-a0452c79a87a' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 2</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>statistic</span>: 13</li><li><span class='xr-has-index'>centile</span>: 5</li></ul></div></li><li class='xr-section-item'><input id='section-8f9feb7d-dc05-4919-8693-f7c93dbfef0c' class='xr-section-summary-in' type='checkbox' checked /><label for='section-8f9feb7d-dc05-4919-8693-f7c93dbfef0c' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-7b758993-b5a6-47df-853f-55cdba5b8916' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7b758993-b5a6-47df-853f-55cdba5b8916' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dec7124b-f219-4133-b395-ab6500cf12e6' class='xr-var-data-in' type='checkbox'><label for='data-dec7124b-f219-4133-b395-ab6500cf12e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; &#x27;Right-Late...</div><input id='attrs-7252a9d7-43d2-4377-8c08-766ac211b2d7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7252a9d7-43d2-4377-8c08-766ac211b2d7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b8d0bdd1-be14-4e40-9e18-115e696117ee' class='xr-var-data-in' type='checkbox'><label for='data-b8d0bdd1-be14-4e40-9e18-115e696117ee' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-63aa1978-389c-44c6-8007-65b68786ad09' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-63aa1978-389c-44c6-8007-65b68786ad09' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-72d664cd-183d-43a8-82d9-50a6834353b8' class='xr-var-data-in' type='checkbox'><label for='data-72d664cd-183d-43a8-82d9-50a6834353b8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-ab79af4a-fe69-4231-969a-d1e215603b41' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ab79af4a-fe69-4231-969a-d1e215603b41' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d74af6be-8e2e-4769-9de5-6be9dc0db3b3' class='xr-var-data-in' type='checkbox'><label for='data-d74af6be-8e2e-4769-9de5-6be9dc0db3b3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-07ee323d-8dbf-4b26-bf2e-34b60fbc0703' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-07ee323d-8dbf-4b26-bf2e-34b60fbc0703' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-65c32920-a238-4587-ba0d-a1fd340a90a2' class='xr-var-data-in' type='checkbox'><label for='data-65c32920-a238-4587-ba0d-a1fd340a90a2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-4f4cbf15-ef3a-4fb5-978e-a3cbd39b6ed4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4f4cbf15-ef3a-4fb5-978e-a3cbd39b6ed4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-265542c2-6250-4c2f-846c-8d1f46042844' class='xr-var-data-in' type='checkbox'><label for='data-265542c2-6250-4c2f-846c-8d1f46042844' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8a8cb14e-2643-4971-9820-c34e3a76d292' class='xr-section-summary-in' type='checkbox' checked /><label for='section-8a8cb14e-2643-4971-9820-c34e3a76d292' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-3e07e827-38f4-4194-9e85-c5227fcdb19a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3e07e827-38f4-4194-9e85-c5227fcdb19a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bebcd675-676f-4da2-aac9-5ba697d89741' class='xr-var-data-in' type='checkbox'><label for='data-bebcd675-676f-4da2-aac9-5ba697d89741' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
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
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 1.07e+04</div><input id='attrs-334ad726-fa8c-40bf-bf00-ae4ccb6798b3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-334ad726-fa8c-40bf-bf00-ae4ccb6798b3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ac554da8-e783-4c76-8d98-9ec69b94bac0' class='xr-var-data-in' type='checkbox'><label for='data-ac554da8-e783-4c76-8d98-9ec69b94bac0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 2721.4, 12891.6],
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
           [  703.5, 10700.3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-5864db1c-d436-4dfc-ae0a-3fc384ffb456' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5864db1c-d436-4dfc-ae0a-3fc384ffb456' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1231d91f-2d0e-4b73-9711-3a7575008b72' class='xr-var-data-in' type='checkbox'><label for='data-1231d91f-2d0e-4b73-9711-3a7575008b72' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
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
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-e61511fc-c068-48eb-91f1-a32d1f46e144' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e61511fc-c068-48eb-91f1-a32d1f46e144' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-405c13cf-e628-483f-917c-756710cb7e9a' class='xr-var-data-in' type='checkbox'><label for='data-405c13cf-e628-483f-917c-756710cb7e9a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.5349 0.1876 ... -1.066 1.178</div><input id='attrs-534e02c3-3d58-43af-9a90-8d8d52f7f325' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-534e02c3-3d58-43af-9a90-8d8d52f7f325' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bd5a7534-4b99-4d4b-b14c-0cf5f845a7d8' class='xr-var-data-in' type='checkbox'><label for='data-bd5a7534-4b99-4d4b-b14c-0cf5f845a7d8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.34878090e-01,  1.87574771e-01],
           [ 2.78737045e-02,  9.80043677e-01],
           [ 2.93060007e-01,  2.55158850e-01],
           [ 4.29423458e-02,  1.50703405e+00],
           [-8.71886881e-01, -1.11810971e+00],
           [-8.06962849e-01, -6.43579715e-01],
           [ 5.19800012e-01,  3.58498240e+00],
           [ 6.86218421e-02, -5.84168727e-01],
           [-8.54922621e-01,  6.34087251e-01],
           [-9.49379352e-01, -4.57153625e-01],
           [ 1.12883422e+00, -1.07128109e+00],
           [-8.05782431e-01, -3.34824755e-01],
           [ 1.06572909e+00, -3.58493760e-01],
           [ 1.80808465e+00, -6.84191306e-01],
           [ 1.04960746e+00, -1.18885547e+00],
           [-1.43837101e+00, -7.66187484e-01],
           [-6.39288363e-01,  4.34703434e-01],
           [-4.28162957e-01,  4.96022022e-01],
           [-4.42057193e-01, -2.01848667e-01],
           [ 1.26569907e+00, -3.84547825e-01],
    ...
           [ 2.15422336e-01, -1.44589468e-01],
           [ 2.24062855e+00, -7.81780597e-01],
           [-1.39833253e+00,  9.63039996e-02],
           [-6.44953787e-01,  1.46554866e-01],
           [-9.09025999e-01,  4.11396374e-01],
           [ 2.51678533e-01,  1.04715971e-01],
           [-8.27713790e-01, -3.88906995e-01],
           [ 1.81161648e-01, -5.32060745e-01],
           [ 2.30653739e+00,  1.00470797e+00],
           [-6.86466926e-01,  1.32965464e+00],
           [-3.11738139e-01,  2.77300955e-01],
           [-1.53168909e+00, -1.99602124e-01],
           [-4.06654470e-01,  2.68071788e-01],
           [-3.07499483e-01, -1.42737052e-01],
           [-5.40625252e-01,  8.88032165e-01],
           [-1.56857705e+00, -4.20568302e-01],
           [-1.38903199e+00,  2.14042075e-01],
           [-4.21329899e-01,  2.44439277e-01],
           [ 3.59560078e+00, -2.40911879e-01],
           [-1.06632609e+00,  1.17799552e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-3.66 -2.043 ... -0.9959 -1.366</div><input id='attrs-89dc902e-17c3-4412-a6ba-1bd650374e72' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-89dc902e-17c3-4412-a6ba-1bd650374e72' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d88a3a07-5645-46ba-928b-8e2849398690' class='xr-var-data-in' type='checkbox'><label for='data-d88a3a07-5645-46ba-928b-8e2849398690' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -3.66025491,  -2.04288507],
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
           [ -0.99591923,  -1.36569557]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.699 -1.343 ... -0.7582 -1.385</div><input id='attrs-ceb475e7-9b03-4639-981b-bbc8772c589e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ceb475e7-9b03-4639-981b-bbc8772c589e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e4ac148b-e7a4-4463-80b3-1f0938e8f1a6' class='xr-var-data-in' type='checkbox'><label for='data-e4ac148b-e7a4-4463-80b3-1f0938e8f1a6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -1.69859914,  -1.34314618],
           [ -0.16476941,  -1.15448821],
           [ -0.23485901,  -0.69146551],
           [ -0.69803073,  -2.1617322 ],
           [ -1.90348743,  -1.92949387],
           [ -0.48182159,  -0.86610774],
           [ -0.33819307,  -7.0642029 ],
           [ -0.19744986,  -0.93537106],
           [ -0.53809174,  -0.80236463],
           [ -1.44993822,  -1.20871024],
           [ -0.8296833 ,  -1.150131  ],
           [ -0.5051114 ,  -0.74101843],
           [ -0.75952538,  -0.63916206],
           [ -1.80801637,  -0.97632372],
           [ -0.96169337,  -1.63650587],
           [ -1.22460535,  -0.91027069],
           [ -0.36880608,  -0.72708744],
           [ -0.3083624 ,  -0.67397702],
           [ -0.47913666,  -0.94131743],
           [ -1.05270565,  -0.60372938],
    ...
           [ -0.21528386,  -0.63562472],
           [ -2.70518827,  -0.88134775],
           [ -1.14189929,  -0.63627032],
           [ -0.39668223,  -0.75090191],
           [ -0.61895744,  -0.83706395],
           [ -1.96155359,  -1.52820494],
           [ -0.50343798,  -0.70469229],
           [ -0.23959798,  -0.95178712],
           [ -2.89242714,  -1.32376997],
           [ -0.45252613,  -1.43615023],
           [ -0.26893176,  -0.64132193],
           [ -1.33743875,  -0.65157473],
           [ -0.3277765 ,  -0.61578456],
           [ -0.23079248,  -0.72427441],
           [ -0.31935683,  -0.99680396],
           [ -1.38697591,  -0.77551326],
           [ -1.13170677,  -0.68333408],
           [ -1.98689823,  -1.53803712],
           [ -6.62631209,  -0.68758391],
           [ -0.75817782,  -1.38517099]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.939e+03 1.186e+04 ... 7.222e+03</div><input id='attrs-fd7432e6-1728-4053-8eeb-ea49075d65b6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fd7432e6-1728-4053-8eeb-ea49075d65b6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0aebe4f6-85e0-4b3e-b3a5-4cccf457e9e3' class='xr-var-data-in' type='checkbox'><label for='data-0aebe4f6-85e0-4b3e-b3a5-4cccf457e9e3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1938.96244993, 11859.99588268],
           [ 1132.48349951,  7003.18050115],
           [  844.189087  ,  6745.13709121],
           [ 1445.70791752,  7970.59935409],
           [ 2032.88115958, 10253.78112712],
           [ 1176.67291469,  6931.030571  ],
           [ 1006.99894989,  7863.22329709],
           [  568.85696859,  6879.47117364],
           [ 1012.00373475,  5517.21033983],
           [ 1265.8071663 ,  7813.79393377],
           [ 1176.87143152,  6624.00489554],
           [  907.89027571,  6289.33128204],
           [ 1014.11751205,  5403.37834033],
           [ 1215.25477667,  5771.7289783 ],
           [ 1326.76588441,  7829.6926832 ],
           [ 1014.2253322 ,  6123.8296703 ],
           [ 1453.52274128,  7473.99675509],
           [ 1018.31980984,  5278.85006319],
           [ 1200.90299767,  7804.09003504],
           [ 1208.21851513,  5010.69253381],
    ...
           [  702.76140359,  6113.16920626],
           [ 1360.44863188,  6490.47654838],
           [ 1290.7688218 ,  6253.37019988],
           [ 1685.64086698,  6248.14684641],
           [ 1164.03486247,  8098.21209217],
           [ 3225.73580961, 15145.29523637],
           [ 1195.52499171,  5486.222342  ],
           [  752.52426269,  8167.28891157],
           [ 1183.78580801,  7262.62737241],
           [ 1018.31980984,  5278.85006319],
           [ 1456.2089206 ,  6226.85361892],
           [ 1290.7688218 ,  6253.37019988],
           [ 1297.65729892,  4881.69878657],
           [  555.066052  ,  6762.78879533],
           [ 1174.75765423,  6737.83689504],
           [ 1017.88665971,  5788.57494715],
           [ 1131.02711517,  6962.797668  ],
           [ 3229.51113557, 15570.6548637 ],
           [ 1360.25011504,  6797.50222384],
           [ 1107.31414466,  7221.72237365]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3669 1.082 ... 0.8928 1.444</div><input id='attrs-28fffdfe-606b-4856-8a8c-aee85b762366' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-28fffdfe-606b-4856-8a8c-aee85b762366' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1377b858-5f50-4452-8f73-ba18bd6e5b0b' class='xr-var-data-in' type='checkbox'><label for='data-1377b858-5f50-4452-8f73-ba18bd6e5b0b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3.66871606e-01,  1.08234027e+00,  1.60835397e-01,
             3.41785246e-01,  7.96292402e-01, -3.22976638e-01,
             3.64016972e-01,  4.82663592e+02,  4.90481487e-01,
             1.78456393e-14,  6.35983028e-01,  9.69566017e-01,
             7.22879206e-01],
           [ 1.95692929e-01,  2.63010451e+00,  1.77431043e-01,
             4.23381634e-01,  1.35204571e+00, -8.25750240e-02,
             1.94926798e-01,  3.50924489e+03,  2.61276489e-01,
             1.02165906e-04,  8.05073202e-01,  8.92780197e-01,
             1.44380979e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-487.7 2.755e+03 ... 1.208e+04</div><input id='attrs-4b3d2f06-9b33-4085-bebc-4ae5f31d1540' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4b3d2f06-9b33-4085-bebc-4ae5f31d1540' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-602bb412-016e-46a4-a90e-a113b7d5ca06' class='xr-var-data-in' type='checkbox'><label for='data-602bb412-016e-46a4-a90e-a113b7d5ca06' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ -487.72428479,  2755.38631825],
            [  509.48186064,  2104.70610906],
            [  217.06385661,  2021.35123707],
            ...,
            [ -260.03816149,  4501.52514775],
            [  736.84801533,  1936.78967725],
            [  483.91204495,  2361.00982706]],
    
           [[  943.87370605,  8126.55375134],
            [  877.01504147,  4994.50889966],
            [  587.02970634,  4808.09838544],
            ...,
            [ 1798.58427673, 11031.64031878],
            [ 1104.61744377,  4804.31526932],
            [  851.68147339,  5228.53541914]],
    
           [[ 1938.96244993, 11859.99588268],
            [ 1132.48349951,  7003.18050115],
            [  844.189087  ,  6745.13709121],
            ...,
            [ 3229.51113557, 15570.6548637 ],
            [ 1360.25011504,  6797.50222384],
            [ 1107.31414466,  7221.72237365]],
    
           [[ 2934.05119381, 15593.43801401],
            [ 1387.95195755,  9011.85210264],
            [ 1101.34846767,  8682.17579698],
            ...,
            [ 4660.43799442, 20109.66940862],
            [ 1615.88278631,  8790.68917835],
            [ 1362.94681593,  9214.90932816]],
    
           [[ 4365.64918465, 20964.60544711],
            [ 1755.48513838, 11901.65489325],
            [ 1471.3143174 , 11468.92294534],
            ...,
            [ 6719.06043264, 26639.78457966],
            [ 1983.65221476, 11658.21477043],
            [ 1730.71624438, 12082.43492024]]], shape=(5, 216, 2))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c8d1aa58-16ec-4e6a-b3ee-fe0c222c0ab2' class='xr-section-summary-in' type='checkbox' checked /><label for='section-c8d1aa58-16ec-4e6a-b3ee-fe0c222c0ab2' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7f31f192fce0&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: 489, &#x27;F&#x27;: 589}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;F&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

.. code:: ipython3

    dct = {"model1": "resources/compare_hbr/model1", "model2": "resources/compare_hbr/model2"}
    comparison = compare_hbr_models(dct)


.. code:: text

    Process: 2844 - 2026-09-25 17:56:25 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (18)
        
    Process: 2844 - 2026-09-25 17:56:25 - Synthesizing data for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:25 - Synthesizing data for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:25 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:25 - Making predictions on 2 response variables.
    Process: 2844 - 2026-09-25 17:56:25 - Computing z-scores for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:25 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:25 - Computing z-scores for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:26 - Computing centiles for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:26 - Computing centiles for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:27 - Computing centiles for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:28 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:28 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:28 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:28 - Computing log-probabilities for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:28 - Computing yhat for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:29 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (20)
        
    Process: 2844 - 2026-09-25 17:56:29 - Synthesizing data for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:29 - Synthesizing data for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:29 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:30 - Making predictions on 2 response variables.
    Process: 2844 - 2026-09-25 17:56:30 - Computing z-scores for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:30 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:30 - Computing z-scores for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:31 - Computing centiles for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:31 - Computing centiles for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:33 - Computing centiles for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:35 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:35 - Computing log-probabilities for 2 response variables.
    Process: 2844 - 2026-09-25 17:56:35 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 2844 - 2026-09-25 17:56:35 - Computing log-probabilities for WM-hypointensities.
    Process: 2844 - 2026-09-25 17:56:35 - Computing yhat for 2 response variables.



.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. code:: text

    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1169: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1169: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(



.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. code:: text

    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1169: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(


.. code:: ipython3

    for k, v in comparison.items():
        print(k)
        display(v)


.. code:: text

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
          <th>model2</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td>2 k̂ &gt; 0.70</td>
          <td>11.0</td>
          <td>-150.0</td>
          <td>12.0</td>
          <td>0.61</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-0.0</td>
          <td>15.0</td>
          <td>0.62</td>
          <td>N &lt; 100</td>
          <td>3 k̂ &gt; 0.70</td>
          <td>7.0</td>
          <td>-160.0</td>
          <td>9.8</td>
          <td>0.39</td>
        </tr>
      </tbody>
    </table>
    </div>


.. code:: text

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
          <th>model2</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td>1 k̂ &gt; 0.70</td>
          <td>8.3</td>
          <td>-150.0</td>
          <td>8.2</td>
          <td>0.61</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-10.0</td>
          <td>12.0</td>
          <td>0.75</td>
          <td>N &lt; 100</td>
          <td></td>
          <td>3.6</td>
          <td>-150.0</td>
          <td>7.8</td>
          <td>0.39</td>
        </tr>
      </tbody>
    </table>
    </div>

