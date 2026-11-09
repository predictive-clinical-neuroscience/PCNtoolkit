Compare normative models
========================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/07_model_comparison.ipynb>`

.. code:: ipython3

    import logging
    import warnings
    
    import pandas as pd
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


.. code:: text

    /tmp/ipykernel_3031/1627229480.py:28: Pandas4Warning: The 'mode.chained_assignment' option is deprecated. It has had no effect since pandas 3.0, which removed the SettingWithCopyWarning it controlled. Chained assignment now emits a ChainedAssignmentError warning instead; use warnings.filterwarnings with category=ChainedAssignmentError to silence or escalate it. This option will be removed in pandas 4.0.
      pd.options.mode.chained_assignment = None  # default='warn'


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

    Process: 3031 - 2026-10-27 14:41:23 - Removed 0 NANs
    Process: 3031 - 2026-10-27 14:41:23 - Dataset "fcon1000" created.
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
            sigma=make_prior(
                dist_name="Normal",
                dist_params=(1.0, 0.5),
                mapping="softplus",
                mapping_params=(0.0, 2.0),
            ),
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

    Process: 3031 - 2026-10-27 14:41:23 - Fitting models on 2 response variables.
    Process: 3031 - 2026-10-27 14:41:23 - Fitting model for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:41:43 - Fitting model for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:41:55 - Saving model to:
    	resources/compare_hbr/model1.
    Process: 3031 - 2026-10-27 14:41:55 - Making predictions on 2 response variables.
    Process: 3031 - 2026-10-27 14:41:55 - Computing z-scores for 2 response variables.
    Process: 3031 - 2026-10-27 14:41:55 - Computing z-scores for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:41:56 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:41:56 - Computing centiles for 2 response variables.
    Process: 3031 - 2026-10-27 14:41:56 - Computing centiles for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:41:58 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:00 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:00 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:00 - Computing log-probabilities for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:01 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:02 - Computing yhat for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:02 - Making predictions on 2 response variables.
    Process: 3031 - 2026-10-27 14:42:02 - Computing z-scores for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:02 - Computing z-scores for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:03 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:03 - Computing centiles for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:03 - Computing centiles for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:04 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:05 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:05 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:05 - Computing log-probabilities for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:06 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:06 - Computing yhat for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:06 - Fitting models on 2 response variables.
    Process: 3031 - 2026-10-27 14:42:06 - Fitting model for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:26 - Fitting model for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:39 - Saving model to:
    	resources/compare_hbr/model2.
    Process: 3031 - 2026-10-27 14:42:39 - Making predictions on 2 response variables.
    Process: 3031 - 2026-10-27 14:42:39 - Computing z-scores for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:39 - Computing z-scores for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:40 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:41 - Computing centiles for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:41 - Computing centiles for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:43 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:46 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:46 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:46 - Computing log-probabilities for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:47 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:48 - Computing yhat for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:49 - Making predictions on 2 response variables.
    Process: 3031 - 2026-10-27 14:42:49 - Computing z-scores for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:49 - Computing z-scores for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:49 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:50 - Computing centiles for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:50 - Computing centiles for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:52 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:54 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:54 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:54 - Computing log-probabilities for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:54 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:54 - Computing yhat for 2 response variables.




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
        Z                  (observations, response_vars) float64 3kB 0.5234 ... 1...
        baseline_logp      (observations, response_vars) float64 3kB -3.66 ... -1...
        logp               (observations, response_vars) float64 3kB -1.692 ... -...
        Yhat               (observations, response_vars) float64 3kB 1.959e+03 .....
        statistics         (response_vars, statistic) float64 208B 0.3665 ... 1.444
        centiles           (centile, observations, response_vars) float64 17kB -4...
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-9e1ec263-c423-40d7-aae8-59b2911afc08' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-9e1ec263-c423-40d7-aae8-59b2911afc08' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 2</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>statistic</span>: 13</li><li><span class='xr-has-index'>centile</span>: 5</li></ul></div></li><li class='xr-section-item'><input id='section-bc6c125e-0a56-4414-ab1d-1e73ac2a3238' class='xr-section-summary-in' type='checkbox' checked /><label for='section-bc6c125e-0a56-4414-ab1d-1e73ac2a3238' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-993fb056-d5bb-41d6-bf43-36263d87dae1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-993fb056-d5bb-41d6-bf43-36263d87dae1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6e47ce78-bbae-4e0e-af0e-1ded12d1a88a' class='xr-var-data-in' type='checkbox'><label for='data-6e47ce78-bbae-4e0e-af0e-1ded12d1a88a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; &#x27;Right-Late...</div><input id='attrs-b7a470a0-f6b7-4250-accb-c9796dd04db3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b7a470a0-f6b7-4250-accb-c9796dd04db3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f8b60440-0412-41a2-940b-c4d7a96c076e' class='xr-var-data-in' type='checkbox'><label for='data-f8b60440-0412-41a2-940b-c4d7a96c076e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-cf109487-2ce6-4830-8048-6cf46c0d5922' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cf109487-2ce6-4830-8048-6cf46c0d5922' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e3591aa2-55b5-4753-a720-34bfc20c62dc' class='xr-var-data-in' type='checkbox'><label for='data-e3591aa2-55b5-4753-a720-34bfc20c62dc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-de2359de-4d8a-4cde-9c22-360abbdf4e14' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-de2359de-4d8a-4cde-9c22-360abbdf4e14' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6fe65562-d988-414d-a532-16c022c0ac13' class='xr-var-data-in' type='checkbox'><label for='data-6fe65562-d988-414d-a532-16c022c0ac13' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-3b3df224-9ace-4f24-a760-9cb4a4abc0f9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3b3df224-9ace-4f24-a760-9cb4a4abc0f9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3aeebed3-b6f2-423c-b257-44061c71e958' class='xr-var-data-in' type='checkbox'><label for='data-3aeebed3-b6f2-423c-b257-44061c71e958' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-37d7e3bc-a0f6-4be3-8b7b-af5d5022aacb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-37d7e3bc-a0f6-4be3-8b7b-af5d5022aacb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1c95c8ba-bc11-4c84-a59a-1d97b09ac3a6' class='xr-var-data-in' type='checkbox'><label for='data-1c95c8ba-bc11-4c84-a59a-1d97b09ac3a6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1fbb511c-7191-49e9-be8e-6c5c3f39e764' class='xr-section-summary-in' type='checkbox' checked /><label for='section-1fbb511c-7191-49e9-be8e-6c5c3f39e764' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-816bdc33-c6fc-40f2-9c83-e08f4278a371' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-816bdc33-c6fc-40f2-9c83-e08f4278a371' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ed020f21-d114-401a-a63f-d32289304432' class='xr-var-data-in' type='checkbox'><label for='data-ed020f21-d114-401a-a63f-d32289304432' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
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
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 1.07e+04</div><input id='attrs-ea64dab9-1554-4e90-a98d-2c4d181a913a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ea64dab9-1554-4e90-a98d-2c4d181a913a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c89af9dd-3cbb-4a44-91da-ac3edd201031' class='xr-var-data-in' type='checkbox'><label for='data-c89af9dd-3cbb-4a44-91da-ac3edd201031' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 2721.4, 12891.6],
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
           [  703.5, 10700.3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-2cf38527-b70e-4b27-b682-d93b92c57cd2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2cf38527-b70e-4b27-b682-d93b92c57cd2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8b3444c6-ead0-4ef4-9640-b53d34984ceb' class='xr-var-data-in' type='checkbox'><label for='data-8b3444c6-ead0-4ef4-9640-b53d34984ceb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
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
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-be9beeba-3a07-45e6-b829-08a1338df1e7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-be9beeba-3a07-45e6-b829-08a1338df1e7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-83398e82-8877-4f07-8220-b5d04a7e57fc' class='xr-var-data-in' type='checkbox'><label for='data-83398e82-8877-4f07-8220-b5d04a7e57fc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.5234 0.1936 ... -1.066 1.178</div><input id='attrs-edc8e0f2-38da-45ad-80d1-fd29f564d822' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-edc8e0f2-38da-45ad-80d1-fd29f564d822' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-53f9c3ba-8361-4f9c-842a-7229945b12ad' class='xr-var-data-in' type='checkbox'><label for='data-53f9c3ba-8361-4f9c-842a-7229945b12ad' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.23355760e-01,  1.93629315e-01],
           [ 3.13566152e-02,  9.78786159e-01],
           [ 2.92926781e-01,  2.58236201e-01],
           [ 4.30369289e-02,  1.50812350e+00],
           [-8.72741489e-01, -1.12236425e+00],
           [-8.06138716e-01, -6.46518460e-01],
           [ 5.20102164e-01,  3.58601191e+00],
           [ 6.65075585e-02, -5.88591771e-01],
           [-8.55642417e-01,  6.33219652e-01],
           [-9.51465399e-01, -4.58957118e-01],
           [ 1.12781985e+00, -1.07525724e+00],
           [-8.12759798e-01, -3.38341540e-01],
           [ 1.06369917e+00, -3.59817655e-01],
           [ 1.81291215e+00, -6.82129159e-01],
           [ 1.05218063e+00, -1.19059272e+00],
           [-1.43697562e+00, -7.65337917e-01],
           [-6.34997428e-01,  4.30783988e-01],
           [-4.29758619e-01,  4.94565357e-01],
           [-4.47526102e-01, -2.06486913e-01],
           [ 1.26445464e+00, -3.83873502e-01],
    ...
           [ 2.15695028e-01, -1.43222656e-01],
           [ 2.24138192e+00, -7.83355316e-01],
           [-1.39506442e+00,  9.47529197e-02],
           [-6.45790639e-01,  1.47333995e-01],
           [-9.03709541e-01,  4.21245135e-01],
           [ 2.53091922e-01,  9.82028646e-02],
           [-8.25738435e-01, -3.87417108e-01],
           [ 1.80181377e-01, -5.38304255e-01],
           [ 2.31318589e+00,  1.00437330e+00],
           [-6.87868001e-01,  1.32830814e+00],
           [-3.11993099e-01,  2.70052858e-01],
           [-1.52841732e+00, -2.01206407e-01],
           [-4.08154644e-01,  2.62969493e-01],
           [-3.10289380e-01, -1.47628795e-01],
           [-5.40382573e-01,  8.84730925e-01],
           [-1.56894501e+00, -4.20865867e-01],
           [-1.38601894e+00,  2.12535248e-01],
           [-4.31710959e-01,  2.45098293e-01],
           [ 3.59988068e+00, -2.41554023e-01],
           [-1.06637287e+00,  1.17798071e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-3.66 -2.043 ... -0.9959 -1.366</div><input id='attrs-1a41c73e-5dcf-41dc-a170-f36300803634' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1a41c73e-5dcf-41dc-a170-f36300803634' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-843cd2be-8512-4c82-b014-f0aa9d82b3db' class='xr-var-data-in' type='checkbox'><label for='data-843cd2be-8512-4c82-b014-f0aa9d82b3db' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -3.66025491,  -2.04288507],
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
           [ -0.99591923,  -1.36569557]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.692 -1.346 ... -0.7575 -1.387</div><input id='attrs-938baddc-6adc-4004-bab9-0ae3b90110a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-938baddc-6adc-4004-bab9-0ae3b90110a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b3b31d19-576a-4b9d-8f72-cf8f0ea6975f' class='xr-var-data-in' type='checkbox'><label for='data-b3b31d19-576a-4b9d-8f72-cf8f0ea6975f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -1.69246251,  -1.34558609],
           [ -0.16498886,  -1.15212078],
           [ -0.23586344,  -0.69117224],
           [ -0.6971005 ,  -2.16182547],
           [ -1.9012316 ,  -1.93361137],
           [ -0.48101346,  -0.86746931],
           [ -0.33980089,  -7.06618334],
           [ -0.19385493,  -0.93871182],
           [ -0.53914065,  -0.80145396],
           [ -1.45144039,  -1.20753539],
           [ -0.82901737,  -1.153852  ],
           [ -0.50992316,  -0.74026695],
           [ -0.75772664,  -0.63933983],
           [ -1.81613289,  -0.97470768],
           [ -0.96294522,  -1.63811925],
           [ -1.22226318,  -0.9100962 ],
           [ -0.36625453,  -0.7250273 ],
           [ -0.30993849,  -0.67296085],
           [ -0.47976374,  -0.94081314],
           [ -1.05199711,  -0.60352953],
    ...
           [ -0.21585931,  -0.63462293],
           [ -2.70724892,  -0.88256008],
           [ -1.13752468,  -0.63582496],
           [ -0.39720978,  -0.75029802],
           [ -0.6140376 ,  -0.83985711],
           [ -1.96106965,  -1.530203  ],
           [ -0.5020883 ,  -0.70388168],
           [ -0.23546498,  -0.95590381],
           [ -2.9073943 ,  -1.3223473 ],
           [ -0.45442502,  -1.43403626],
           [ -0.26832432,  -0.63896735],
           [ -1.33262563,  -0.6515953 ],
           [ -0.32775835,  -0.61391532],
           [ -0.22868516,  -0.72593937],
           [ -0.31955245,  -0.99337655],
           [ -1.38713918,  -0.77536461],
           [ -1.12767794,  -0.68196486],
           [ -1.99090147,  -1.54008772],
           [ -6.64150136,  -0.68753944],
           [ -0.75749333,  -1.38692925]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.959e+03 1.182e+04 ... 7.222e+03</div><input id='attrs-e508176b-2fa5-4eda-96a9-6007dd8ee650' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e508176b-2fa5-4eda-96a9-6007dd8ee650' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6e9070e2-6965-4a92-a819-00083fbfd933' class='xr-var-data-in' type='checkbox'><label for='data-6e9070e2-6965-4a92-a819-00083fbfd933' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1959.03896134, 11823.18532051],
           [ 1131.18391493,  7007.04345567],
           [  844.23815825,  6735.78065621],
           [ 1445.62441596,  7976.41591871],
           [ 2030.83974088, 10275.02340367],
           [ 1176.2817654 ,  6939.39543654],
           [ 1006.86818154,  7861.71048053],
           [  569.69534881,  6892.84589883],
           [ 1012.31380636,  5519.85082631],
           [ 1267.08581825,  7816.96956828],
           [ 1177.01381663,  6634.44670995],
           [  910.48429281,  6299.97024594],
           [ 1014.619617  ,  5406.68359813],
           [ 1213.98972105,  5764.51712579],
           [ 1325.87754664,  7832.08257727],
           [ 1013.81094444,  6121.22489834],
           [ 1451.91632196,  7485.67550781],
           [ 1019.01999486,  5282.73356518],
           [ 1203.2404411 ,  7820.72573491],
           [ 1208.19805562,  5008.24880122],
    ...
           [  702.62936302,  6108.46363846],
           [ 1359.67235206,  6494.14364779],
           [ 1289.52212233,  6257.91239598],
           [ 1685.71606456,  6246.01902211],
           [ 1161.76772573,  8067.09266608],
           [ 3222.79973474, 15183.95808994],
           [ 1194.73651813,  5481.38105165],
           [  752.96371235,  8187.32360869],
           [ 1182.19897835,  7264.47273615],
           [ 1019.01999486,  5282.73356518],
           [ 1456.39931484,  6247.58333582],
           [ 1289.52212233,  6257.91239598],
           [ 1298.40549307,  4895.87019104],
           [  556.08213852,  6777.17003396],
           [ 1174.70800599,  6747.61393814],
           [ 1017.71797532,  5789.14432308],
           [ 1129.79936896,  6966.99065471],
           [ 3248.94286303, 15562.71410623],
           [ 1358.94030083,  6799.09237437],
           [ 1107.21534922,  7222.32542764]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3665 1.087 ... 0.8928 1.444</div><input id='attrs-b3d1f1ed-21d1-496f-b259-310e8f124f34' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b3d1f1ed-21d1-496f-b259-310e8f124f34' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-98ed93a6-b48e-466e-9e00-391976e14370' class='xr-var-data-in' type='checkbox'><label for='data-98ed93a6-b48e-466e-9e00-391976e14370' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3.66526997e-01,  1.08748099e+00,  1.60835397e-01,
             3.41710376e-01,  7.96290943e-01, -3.22978097e-01,
             3.63680145e-01,  4.82791388e+02,  4.93359877e-01,
             1.18980230e-14,  6.36319855e-01,  9.69384571e-01,
             7.25827842e-01],
           [ 1.96060979e-01,  2.62971790e+00,  1.77431043e-01,
             4.23603067e-01,  1.35193758e+00, -8.26831575e-02,
             1.95347400e-01,  3.50832809e+03,  2.61962444e-01,
             9.78100213e-05,  8.04652600e-01,  8.92788657e-01,
             1.44369350e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-460.5 2.725e+03 ... 1.208e+04</div><input id='attrs-d3d208c7-165a-4a29-904a-36c0778a95b0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d3d208c7-165a-4a29-904a-36c0778a95b0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a81b7137-4527-4f69-906d-c8e82718f743' class='xr-var-data-in' type='checkbox'><label for='data-a81b7137-4527-4f69-906d-c8e82718f743' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ -460.49944199,  2725.09429405],
            [  508.3785299 ,  2109.51035281],
            [  217.10270927,  2012.97062951],
            ...,
            [ -235.235747  ,  4476.15980649],
            [  735.69153317,  1939.33452548],
            [  483.96658156,  2362.56757875]],
    
           [[  966.88146689,  8092.4161851 ],
            [  875.79593287,  4998.75783985],
            [  587.07458734,  4799.14209887],
            ...,
            [ 1820.21831161, 11016.55442513],
            [ 1103.370505  ,  4806.29690382],
            [  851.64555339,  5229.52995709]],
    
           [[ 1959.03896134, 11823.18532051],
            [ 1131.18391493,  7007.04345567],
            [  844.23815825,  6735.78065621],
            ...,
            [ 3248.94286303, 15562.71410623],
            [ 1358.94030083,  6799.09237437],
            [ 1107.21534922,  7222.32542764]],
    
           [[ 2951.19645579, 15553.95445592],
            [ 1386.57189699,  9015.32907148],
            [ 1101.40172915,  8672.41921354],
            ...,
            [ 4677.66741446, 20108.87378734],
            [ 1614.51009666,  8791.88784492],
            [ 1362.78514505,  9215.12089819]],
    
           [[ 4378.57736467, 20921.27634697],
            [ 1753.98929997, 11904.57655853],
            [ 1471.37360722, 11458.5906829 ],
            ...,
            [ 6733.12147307, 26649.26840598],
            [ 1982.18906849, 11658.85022326],
            [ 1730.46411688, 12082.08327653]]], shape=(5, 216, 2))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4ffe003e-c4e7-4ba3-b6dc-1aaa678d2dca' class='xr-section-summary-in' type='checkbox' checked /><label for='section-4ffe003e-c4e7-4ba3-b6dc-1aaa678d2dca' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7f32011e34c0&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: 489, &#x27;F&#x27;: 589}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;F&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

.. code:: ipython3

    dct = {
        "model1": "resources/compare_hbr/model1",
        "model2": "resources/compare_hbr/model2",
    }
    comparison = compare_hbr_models(dct)


.. code:: text

    Process: 3031 - 2026-10-27 14:42:55 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (20)
        
    Process: 3031 - 2026-10-27 14:42:55 - Synthesizing data for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:55 - Synthesizing data for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:56 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:56 - Making predictions on 2 response variables.
    Process: 3031 - 2026-10-27 14:42:56 - Computing z-scores for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:56 - Computing z-scores for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:56 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:56 - Computing centiles for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:56 - Computing centiles for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:58 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:42:59 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:59 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:42:59 - Computing log-probabilities for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:42:59 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:43:00 - Computing yhat for 2 response variables.
    Process: 3031 - 2026-10-27 14:43:00 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (17)
        
    Process: 3031 - 2026-10-27 14:43:00 - Synthesizing data for 2 response variables.
    Process: 3031 - 2026-10-27 14:43:00 - Synthesizing data for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:43:01 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:43:01 - Making predictions on 2 response variables.
    Process: 3031 - 2026-10-27 14:43:01 - Computing z-scores for 2 response variables.
    Process: 3031 - 2026-10-27 14:43:01 - Computing z-scores for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:43:01 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:43:02 - Computing centiles for 2 response variables.
    Process: 3031 - 2026-10-27 14:43:02 - Computing centiles for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:43:04 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:43:06 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:43:06 - Computing log-probabilities for 2 response variables.
    Process: 3031 - 2026-10-27 14:43:06 - Computing log-probabilities for WM-hypointensities.
    Process: 3031 - 2026-10-27 14:43:06 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3031 - 2026-10-27 14:43:07 - Computing yhat for 2 response variables.



.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. code:: text

    /opt/hostedtoolcache/Python/3.13.16/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1169: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/hostedtoolcache/Python/3.13.16/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1169: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
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

    /opt/hostedtoolcache/Python/3.13.16/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1169: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/hostedtoolcache/Python/3.13.16/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1169: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
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
          <td>3 k̂ &gt; 0.70</td>
          <td>16.2</td>
          <td>-150.0</td>
          <td>13.0</td>
          <td>0.53</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-0.0</td>
          <td>18.0</td>
          <td>0.56</td>
          <td>N &lt; 100</td>
          <td>2 k̂ &gt; 0.70</td>
          <td>11.2</td>
          <td>-150.0</td>
          <td>12.0</td>
          <td>0.47</td>
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
          <td>2 k̂ &gt; 0.70</td>
          <td>15.0</td>
          <td>-150.0</td>
          <td>11.0</td>
          <td>0.57</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-10.0</td>
          <td>19.0</td>
          <td>0.76</td>
          <td>N &lt; 100</td>
          <td>3 k̂ &gt; 0.70</td>
          <td>22.3</td>
          <td>-170.0</td>
          <td>18.0</td>
          <td>0.43</td>
        </tr>
      </tbody>
    </table>
    </div>

