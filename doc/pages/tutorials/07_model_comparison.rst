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

    Process: 3265 - 2026-05-22 16:15:49 - Removed 0 NANs
    Process: 3265 - 2026-05-22 16:15:49 - Dataset "fcon1000" created.
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
        progressbar=True,
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
        progressbar=True,
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

    Process: 3265 - 2026-05-22 16:15:49 - Fitting models on 2 response variables.
    Process: 3265 - 2026-05-22 16:15:49 - Fitting model for WM-hypointensities.


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/pytensor/link/c/cmodule.py:2986: UserWarning: PyTensor could not link to a BLAS installation. Operations that might benefit from BLAS will be severely degraded.
    This usually happens when PyTensor is installed via pip. We recommend it be installed via conda/mamba/pixi instead.
    Alternatively, you can use an experimental backend such as Numba or JAX that perform their own BLAS optimizations, by setting `pytensor.config.mode == 'NUMBA'` or passing `mode='NUMBA'` when compiling a PyTensor function.
    For more options and details see https://pytensor.readthedocs.io/en/latest/troubleshooting.html#how-do-i-configure-test-my-blas-library
      warnings.warn(



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for 24 seconds</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>127</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>127</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>95</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 3265 - 2026-05-22 16:16:24 - Fitting model for Right-Lateral-Ventricle.



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for 19 seconds</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.12</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.15</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.14</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.13</td>
                        <td>31</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 3265 - 2026-05-22 16:16:47 - Saving model to:
    	resources/compare_hbr/model1.
    Process: 3265 - 2026-05-22 16:16:47 - Making predictions on 2 response variables.
    Process: 3265 - 2026-05-22 16:16:47 - Computing z-scores for 2 response variables.
    Process: 3265 - 2026-05-22 16:16:47 - Computing z-scores for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:16:48 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:16:49 - Computing centiles for 2 response variables.
    Process: 3265 - 2026-05-22 16:16:49 - Computing centiles for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:16:51 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:16:54 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:16:54 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:16:54 - Computing log-probabilities for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:16:55 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:16:56 - Computing yhat for 2 response variables.
    Process: 3265 - 2026-05-22 16:16:57 - Making predictions on 2 response variables.
    Process: 3265 - 2026-05-22 16:16:57 - Computing z-scores for 2 response variables.
    Process: 3265 - 2026-05-22 16:16:57 - Computing z-scores for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:16:57 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:16:57 - Computing centiles for 2 response variables.
    Process: 3265 - 2026-05-22 16:16:57 - Computing centiles for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:16:59 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:17:00 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:17:00 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:17:00 - Computing log-probabilities for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:17:01 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:17:01 - Computing yhat for 2 response variables.
    Process: 3265 - 2026-05-22 16:17:02 - Fitting models on 2 response variables.
    Process: 3265 - 2026-05-22 16:17:02 - Fitting model for WM-hypointensities.



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for 20 seconds</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>63</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.11</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.10</td>
                        <td>63</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 3265 - 2026-05-22 16:17:36 - Fitting model for Right-Lateral-Ventricle.



.. raw:: html

    
    <style>
        :root {
            --column-width-1: 40%; /* Progress column width */
            --column-width-2: 15%; /* Chain column width */
            --column-width-3: 15%; /* Divergences column width */
            --column-width-4: 15%; /* Step Size column width */
            --column-width-5: 15%; /* Gradients/Draw column width */
        }
    
        .nutpie {
            max-width: 800px;
            margin: 10px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            //color: #333;
            //background-color: #fff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            font-size: 14px; /* Smaller font size for a more compact look */
        }
        .nutpie table {
            width: 100%;
            border-collapse: collapse; /* Remove any extra space between borders */
        }
        .nutpie th, .nutpie td {
            padding: 8px 10px; /* Reduce padding to make table more compact */
            text-align: left;
            border-bottom: 1px solid #888;
        }
        .nutpie th {
            //background-color: #f0f0f0;
        }
    
        .nutpie th:nth-child(1) { width: var(--column-width-1); }
        .nutpie th:nth-child(2) { width: var(--column-width-2); }
        .nutpie th:nth-child(3) { width: var(--column-width-3); }
        .nutpie th:nth-child(4) { width: var(--column-width-4); }
        .nutpie th:nth-child(5) { width: var(--column-width-5); }
    
        .nutpie progress {
            width: 100%;
            height: 15px; /* Smaller progress bars */
            border-radius: 5px;
        }
        progress::-webkit-progress-bar {
            background-color: #eee;
            border-radius: 5px;
        }
        progress::-webkit-progress-value {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        progress::-moz-progress-bar {
            background-color: #5cb85c;
            border-radius: 5px;
        }
        .nutpie .progress-cell {
            width: 100%;
        }
    
        .nutpie p strong { font-size: 16px; font-weight: bold; }
    
        @media (prefers-color-scheme: dark) {
            .nutpie {
                //color: #ddd;
                //background-color: #1e1e1e;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            }
            .nutpie table, .nutpie th, .nutpie td {
                border-color: #555;
                color: #ccc;
            }
            .nutpie th {
                background-color: #2a2a2a;
            }
            .nutpie progress::-webkit-progress-bar {
                background-color: #444;
            }
            .nutpie progress::-webkit-progress-value {
                background-color: #3178c6;
            }
            .nutpie progress::-moz-progress-bar {
                background-color: #3178c6;
            }
        }
    </style>




.. raw:: html

    
    <div class="nutpie">
        <p><strong>Sampler Progress</strong></p>
        <p>Total Chains: <span id="total-chains">4</span></p>
        <p>Active Chains: <span id="active-chains">0</span></p>
        <p>
            Finished Chains:
            <span id="active-chains">4</span>
        </p>
        <p>Sampling for 12 seconds</p>
        <p>
            Estimated Time to Completion:
            <span id="eta">now</span>
        </p>
    
        <progress
            id="total-progress-bar"
            max="8000"
            value="8000">
        </progress>
        <table>
            <thead>
                <tr>
                    <th>Progress</th>
                    <th>Draws</th>
                    <th>Divergences</th>
                    <th>Step Size</th>
                    <th>Gradients/Draw</th>
                </tr>
            </thead>
            <tbody id="chain-details">
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.12</td>
                        <td>95</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.14</td>
                        <td>31</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.12</td>
                        <td>159</td>
                    </tr>
    
                    <tr>
                        <td class="progress-cell">
                            <progress
                                max="2000"
                                value="2000">
                            </progress>
                        </td>
                        <td>2000</td>
                        <td>0</td>
                        <td>0.15</td>
                        <td>31</td>
                    </tr>
    
                </tr>
            </tbody>
        </table>
    </div>



.. parsed-literal::

    Process: 3265 - 2026-05-22 16:17:56 - Saving model to:
    	resources/compare_hbr/model2.
    Process: 3265 - 2026-05-22 16:17:57 - Making predictions on 2 response variables.
    Process: 3265 - 2026-05-22 16:17:57 - Computing z-scores for 2 response variables.
    Process: 3265 - 2026-05-22 16:17:57 - Computing z-scores for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:17:57 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:17:58 - Computing centiles for 2 response variables.
    Process: 3265 - 2026-05-22 16:17:58 - Computing centiles for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:02 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:06 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:06 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:06 - Computing log-probabilities for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:07 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:08 - Computing yhat for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:10 - Making predictions on 2 response variables.
    Process: 3265 - 2026-05-22 16:18:10 - Computing z-scores for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:10 - Computing z-scores for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:10 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:11 - Computing centiles for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:11 - Computing centiles for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:14 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:16 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:16 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:16 - Computing log-probabilities for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:17 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:18 - Computing yhat for 2 response variables.




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
        Z                  (observations, response_vars) float64 3kB 0.5254 ... 1...
        baseline_logp      (observations, response_vars) float64 3kB -3.66 ... -1...
        logp               (observations, response_vars) float64 3kB -1.705 ... -...
        Yhat               (observations, response_vars) float64 3kB 1.945e+03 .....
        statistics         (response_vars, statistic) float64 208B 0.3624 ... 1.446
        centiles           (centile, observations, response_vars) float64 17kB -5...
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-96958285-d13a-4e6d-a83d-53a6a1f3bd52' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-96958285-d13a-4e6d-a83d-53a6a1f3bd52' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 2</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>statistic</span>: 13</li><li><span class='xr-has-index'>centile</span>: 5</li></ul></div></li><li class='xr-section-item'><input id='section-eddf98f7-d4d2-4113-b8d0-86195dcf6598' class='xr-section-summary-in' type='checkbox' checked /><label for='section-eddf98f7-d4d2-4113-b8d0-86195dcf6598' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-da308bd4-d4da-4c7e-8273-3555a9d835d7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-da308bd4-d4da-4c7e-8273-3555a9d835d7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-56aa68cc-bb49-47d2-9063-3ecb6ce94da8' class='xr-var-data-in' type='checkbox'><label for='data-56aa68cc-bb49-47d2-9063-3ecb6ce94da8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; &#x27;Right-Late...</div><input id='attrs-7022a73a-f8f7-426e-bbf1-2dcac102519c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7022a73a-f8f7-426e-bbf1-2dcac102519c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-20ca64f4-8350-4f2a-875a-744d6c908c5c' class='xr-var-data-in' type='checkbox'><label for='data-20ca64f4-8350-4f2a-875a-744d6c908c5c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-980b612b-f15d-4e3e-87c0-f2e6022c69f2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-980b612b-f15d-4e3e-87c0-f2e6022c69f2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2ebfa153-ba64-4f1b-995f-11fe33e72083' class='xr-var-data-in' type='checkbox'><label for='data-2ebfa153-ba64-4f1b-995f-11fe33e72083' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-89b04af6-f03e-4835-898e-671320a6302a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-89b04af6-f03e-4835-898e-671320a6302a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ca95d1ce-cb60-4d37-97c5-c731ca594110' class='xr-var-data-in' type='checkbox'><label for='data-ca95d1ce-cb60-4d37-97c5-c731ca594110' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-657bf623-ce38-4cff-8595-83c8e88ae683' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-657bf623-ce38-4cff-8595-83c8e88ae683' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a3c14c01-e454-4a28-9b6e-4342494162b8' class='xr-var-data-in' type='checkbox'><label for='data-a3c14c01-e454-4a28-9b6e-4342494162b8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-802de2fa-e438-456c-a9b9-071d736959b7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-802de2fa-e438-456c-a9b9-071d736959b7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1e2575ba-6858-4980-98aa-be8105d4066b' class='xr-var-data-in' type='checkbox'><label for='data-1e2575ba-6858-4980-98aa-be8105d4066b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bc5ad6d9-38b5-4978-8270-43d2ecf1cd25' class='xr-section-summary-in' type='checkbox' checked /><label for='section-bc5ad6d9-38b5-4978-8270-43d2ecf1cd25' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-c37e4520-4abb-4bb1-8983-8a21c8643743' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c37e4520-4abb-4bb1-8983-8a21c8643743' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-65d0045a-7351-4041-bf49-d7bd103818f1' class='xr-var-data-in' type='checkbox'><label for='data-65d0045a-7351-4041-bf49-d7bd103818f1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
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
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 1.07e+04</div><input id='attrs-e37a938b-5638-4a9b-b67d-1257bc8b8016' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e37a938b-5638-4a9b-b67d-1257bc8b8016' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-be333c6b-1c35-4c63-ae47-2737aa66da3e' class='xr-var-data-in' type='checkbox'><label for='data-be333c6b-1c35-4c63-ae47-2737aa66da3e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 2721.4, 12891.6],
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
           [  703.5, 10700.3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-3b2e5ff2-045e-419c-8d09-203d3d34e65b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3b2e5ff2-045e-419c-8d09-203d3d34e65b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5c52dbe0-8053-4dec-9264-12b8e633e8be' class='xr-var-data-in' type='checkbox'><label for='data-5c52dbe0-8053-4dec-9264-12b8e633e8be' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
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
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-3eab2968-99cd-4262-a9b6-44188b9884e9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3eab2968-99cd-4262-a9b6-44188b9884e9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-de2fb304-fad6-4783-ab8a-b73ed295b55b' class='xr-var-data-in' type='checkbox'><label for='data-de2fb304-fad6-4783-ab8a-b73ed295b55b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.5254 0.1852 ... -1.071 1.176</div><input id='attrs-a1f35beb-a698-454b-8531-35f45df2a081' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a1f35beb-a698-454b-8531-35f45df2a081' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3b972033-d504-4b2c-98b1-9abca66ef4d3' class='xr-var-data-in' type='checkbox'><label for='data-3b972033-d504-4b2c-98b1-9abca66ef4d3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.25377897e-01,  1.85194819e-01],
           [ 3.54589651e-02,  9.83385149e-01],
           [ 2.90003318e-01,  2.56812618e-01],
           [ 4.38144485e-02,  1.50535015e+00],
           [-8.67440642e-01, -1.11460671e+00],
           [-8.00267670e-01, -6.45421640e-01],
           [ 5.17119315e-01,  3.58120718e+00],
           [ 6.47293458e-02, -5.84864073e-01],
           [-8.51356983e-01,  6.31933400e-01],
           [-9.46347926e-01, -4.55460423e-01],
           [ 1.12701215e+00, -1.07474197e+00],
           [-8.13801043e-01, -3.35926827e-01],
           [ 1.06158927e+00, -3.60477732e-01],
           [ 1.82088994e+00, -6.82600449e-01],
           [ 1.05722669e+00, -1.18579261e+00],
           [-1.43417583e+00, -7.57027548e-01],
           [-6.34552223e-01,  4.31348167e-01],
           [-4.27210263e-01,  4.93491990e-01],
           [-4.46009901e-01, -2.06597024e-01],
           [ 1.26601478e+00, -3.86649766e-01],
    ...
           [ 2.03918984e-01, -1.47794242e-01],
           [ 2.23309784e+00, -7.83128218e-01],
           [-1.39412679e+00,  9.50369935e-02],
           [-6.52565783e-01,  1.44749805e-01],
           [-9.11813113e-01,  4.22238840e-01],
           [ 2.03535264e-01,  1.05672863e-01],
           [-8.25506025e-01, -3.87376331e-01],
           [ 1.81868057e-01, -5.34525287e-01],
           [ 2.33262903e+00,  1.00935207e+00],
           [-6.84911145e-01,  1.32761765e+00],
           [-3.04429236e-01,  2.70959291e-01],
           [-1.52711220e+00, -2.00577430e-01],
           [-4.01965093e-01,  2.63088238e-01],
           [-3.14419066e-01, -1.43469354e-01],
           [-5.35625633e-01,  8.83846526e-01],
           [-1.56496410e+00, -4.20709595e-01],
           [-1.38004021e+00,  2.17679722e-01],
           [-4.23746722e-01,  2.36885107e-01],
           [ 3.59533800e+00, -2.40920032e-01],
           [-1.07084620e+00,  1.17562419e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-3.66 -2.043 ... -0.9959 -1.366</div><input id='attrs-ac9a5193-46ae-4394-af1f-3c794d9da1f8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ac9a5193-46ae-4394-af1f-3c794d9da1f8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5d3d3d29-cd49-49b2-964c-4ecaf0bcc324' class='xr-var-data-in' type='checkbox'><label for='data-5d3d3d29-cd49-49b2-964c-4ecaf0bcc324' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -3.66025491,  -2.04288507],
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
           [ -0.99591923,  -1.36569557]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.705 -1.35 ... -0.7634 -1.385</div><input id='attrs-a8ae8074-c16b-48e4-8080-db16da117ae6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a8ae8074-c16b-48e4-8080-db16da117ae6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ab4bd63a-71c4-46e4-902b-701c275d5f2d' class='xr-var-data-in' type='checkbox'><label for='data-ab4bd63a-71c4-46e4-902b-701c275d5f2d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -1.70514125,  -1.35027572],
           [ -0.16573487,  -1.15867585],
           [ -0.2365483 ,  -0.69106547],
           [ -0.70256955,  -2.1607236 ],
           [ -1.91039463,  -1.93092188],
           [ -0.47765656,  -0.86779045],
           [ -0.34002303,  -7.04963852],
           [ -0.19126917,  -0.93523112],
           [ -0.53880344,  -0.8020281 ],
           [ -1.45351442,  -1.20662312],
           [ -0.83103714,  -1.15406574],
           [ -0.5138975 ,  -0.74064954],
           [ -0.75851639,  -0.64045349],
           [ -1.82746953,  -0.97456024],
           [ -0.96709788,  -1.6332771 ],
           [ -1.22209954,  -0.90544018],
           [ -0.3683214 ,  -0.72683184],
           [ -0.31029247,  -0.67224737],
           [ -0.47752042,  -0.94174919],
           [ -1.05214748,  -0.60231921],
    ...
           [ -0.21613863,  -0.63619004],
           [ -2.69168174,  -0.88283984],
           [ -1.1385021 ,  -0.63739233],
           [ -0.39900024,  -0.7510039 ],
           [ -0.61953588,  -0.83942879],
           [ -2.02441944,  -1.58443442],
           [ -0.5044567 ,  -0.70509372],
           [ -0.23179866,  -0.95252437],
           [ -2.9476735 ,  -1.32765183],
           [ -0.45382361,  -1.43293877],
           [ -0.27113525,  -0.64171499],
           [ -1.33290118,  -0.65303005],
           [ -0.32853208,  -0.61593967],
           [ -0.23045577,  -0.72498575],
           [ -0.32035915,  -0.99383325],
           [ -1.38068967,  -0.77579128],
           [ -1.12088404,  -0.6852418 ],
           [ -2.05255868,  -1.58319699],
           [ -6.6265993 ,  -0.6882551 ],
           [ -0.76339124,  -1.38539375]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.945e+03 1.187e+04 ... 7.227e+03</div><input id='attrs-592a824c-0d14-4276-a02c-ab3a35ef7fb6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-592a824c-0d14-4276-a02c-ab3a35ef7fb6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3ac71aff-cf27-4d68-9233-6e37129d2d0c' class='xr-var-data-in' type='checkbox'><label for='data-3ac71aff-cf27-4d68-9233-6e37129d2d0c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1944.92718404, 11871.63871475],
           [ 1129.57392419,  6991.33765955],
           [  845.04397355,  6739.4266303 ],
           [ 1444.98012353,  7970.72552465],
           [ 2039.15937783, 10261.39191196],
           [ 1174.47884474,  6937.91080023],
           [ 1007.29748109,  7863.8576568 ],
           [  570.31166426,  6881.6865611 ],
           [ 1011.79342198,  5521.56679453],
           [ 1268.06339543,  7804.70876337],
           [ 1175.89807312,  6634.83980431],
           [  911.31751105,  6293.61480776],
           [ 1014.13521844,  5409.20455379],
           [ 1213.1281158 ,  5765.40298815],
           [ 1324.20807347,  7816.56783163],
           [ 1014.63803877,  6100.51315731],
           [ 1452.42464494,  7482.21363693],
           [ 1018.27542832,  5286.22597243],
           [ 1202.04539702,  7819.27851382],
           [ 1208.40159706,  5013.23847294],
    ...
           [  706.88328296,  6121.88241468],
           [ 1360.13834877,  6494.90555065],
           [ 1290.66179025,  6256.57838641],
           [ 1687.88308942,  6253.06567566],
           [ 1164.16105954,  8063.83945002],
           [ 3297.85813078, 15091.5273789 ],
           [ 1195.54304477,  5482.83676489],
           [  752.51297254,  8174.96087923],
           [ 1178.90780249,  7250.42272235],
           [ 1018.27542832,  5286.22597243],
           [ 1453.80062472,  6243.90333945],
           [ 1290.66179025,  6256.57838641],
           [ 1296.17797991,  4895.28950757],
           [  557.50963911,  6765.88803118],
           [ 1173.55627666,  6747.20204506],
           [ 1016.085815  ,  5789.53871189],
           [ 1128.40178213,  6951.44896781],
           [ 3291.52862842, 15544.60434151],
           [ 1358.71912039,  6797.97654657],
           [ 1109.41427486,  7226.51676669]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3624 1.119 ... 0.8923 1.446</div><input id='attrs-a42f8685-6f07-4646-ba4d-87467d127fe4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a42f8685-6f07-4646-ba4d-87467d127fe4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-483c1d29-3125-4b47-8dba-692ec27b322e' class='xr-var-data-in' type='checkbox'><label for='data-483c1d29-3125-4b47-8dba-692ec27b322e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3.62388743e-01,  1.11891797e+00,  1.60835397e-01,
             3.42122763e-01,  8.00287734e-01, -3.18981306e-01,
             3.59234091e-01,  4.84475117e+02,  4.89036337e-01,
             2.18426064e-14,  6.40765909e-01,  9.68348730e-01,
             7.38359828e-01],
           [ 1.96396689e-01,  2.63245275e+00,  1.77431043e-01,
             4.23378840e-01,  1.35321431e+00, -8.14064330e-02,
             1.95645772e-01,  3.50767757e+03,  2.62883004e-01,
             9.22375276e-05,  8.04354228e-01,  8.92289937e-01,
             1.44613411e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-507.0 2.723e+03 ... 1.209e+04</div><input id='attrs-b9350d3c-b2a2-4db6-a713-26a946def379' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b9350d3c-b2a2-4db6-a713-26a946def379' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5aa71f20-b03a-42be-8702-43503c556be9' class='xr-var-data-in' type='checkbox'><label for='data-5aa71f20-b03a-42be-8702-43503c556be9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ -507.04946853,  2723.16155057],
            [  506.15309475,  2089.97631683],
            [  216.19093228,  2011.00003083],
            ...,
            [ -433.45288312,  3919.19721494],
            [  734.5891422 ,  1933.86174312],
            [  485.28429667,  2362.40196324]],
    
           [[  939.46803997,  8120.20820687],
            [  873.93357259,  4981.4822332 ],
            [  587.17608563,  4800.48493747],
            ...,
            [ 1764.06028228, 10777.48231907],
            [ 1102.7879748 ,  4803.39446039],
            [  853.48312926,  5231.93468051]],
    
           [[ 1944.92718404, 11871.63871475],
            [ 1129.57392419,  6991.33765955],
            [  845.04397355,  6739.4266303 ],
            ...,
            [ 3291.52862842, 15544.60434151],
            [ 1358.71912039,  6797.97654657],
            [ 1109.41427486,  7226.51676669]],
    
           [[ 2950.38632812, 15623.06922263],
            [ 1385.21427579,  9001.19308589],
            [ 1102.91186147,  8678.36832313],
            ...,
            [ 4818.99697457, 20311.72636394],
            [ 1614.65026599,  8792.55863275],
            [ 1365.34542045,  9221.09885287]],
    
           [[ 4396.90383662, 21020.11587893],
            [ 1752.99475363, 11892.69900227],
            [ 1473.89701482, 11467.85322977],
            ...,
            [ 7016.51013997, 27170.01146807],
            [ 1982.84909858, 11662.09135002],
            [ 1733.54425305, 12090.63157014]]], shape=(5, 216, 2))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-29f7d2cf-c235-45b8-a94f-5a14e596d197' class='xr-section-summary-in' type='checkbox' checked /><label for='section-29f7d2cf-c235-45b8-a94f-5a14e596d197' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7f6972f54680&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: 489, &#x27;F&#x27;: 589}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;F&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

.. code:: ipython3

    dct = {"model1": "resources/compare_hbr/model1", "model2": "resources/compare_hbr/model2"}
    comparison = compare_hbr_models(dct)


.. parsed-literal::

    Process: 3265 - 2026-05-22 16:18:19 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (21)
        
    Process: 3265 - 2026-05-22 16:18:19 - Synthesizing data for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:19 - Synthesizing data for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:20 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:20 - Making predictions on 2 response variables.
    Process: 3265 - 2026-05-22 16:18:20 - Computing z-scores for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:20 - Computing z-scores for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:20 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:21 - Computing centiles for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:21 - Computing centiles for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:22 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:23 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:23 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:23 - Computing log-probabilities for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:24 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:24 - Computing yhat for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:25 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 3265 - 2026-05-22 16:18:25 - Synthesizing data for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:25 - Synthesizing data for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:26 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:26 - Making predictions on 2 response variables.
    Process: 3265 - 2026-05-22 16:18:26 - Computing z-scores for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:26 - Computing z-scores for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:27 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:27 - Computing centiles for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:27 - Computing centiles for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:30 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:33 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:33 - Computing log-probabilities for 2 response variables.
    Process: 3265 - 2026-05-22 16:18:33 - Computing log-probabilities for WM-hypointensities.
    Process: 3265 - 2026-05-22 16:18:34 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3265 - 2026-05-22 16:18:35 - Computing yhat for 2 response variables.



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. parsed-literal::

    /opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
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

    /opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/arviz/stats/stats.py:782: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(


.. code:: ipython3

    for k, v in comparison.items():
        print(k)
        display(v)


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
          <th>elpd_loo</th>
          <th>p_loo</th>
          <th>elpd_diff</th>
          <th>weight</th>
          <th>se</th>
          <th>dse</th>
          <th>warning</th>
          <th>scale</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model2</th>
          <td>0</td>
          <td>-144.900425</td>
          <td>8.252381</td>
          <td>0.000000</td>
          <td>0.538644</td>
          <td>10.247952</td>
          <td>0.000000</td>
          <td>True</td>
          <td>log</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-157.206130</td>
          <td>6.991130</td>
          <td>12.305704</td>
          <td>0.461356</td>
          <td>11.433660</td>
          <td>15.248923</td>
          <td>True</td>
          <td>log</td>
        </tr>
      </tbody>
    </table>
    </div>


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
          <th>elpd_loo</th>
          <th>p_loo</th>
          <th>elpd_diff</th>
          <th>weight</th>
          <th>se</th>
          <th>dse</th>
          <th>warning</th>
          <th>scale</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model1</th>
          <td>0</td>
          <td>-146.825669</td>
          <td>3.835875</td>
          <td>0.00000</td>
          <td>0.578616</td>
          <td>7.063790</td>
          <td>0.000000</td>
          <td>True</td>
          <td>log</td>
        </tr>
        <tr>
          <th>model2</th>
          <td>1</td>
          <td>-166.488959</td>
          <td>16.411194</td>
          <td>19.66329</td>
          <td>0.421384</td>
          <td>12.459351</td>
          <td>15.072901</td>
          <td>True</td>
          <td>log</td>
        </tr>
      </tbody>
    </table>
    </div>

