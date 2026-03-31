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

    Process: 83743 - 2025-11-20 13:24:56 - Removed 0 NANs
    Process: 83743 - 2025-11-20 13:24:57 - Dataset "fcon1000" created.
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

    Process: 83228 - 2025-11-20 13:24:22 - Fitting models on 2 response variables.
    Process: 83228 - 2025-11-20 13:24:22 - Fitting model for WM-hypointensities.
    

.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

.. code:: ipython3

    dct = {"model1": "resources/compare_hbr/model1", "model2": "resources/compare_hbr/model2"}
    comparison = compare_hbr_models(dct)


.. parsed-literal::

    Process: 83743 - 2025-11-20 13:24:59 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (20)
        
    Process: 83743 - 2025-11-20 13:24:59 - Synthesizing data for 2 response variables.
    Process: 83743 - 2025-11-20 13:24:59 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:00 - Synthesizing data for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:00 - Making predictions on 2 response variables.
    Process: 83743 - 2025-11-20 13:25:00 - Computing z-scores for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:00 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:00 - Computing z-scores for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:00 - Computing centiles for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:00 - Computing centiles for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:01 - Computing centiles for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:01 - Computing log-probabilities for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:01 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:02 - Computing log-probabilities for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:02 - Computing yhat for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:02 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (21)
        
    Process: 83743 - 2025-11-20 13:25:02 - Synthesizing data for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:02 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:02 - Synthesizing data for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:03 - Making predictions on 2 response variables.
    Process: 83743 - 2025-11-20 13:25:03 - Computing z-scores for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:03 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:03 - Computing z-scores for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:03 - Computing centiles for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:03 - Computing centiles for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:05 - Computing centiles for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:06 - Computing log-probabilities for 2 response variables.
    Process: 83743 - 2025-11-20 13:25:06 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 83743 - 2025-11-20 13:25:06 - Computing log-probabilities for WM-hypointensities.
    Process: 83743 - 2025-11-20 13:25:07 - Computing yhat for 2 response variables.
    


.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. parsed-literal::

    Output()


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:797: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    


.. parsed-literal::

    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:797: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/anaconda3/envs/ptk/lib/python3.12/site-packages/arviz/stats/stats.py:797: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
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
          <td>-144.187487</td>
          <td>3.818650</td>
          <td>0.000000</td>
          <td>0.67241</td>
          <td>8.123069</td>
          <td>0.000000</td>
          <td>False</td>
          <td>log</td>
        </tr>
        <tr>
          <th>model2</th>
          <td>1</td>
          <td>-170.325989</td>
          <td>20.296006</td>
          <td>26.138503</td>
          <td>0.32759</td>
          <td>11.779183</td>
          <td>15.218179</td>
          <td>True</td>
          <td>log</td>
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
          <td>-146.022852</td>
          <td>12.196037</td>
          <td>0.000000</td>
          <td>0.552229</td>
          <td>10.274826</td>
          <td>0.000000</td>
          <td>True</td>
          <td>log</td>
        </tr>
        <tr>
          <th>model1</th>
          <td>1</td>
          <td>-159.703623</td>
          <td>11.732313</td>
          <td>13.680771</td>
          <td>0.447771</td>
          <td>13.408503</td>
          <td>17.475941</td>
          <td>True</td>
          <td>log</td>
        </tr>
      </tbody>
    </table>
    </div>

