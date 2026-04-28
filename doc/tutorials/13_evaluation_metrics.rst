Evaluation metrics
==================

After fitting a normative model and running predictions, PCNtoolkit
computes a set of evaluation metrics. This notebook explains what each
metric measures, how it is computed, and how to interpret the values.

--------------

Two families of metrics
-----------------------

+------------------------+-----------------+---------------------------+
| Family                 | Uses            | Metrics                   |
+========================+=================+===========================+
| **Point prediction**   | Only ``Yhat``   | MAPE, RMSE, SMSE, R²,     |
|                        | (median         | EXPV, Rho                 |
|                        | prediction)     |                           |
+------------------------+-----------------+---------------------------+
| **Probabilistic**      | Full predicted  | MACE, MSLL, MLL, ShapiroW |
|                        | distribution    |                           |
|                        | (``logp``,      |                           |
|                        | centiles,       |                           |
|                        | Z-scores)       |                           |
+------------------------+-----------------+---------------------------+

**Point prediction metrics** only look at whether the model’s best guess
(the median prediction) is close to the true value, like checking if a
weather forecast said “18°C” when it was actually 20°C.

**Probabilistic metrics** check whether the model’s uncertainty
estimates are accurate. For example, did it say “I’m 90% sure the value
falls between 15°C and 25°C” and was it actually right 90% of the time?
This is important for normative models as we estimate uncertainty (with
centiles, z-scores).

Setup
-----

You can access the evaluation metrics with the code below:

.. code:: python

    # fit and predict with a normative model
    model.predict(test_data)
    
    # Access the statistics DataArray
    test_data['statistics']  # shape: (n_response_vars, n_metrics)
    
    # As a readable DataFrame
    test_all.get_statistics_df()

Point prediction metrics
------------------------

These metrics all compare ``Y`` (true values) against ``Yhat`` (the
model’s median prediction). They tell you how well the model tracks the
central tendency of the data, but they say nothing about the quality of
the uncertainty estimates.

--------------

R² — Coefficient of determination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}

R² answers the question: *how much better is my model than simply always
predicting the* **mean**\ *?*

Unlike EXPV, R² is penalized by systematic mean shifts.

- ≤1 — higher is better
- 0 = no better than mean
- negative = worse than mean

--------------

EXPV — Explained variance
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \text{EXPV} = 1 - \frac{\text{Var}(y - \hat{y} - \overline{(y - \hat{y})})}{\text{Var}(y)}

Similar to R², but it measures how much of the **variance** in the true
values is explained by the model, after removing any systematic mean
offset from the residuals.

- Range: 0 to 1 — higher is better
- A score of 1 means the model perfectly explains the variance in the
  data
- A score of 0 means the model explains no more variance than simply
  predicting the mean

--------------

RMSE — Root mean squared error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \text{RMSE} = \sqrt{\frac{1}{n} \sum_i (y_i - \hat{y}_i)^2}

The average magnitude of prediction error, in the same units as the
response variable. Larger errors are penalized more than small ones due
to the squaring.

- Range: 0 to ∞ — lower is better

--------------

SMSE — Standardized mean squared error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \text{SMSE} = \frac{\text{MSE}}{\text{Var}(y)} = \frac{\frac{1}{n}\sum_i(y_i - \hat{y}_i)^2}{\text{Var}(y)}

SMSE normalizes the MSE by the variance of the target, making it
scale-independent and comparable across different response variables.

- SMSE = 1 means your model does no better than always predicting the
  mean
- SMSE < 1 means improvement over the mean predictor
- SMSE is directly related to R²: SMSE ≈ 1 − R²

--------------

MAPE — Mean absolute percentage error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \text{MAPE} = \frac{1}{n} \sum_i \frac{|y_i - \hat{y}_i|}{|y_i|}

The average absolute percentage error between predictions and true
values. Scale-independent, making it interpretable without knowing the
units of the response variable.

- Range: 0 to ∞
- lower is better

..

   ⚠️ **Note:** MAPE is undefined when any true value :math:`y_i = 0`,
   and becomes unstable when values are close to zero. Use with caution
   for response variables that can be near zero.

--------------

Rho — Spearman rank correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \rho = \text{Pearson}(\text{rank}(y),\ \text{rank}(\hat{y}))

The rank-based correlation between true values and predictions. It
checks if the model correctly orders subjects (e.g., if subject A truly
has more WM-hypointensities than subject B, does the model also predict
a higher value for A than for B?). Unlike Pearson correlation,
Spearman’s ρ is **robust to outliers** and does not assume a linear
relationship.

- Range: −1 to 1
- higher is better
- ``Rho_p`` is the associated p-value testing whether ρ is significantly
  different from zero

Probabilistic metrics
---------------------

--------------

MLL — Mean log loss
~~~~~~~~~~~~~~~~~~~

.. math:: \text{MLL} = -\frac{1}{n} \sum_i \log p(y \mid \mathcal{D}, x_*)

Note: In earlier PCNtoolkit releases, this metric was called ``NLL``
(Negative Log Likelihood). It is now named ``MLL`` to match the
literature and avoid confusion with the different ``NLL`` used
internally for BLR hyperparameter estimation.

Where: - :math:`y`: the test or training response variable. We typically
select the test set here, to see how well the normative model fitted on
training data generalises to test set. - :math:`\mathcal{D}`: the
training dataset used to fit the model - :math:`x_*`: the test covariate
- :math:`p(y_i \mid \mathcal{D}, x_*)`: the probability the model
assigns to the true value given the test input

Measures how “surprised” the model is by the data y, on average.

- Range: 0 to ∞
- lower is better

..

   ⚠️ **Important:** MLL is an **absolute** quantity that is
   scale-dependent (it depends on the units and variance of the response
   variable). This makes it difficult to interpret in isolation. To
   compare models meaningfully, use **MSLL** instead, which normalizes
   MLL against a baseline.

This metric is adopted from `Section 2.5 of Gaussian Processes for
Machine Learning book by C. E. Rasmussen & C. K. I.
Williams <https://gaussianprocess.org/gpml/chapters/RW.pdf#page=27>`__.

--------------

MSLL — Mean standardized log loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \text{MSLL} = \underbrace{-\frac{1}{n}\sum_i \log p(y \mid \mathcal{D}, x_*)}_{\text{MLL}_{\text{model}}} - \underbrace{\left(-\frac{1}{n}\sum_i \log \mathcal{N}\!\left(y \mid \bar{y},\, \hat{\sigma}^2\right)\right)}_{\text{MLL}_{\text{Gaussian baseline}}}

where the Gaussian baseline fits a single normal distribution to the
training responses: - :math:`\bar{y} = \frac{1}{n}\sum_i y_i` — training
sample mean -
:math:`\hat{\sigma}^2 = \frac{1}{n}\sum_i (y_i - \bar{y})^2` — training
sample variance

MSLL is a relative metric. It compares the model’s mean log loss against
a Gaussian baseline. The “standardized” in the name refers to this
subtraction.

======== ============================================
Value    Meaning
======== ============================================
MSLL < 0 Model beats the Gaussian baseline
MSLL = 0 Model is equivalent to the Gaussian baseline
MSLL > 0 Model is worse than the Gaussian baseline
======== ============================================

This metric is adopted from `Section 2.5 of Gaussian Processes for
Machine Learning book by C. E. Rasmussen & C. K. I.
Williams <https://gaussianprocess.org/gpml/chapters/RW.pdf#page=27>`__.

--------------

MACE — Mean absolute centile error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \text{MACE} = \frac{1}{b} \sum_{k=1}^{b} \left( \frac{1}{m} \sum_{j=1}^{m} \left| q_j - \frac{\sum_{i=1}^{n} \mathbf{1}\{\hat{q}_{ij} \geq y_i\}}{n} \right| \right)

where: - :math:`b` are the unique combinations of batch effects -
:math:`m` is the number of centiles used for calibration - :math:`q_j`
is the :math:`j`-th target centile level (e.g. 0.05, 0.25, 0.50, 0.75,
0.95) - :math:`\hat{q}_{ij}` is the predicted :math:`j`-th centile value
for the :math:`i`-th subject - :math:`y_i` is the true value for the
:math:`i`-th subject - :math:`n` is the number of subjects in the batch
group - :math:`\mathbf{1}\{\hat{q}_{ij} \geq y_i\}` is an indicator
function that outputs 1 or 0, depending on whether :math:`y_i` lies
below or above its predicted :math:`j`-th centile value, respectively.
So, :math:`\frac{\sum_i \mathbf{1}\{\hat{q}_{ij} \geq y_i\}}{n}` is the
empirical fraction of subjects below the :math:`j`-th centile curve

The maths above might seem complicated. To put simply, the MACE checks,
for each predicted centile level (e.g. the 10th, 25th, 50th, 75th, 95th
centile curve), what fraction of subjects actually falls below it in the
data. A perfectly calibrated model has exactly 10% of subjects below its
10th centile, 25% below its 25th centile, and so on. MACE averages the
absolute deviation from this perfectly calibrated model across all
centile levels.

Important: MACE is averaged across unique combinations of batch effects
(e.g., site and sex combinations) and each combination contributes
equally. This means small groups have the same influence as large
groups, and hence they may add disproportionate amount of noise to MACE.

- MACE values close to 0 indicate the predicted centile curves closely
  match the distribution of the data.

**Connection to the QQ plot:** The QQ plot is the “uncompressed” version
of MACE. Each point on the QQ plot corresponds to MACE at a specific
quantile level. Systematic deviations from the diagonal (e.g. an S-curve
or U-curve) indicate where along the distribution calibration breaks
down - information that MACE collapses into a single number.

This metric is adopted from *equation 4* of this paper: > Zamanzadeh,
M., Verduyn, Y., de Boer, A. et al. Normative modeling of MEG brain
oscillations across the human lifespan. Commun Biol (2026).
https://doi.org/10.1038/s42003-026-09825-2

--------------

ShapiroW — Shapiro–Wilk W statistic on Z-scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: W = \frac{\left(\sum_i a_i Z_{(i)}\right)^2}{\sum_i (Z_i - \bar{Z})^2}

where :math:`Z_{(i)}` are the Z-scores sorted from smallest to largest
and :math:`a_i` are fixed weights that reflect how much each sorted
Z-score should contribute, based on what a perfect normal distribution
would look like.

The Z-score for each subject is:

.. math:: Z_i = \frac{y_i - \hat{\mu}_i}{\hat{\sigma}_i}

where :math:`\hat{\mu}_i` is the model’s predicted mean and
:math:`\hat{\sigma}_i` is its predicted uncertainty for subject
:math:`i`.

For a normative model, if the model is perfectly calibrated, the
Z-scores should follow a standard normal distribution
:math:`\mathcal{N}(0, 1)`, regardless of whether the original data was
Gaussian or not. A W close to 1 means the model successfully normalized
the non-Gaussian original data into approximately standard-normal
Z-scores.

- Range: 0 to 1 — closer to 1 is better

+------------------------+---------------------------------------------+
| W value                | Interpretation                              |
+========================+=============================================+
| W ≈ 1.0                | Z-scores are well-normalized; model         |
|                        | uncertainty is well-calibrated              |
+------------------------+---------------------------------------------+
| W ≈ 0.95               | Mild departure from normality, likely       |
|                        | slight miscalibration in the tails          |
+------------------------+---------------------------------------------+
| W ≪ 1.0                | Z-scores are substantially non-normal;      |
|                        | model may be missing distributional         |
|                        | structure                                   |
+------------------------+---------------------------------------------+

You can read more about the Shapiro–Wilk test in
`this <https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test>`__
wikipedia page.

Summary table
-------------

+------------+-----------------+---------------+--------------+-----------+
| Metric     | Family          | Input         | Better when  | Range     |
+============+=================+===============+==============+===========+
| R²         | Point           | Y, Yhat       | Higher       | ≤ 1       |
+------------+-----------------+---------------+--------------+-----------+
| EXPV       | Point           | Y, Yhat       | Higher       | 0-1       |
+------------+-----------------+---------------+--------------+-----------+
| Rho        | Point           | Y, Yhat       | Higher       | −1 to 1   |
+------------+-----------------+---------------+--------------+-----------+
| RMSE       | Point           | Y, Yhat       | Lower        | ≥ 0       |
+------------+-----------------+---------------+--------------+-----------+
| SMSE       | Point           | Y, Yhat       | Lower        | ≥ 0       |
+------------+-----------------+---------------+--------------+-----------+
| MAPE       | Point           | Y, Yhat       | Lower        | ≥ 0       |
+------------+-----------------+---------------+--------------+-----------+
| MLL        | Probabilistic   | logp          | Lower        | ≥ 0       |
+------------+-----------------+---------------+--------------+-----------+
| MSLL       | Probabilistic   | logp,         | Lower        | unbounded |
|            |                 | baseline_logp | (negative)   |           |
+------------+-----------------+---------------+--------------+-----------+
| MACE       | Probabilistic   | centiles, Y   | Lower        | 0-1       |
+------------+-----------------+---------------+--------------+-----------+
| ShapiroW   | Probabilistic   | Z-scores      | Higher       | 0–1       |
+------------+-----------------+---------------+--------------+-----------+
