The NormData class
==================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/01_loading_data.ipynb>`

A key component of the PCNtoolkit is the NormData object. It is a
container for the data that will be used to fit the normative model. The
NormData object keeps track of the all the dimensions of your data, the
features and response variables, batch effects, preprocessing steps, and
more.

.. code:: ipython3

    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    from pcntoolkit import NormData

Creating a NormData object
--------------------------

There are currently two easy ways to create a NormData object. 1. Load
from a pandas dataframe 2. Load from numpy arrays

Here are examples of both.

.. code:: ipython3

    # Creating a NormData object from a pandas dataframe
    
    # Download an example dataset:
    data = pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    )
    
    # specify the column names to use
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = ["WM-hypointensities", "Left-Lateral-Ventricle", "Brain-Stem"]
    
    # create a NormData object
    norm_data = NormData.from_dataframe(
        name="fcon1000",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        remove_outliers=True,
        z_threshold=10,
    )
    norm_data.coords


.. code:: text

    Process: 2590 - 2026-10-27 14:33:24 - Removed 2 outliers for WM-hypointensities
    Process: 2590 - 2026-10-27 14:33:24 - Removed 2 outliers
    Process: 2590 - 2026-10-27 14:33:24 - Dataset "fcon1000" created.
        - 1076 observations
        - 1076 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. code:: text

    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/dataio/norm_data.py:464: UserWarning: No grouping was set for outlier removal, so thresholds are computed across all rows and site differences are ignored. Consider remove_outliers_group_by=['site'] (if available) or your site column.
      dataframe = cls.remove_outliers(




.. code:: text

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'



.. code:: ipython3

    # Creating a NormData object from numpy arrays
    
    from pcntoolkit import NormData
    
    # synthesize some data
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10)
    batch_effects = np.random.randint(0, 2, 100)[:, None]
    subject_ids = np.arange(100)
    
    # Create a NormData object
    np_norm_data = NormData.from_ndarrays(
        "fcon1000", X=X, Y=Y, batch_effects=batch_effects, subject_ids=subject_ids
    )
    np_norm_data.coords


.. code:: text

    Process: 2590 - 2026-10-27 14:33:24 - Dataset "fcon1000" created.
        - 100 observations
        - 100 unique subjects
        - 10 covariates
        - 10 response variables
        - 1 batch effects:
        	batch_effect_0 (2)
        




.. code:: text

    Coordinates:
      * observations       (observations) int64 800B 0 1 2 3 4 5 ... 95 96 97 98 99
      * response_vars      (response_vars) <U14 560B 'response_var_0' ... 'respon...
      * covariates         (covariates) <U11 440B 'covariate_0' ... 'covariate_9'
      * batch_effect_dims  (batch_effect_dims) <U14 56B 'batch_effect_0'



As you can see, it is very simple to create a NormData object.

There is an important difference though: the coordinates of the NormData
object that was created with ``from_dataframe`` have the name of the
column in the dataframe, but the ``from_ndarrays`` method creates
coordinates with generic names. This is why the from_dataframe method is
favorable.

Casting back to a pandas dataframe
----------------------------------

The NormData object can be cast back to a pandas dataframe using the
``to_dataframe`` method. This will return a pandas dataframe with a
columnar multi-index.

.. code:: ipython3

    df = norm_data.to_dataframe()
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    
        .dataframe thead tr:last-of-type th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>X</th>
          <th colspan="3" halign="left">Y</th>
          <th colspan="2" halign="left">batch_effects</th>
          <th>subject_ids</th>
        </tr>
        <tr>
          <th></th>
          <th>age</th>
          <th>Brain-Stem</th>
          <th>Left-Lateral-Ventricle</th>
          <th>WM-hypointensities</th>
          <th>sex</th>
          <th>site</th>
          <th>subject_ids</th>
        </tr>
        <tr>
          <th>observations</th>
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
          <th>0</th>
          <td>25.63</td>
          <td>20663.2</td>
          <td>4049.4</td>
          <td>1686.7</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.34</td>
          <td>19954.0</td>
          <td>9312.6</td>
          <td>1371.1</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.20</td>
          <td>21645.2</td>
          <td>8972.6</td>
          <td>1414.8</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>31.39</td>
          <td>20790.6</td>
          <td>6798.6</td>
          <td>1830.6</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>13.58</td>
          <td>17692.6</td>
          <td>6112.5</td>
          <td>1642.4</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
    </div>



Inspecting the NormData
-----------------------

So let’s go over the attributes of the NormData object. Because it is a
subclass of xarray.Dataset, it has all the attributes of a
xarray.Dataset, but it has some additional attributes that are specific
to normative modelling.

The data variables
~~~~~~~~~~~~~~~~~~

The data variables of the NormData object are: - ``X``: The covariates -
``Y``: The response variables - ``batch_effects``: The batch effects -
``subjects``: The subject ids

And all these data variables are xarray.DataArrays, with corresponding
dimensions, stored in the ``data_vars`` attribute of the NormData
object.

.. code:: ipython3

    norm_data.data_vars




.. code:: text

    Data variables:
        subject_ids    (observations) int64 9kB 0 1 2 3 4 ... 1072 1073 1074 1075
        Y              (observations, response_vars) float64 26kB 1.687e+03 ... 1...
        X              (observations, covariates) float64 9kB 25.63 18.34 ... 23.0
        batch_effects  (observations, batch_effect_dims) <U17 146kB '1' ... 'Sain...



The coordinates
~~~~~~~~~~~~~~~

Because it is a subclass of xarray.Dataset, the NormData object also
holds all the coordinates of the data, found under the ``coords``
attribute.

The coordinates are: - ``observations``: The index of the observations -
``response_vars``: The names of the response variables - ``covariates``:
The names of the covariates - ``batch_effect_dims``: The names of the
batch effect dimensions

.. code:: ipython3

    norm_data.coords




.. code:: text

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'



Indexing using the coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarrays powerful indexing methods can also be used on NormData.

Selecting a response variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example, to select the data for a specific response variable, you
can use the ``response_vars`` coordinate:

.. code:: python

   norm_data.sel(response_vars="WM-hypointensities")

This will return a new NormData object with only the data for the
response variable “WM-hypointensities”.

.. code:: ipython3

    norm_data.sel(response_vars="WM-hypointensities")




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 181kB
    Dimensions:            (observations: 1076, covariates: 1, batch_effect_dims: 2)
    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
        response_vars      &lt;U22 88B &#x27;WM-hypointensities&#x27;
    Data variables:
        subject_ids        (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
        Y                  (observations) float64 9kB 1.687e+03 1.371e+03 ... 509.1
        X                  (observations, covariates) float64 9kB 25.63 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 146kB &#x27;1&#x27; ... &#x27;...
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-20943cf8-f8d1-458b-a004-b0c251856b31' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-20943cf8-f8d1-458b-a004-b0c251856b31' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 1076</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div></li><li class='xr-section-item'><input id='section-6d812b98-dccf-457e-a9b5-ba08d0a950bb' class='xr-section-summary-in' type='checkbox' checked /><label for='section-6d812b98-dccf-457e-a9b5-ba08d0a950bb' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1072 1073 1074 1075</div><input id='attrs-3ca7f785-eb14-4b22-b23e-f3af7aa2230e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3ca7f785-eb14-4b22-b23e-f3af7aa2230e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-126395b6-fe90-40fb-ad3c-63e84da7a319' class='xr-var-data-in' type='checkbox'><label for='data-126395b6-fe90-40fb-ad3c-63e84da7a319' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1073, 1074, 1075], shape=(1076,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-db4854a1-8995-445f-af8b-cb7096bacd2f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-db4854a1-8995-445f-af8b-cb7096bacd2f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-52224e1b-fb62-4575-93ed-837f429e1d55' class='xr-var-data-in' type='checkbox'><label for='data-52224e1b-fb62-4575-93ed-837f429e1d55' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-7c8385b5-2681-415a-8e5d-dcf901888139' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7c8385b5-2681-415a-8e5d-dcf901888139' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-44275c8d-b297-4829-b267-ffb515cb7ec9' class='xr-var-data-in' type='checkbox'><label for='data-44275c8d-b297-4829-b267-ffb515cb7ec9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-de0ef316-5387-48ac-8bbd-f91dfb4459a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-de0ef316-5387-48ac-8bbd-f91dfb4459a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ee29ec33-771b-428b-9aaa-9c12e5221a04' class='xr-var-data-in' type='checkbox'><label for='data-ee29ec33-771b-428b-9aaa-9c12e5221a04' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-11192233-95ff-45cc-b6de-492785afc3bb' class='xr-section-summary-in' type='checkbox' checked /><label for='section-11192233-95ff-45cc-b6de-492785afc3bb' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1072 1073 1074 1075</div><input id='attrs-947dca4f-b17f-41b4-b002-b5a5429e443a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-947dca4f-b17f-41b4-b002-b5a5429e443a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-64ee07eb-ab5f-4a58-8df2-8834e2e3bd8b' class='xr-var-data-in' type='checkbox'><label for='data-64ee07eb-ab5f-4a58-8df2-8834e2e3bd8b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1073, 1074, 1075], shape=(1076,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 1.371e+03 ... 448.3 509.1</div><input id='attrs-be88dea4-7756-4c3a-968c-935f29c1c262' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-be88dea4-7756-4c3a-968c-935f29c1c262' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b11e12b2-378b-4333-9106-93062af73f7f' class='xr-var-data-in' type='checkbox'><label for='data-b11e12b2-378b-4333-9106-93062af73f7f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1686.7, 1371.1, 1414.8, ..., 1061. ,  448.3,  509.1], shape=(1076,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 27.0 29.0 23.0</div><input id='attrs-9423116f-ca99-4d72-b3b9-12851e972ab4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9423116f-ca99-4d72-b3b9-12851e972ab4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-83fc0b14-8a17-491f-a07a-19773863419a' class='xr-var-data-in' type='checkbox'><label for='data-83fc0b14-8a17-491f-a07a-19773863419a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           ...,
           [27.  ],
           [29.  ],
           [23.  ]], shape=(1076, 1))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;SaintLouis&#x27;</div><input id='attrs-22216526-b6ad-4f8f-94fc-ed3d9733ad04' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-22216526-b6ad-4f8f-94fc-ed3d9733ad04' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9ec75d96-491a-4ebf-b143-482e01ea8644' class='xr-var-data-in' type='checkbox'><label for='data-9ec75d96-491a-4ebf-b143-482e01ea8644' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           ...,
           [&#x27;1&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;]], shape=(1076, 2), dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-52b8549e-5b8d-4544-8f76-fc3d753e564d' class='xr-section-summary-in' type='checkbox' checked /><label for='section-52b8549e-5b8d-4544-8f76-fc3d753e564d' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7f91452aa3e0&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: 489, &#x27;0&#x27;: 587}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 83, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;0&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



Selecting a number of observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

But we can also filter out a slice of the data. For example, to select
the first 10 observations, you can use the ``observations`` coordinate:

.. code:: python

   norm_data.sel(observations=slice(0, 9))

This will return a new NormData object with only the first 10
observations.

.. code:: ipython3

    norm_data.sel(observations=slice(0, 9))




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 2kB
    Dimensions:            (observations: 10, response_vars: 3, covariates: 1,
                            batch_effect_dims: 2)
    Coordinates:
      * observations       (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
      * response_vars      (response_vars) &lt;U22 264B &#x27;WM-hypointensities&#x27; ... &#x27;Br...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
    Data variables:
        subject_ids        (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
        Y                  (observations, response_vars) float64 240B 1.687e+03 ....
        X                  (observations, covariates) float64 80B 25.63 ... 19.88
        batch_effects      (observations, batch_effect_dims) &lt;U17 1kB &#x27;1&#x27; ... &#x27;An...
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-dbb7e73c-17fd-43dc-a227-bf7ad52dca24' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-dbb7e73c-17fd-43dc-a227-bf7ad52dca24' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div></li><li class='xr-section-item'><input id='section-89c8bdd2-33c2-4248-b066-558bcba5de62' class='xr-section-summary-in' type='checkbox' checked /><label for='section-89c8bdd2-33c2-4248-b066-558bcba5de62' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-9a888d86-80bb-4762-9ae0-a3bd54ac451d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9a888d86-80bb-4762-9ae0-a3bd54ac451d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b0645721-585f-4484-8455-3f208cf9dfec' class='xr-var-data-in' type='checkbox'><label for='data-b0645721-585f-4484-8455-3f208cf9dfec' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-996e26ff-a57e-4fd4-8379-6f0c02d2f159' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-996e26ff-a57e-4fd4-8379-6f0c02d2f159' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ffee42a4-77b4-4064-a85f-6ccef7d3d8e3' class='xr-var-data-in' type='checkbox'><label for='data-ffee42a4-77b4-4064-a85f-6ccef7d3d8e3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-4d8b8c3e-b24d-4c4b-bc34-e736ee259382' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4d8b8c3e-b24d-4c4b-bc34-e736ee259382' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1f9d5721-34b1-4152-97d7-5ca7dbd0f8cf' class='xr-var-data-in' type='checkbox'><label for='data-1f9d5721-34b1-4152-97d7-5ca7dbd0f8cf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-d14d4fdb-7d04-4100-b97f-8076acd925c3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d14d4fdb-7d04-4100-b97f-8076acd925c3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-233a88e3-8426-4e4a-96a1-ac0496bdf7af' class='xr-var-data-in' type='checkbox'><label for='data-233a88e3-8426-4e4a-96a1-ac0496bdf7af' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-eb0126dd-0664-40df-8def-b2914866a3e5' class='xr-section-summary-in' type='checkbox' checked /><label for='section-eb0126dd-0664-40df-8def-b2914866a3e5' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-370510fd-b401-475a-9904-eb3185daf085' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-370510fd-b401-475a-9904-eb3185daf085' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-76351df0-44a7-4e54-a2fd-f99583e9e760' class='xr-var-data-in' type='checkbox'><label for='data-76351df0-44a7-4e54-a2fd-f99583e9e760' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-07c63a8f-8987-4831-8caf-7336b05b87cf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-07c63a8f-8987-4831-8caf-7336b05b87cf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-56e205d9-9c48-43aa-9463-dede15c4c24d' class='xr-var-data-in' type='checkbox'><label for='data-56e205d9-9c48-43aa-9463-dede15c4c24d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-8a356e2b-fbfe-46d9-80c8-61d64b9b4216' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8a356e2b-fbfe-46d9-80c8-61d64b9b4216' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-41aeb00c-8543-4ea8-83be-8315255a23cd' class='xr-var-data-in' type='checkbox'><label for='data-41aeb00c-8543-4ea8-83be-8315255a23cd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-757ef5fb-3d52-4c02-b774-ed292096cdc8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-757ef5fb-3d52-4c02-b774-ed292096cdc8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-11f4e1b6-be99-4aa8-a00b-4ce4220f5c0c' class='xr-var-data-in' type='checkbox'><label for='data-11f4e1b6-be99-4aa8-a00b-4ce4220f5c0c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6635fa2d-a3b1-4b18-910f-4ac61ab17665' class='xr-section-summary-in' type='checkbox' checked /><label for='section-6635fa2d-a3b1-4b18-910f-4ac61ab17665' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7f91452aa3e0&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: 489, &#x27;0&#x27;: 587}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 83, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;0&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



NormData with predictions
-------------------------

After fitting a model and predicting on NormData, the NormData object
will have new attributes holding the predictions.

Specifically, the NormData object will be extended with new data
variables:

-  ``Z``: The predicted Z scores for each response variable
-  ``centiles``: The predicted centiles
-  ``logp``: The predicted log-p-values for each response variable
-  ``Yhat``: The predicted mean of the response variable
-  ``Y_harmonized``: The harmonized response variables
-  ``statistics``: An array of statistics for each response variable

And the following new coordinates: - ``centile``: The specific centile
values - ``statistic``: The name of the computed statistics

.. code:: ipython3

    from pcntoolkit import BLR, NormativeModel
    
    # We create a very simple BLR model because it is fast to fit
    model = NormativeModel(BLR())
    model.fit(norm_data)  # Fitting on the data also makes predictions for that data


.. code:: text

    Process: 2590 - 2026-10-27 14:33:24 - Fitting models on 3 response variables.
    Process: 2590 - 2026-10-27 14:33:24 - Fitting model for WM-hypointensities.
    Process: 2590 - 2026-10-27 14:33:24 - Fitting model for Left-Lateral-Ventricle.
    Process: 2590 - 2026-10-27 14:33:24 - Fitting model for Brain-Stem.
    Process: 2590 - 2026-10-27 14:33:25 - Saving model to:
    	/home/runner/.pcntoolkit/saves.
    Process: 2590 - 2026-10-27 14:33:25 - Making predictions on 3 response variables.
    Process: 2590 - 2026-10-27 14:33:25 - Computing z-scores for 3 response variables.
    Process: 2590 - 2026-10-27 14:33:25 - Computing z-scores for Brain-Stem.
    Process: 2590 - 2026-10-27 14:33:25 - Computing z-scores for Left-Lateral-Ventricle.
    Process: 2590 - 2026-10-27 14:33:25 - Computing z-scores for WM-hypointensities.
    Process: 2590 - 2026-10-27 14:33:25 - Computing centiles for 3 response variables.
    Process: 2590 - 2026-10-27 14:33:25 - Computing centiles for Brain-Stem.
    Process: 2590 - 2026-10-27 14:33:25 - Computing centiles for Left-Lateral-Ventricle.
    Process: 2590 - 2026-10-27 14:33:25 - Computing centiles for WM-hypointensities.
    Process: 2590 - 2026-10-27 14:33:25 - Computing log-probabilities for 3 response variables.
    Process: 2590 - 2026-10-27 14:33:25 - Computing log-probabilities for 3 response variables.
    Process: 2590 - 2026-10-27 14:33:25 - Computing log-probabilities for Brain-Stem.
    Process: 2590 - 2026-10-27 14:33:25 - Computing log-probabilities for Left-Lateral-Ventricle.
    Process: 2590 - 2026-10-27 14:33:25 - Computing log-probabilities for WM-hypointensities.
    Process: 2590 - 2026-10-27 14:33:25 - Computing yhat for 3 response variables.
    Process: 2590 - 2026-10-27 14:33:25 - Computing yhat for Brain-Stem.
    Process: 2590 - 2026-10-27 14:33:25 - Computing yhat for Left-Lateral-Ventricle.
    Process: 2590 - 2026-10-27 14:33:25 - Computing yhat for WM-hypointensities.
    Process: 2590 - 2026-10-27 14:33:26 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 2590 - 2026-10-27 14:33:26 - Computing centiles for 3 response variables.
    Process: 2590 - 2026-10-27 14:33:26 - Computing centiles for Brain-Stem.
    Process: 2590 - 2026-10-27 14:33:26 - Computing centiles for Left-Lateral-Ventricle.
    Process: 2590 - 2026-10-27 14:33:26 - Computing centiles for WM-hypointensities.
    Process: 2590 - 2026-10-27 14:33:26 - Harmonizing data on 3 response variables.
    Process: 2590 - 2026-10-27 14:33:26 - Harmonizing data for Brain-Stem.
    Process: 2590 - 2026-10-27 14:33:26 - Harmonizing data for Left-Lateral-Ventricle.
    Process: 2590 - 2026-10-27 14:33:26 - Harmonizing data for WM-hypointensities.


.. code:: ipython3

    norm_data.data_vars




.. code:: text

    Data variables:
        subject_ids    (observations) int64 9kB 0 1 2 3 4 ... 1072 1073 1074 1075
        Y              (observations, response_vars) float64 26kB 1.687e+03 ... 1...
        X              (observations, covariates) float64 9kB 25.63 18.34 ... 23.0
        batch_effects  (observations, batch_effect_dims) <U17 146kB '1' ... 'Sain...
        Z              (observations, response_vars) float64 26kB 0.7423 ... -0.2588
        centiles       (centile, observations, response_vars) float64 129kB 152.3...
        baseline_logp  (observations, response_vars) float64 26kB -1.131 ... -0.9549
        logp           (observations, response_vars) float64 26kB -1.158 ... -0.9536
        Yhat           (observations, response_vars) float64 26kB 1.21e+03 ... 2....
        statistics     (response_vars, statistic) float64 312B 0.1172 ... 0.1246



.. code:: ipython3

    norm_data.coords




.. code:: text

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) <U8 416B 'EXPV' 'Kurtosis' ... 'Skewness'



Indexing of predicted data
~~~~~~~~~~~~~~~~~~~~~~~~~~

All the indexing methods can still be used, and they will also slice
through the newly added data variables. So for example, to select the
first 10 observations, you can use:

.. code:: python

   norm_data.sel(observations=slice(0, 9))

This will return a new NormData object with only the first 10
observations.

.. code:: ipython3

    norm_data.sel(observations=slice(0, 9))




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 5kB
    Dimensions:            (observations: 10, response_vars: 3, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 13)
    Coordinates:
      * observations       (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
      * response_vars      (response_vars) &lt;U22 264B &#x27;WM-hypointensities&#x27; ... &#x27;Br...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 416B &#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;
    Data variables:
        subject_ids        (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
        Y                  (observations, response_vars) float64 240B 1.687e+03 ....
        X                  (observations, covariates) float64 80B 25.63 ... 19.88
        batch_effects      (observations, batch_effect_dims) &lt;U17 1kB &#x27;1&#x27; ... &#x27;An...
        Z                  (observations, response_vars) float64 240B 0.7423 ... ...
        centiles           (centile, observations, response_vars) float64 1kB 152...
        baseline_logp      (observations, response_vars) float64 240B -1.131 ... ...
        logp               (observations, response_vars) float64 240B -1.158 ... ...
        Yhat               (observations, response_vars) float64 240B 1.21e+03 .....
        statistics         (response_vars, statistic) float64 312B 0.1172 ... 0.1246
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-6e1bba11-0e1c-4377-96f1-413046116bbf' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-6e1bba11-0e1c-4377-96f1-413046116bbf' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 13</li></ul></div></li><li class='xr-section-item'><input id='section-89b50de0-6d79-4562-9346-2af867c2245f' class='xr-section-summary-in' type='checkbox' checked /><label for='section-89b50de0-6d79-4562-9346-2af867c2245f' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-2f4940c3-0e32-4b07-851e-667558e21d01' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2f4940c3-0e32-4b07-851e-667558e21d01' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c3862d43-de64-4305-8237-bb153a891750' class='xr-var-data-in' type='checkbox'><label for='data-c3862d43-de64-4305-8237-bb153a891750' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-b8b17d58-75bb-4b88-97be-bf860ac8df71' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b8b17d58-75bb-4b88-97be-bf860ac8df71' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7766e950-69e0-4c59-b5e8-f941c22fd739' class='xr-var-data-in' type='checkbox'><label for='data-7766e950-69e0-4c59-b5e8-f941c22fd739' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-570ce316-297d-40c6-90ca-de1bba9a500a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-570ce316-297d-40c6-90ca-de1bba9a500a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-607bc381-683e-4848-ac58-c35a73b6c95f' class='xr-var-data-in' type='checkbox'><label for='data-607bc381-683e-4848-ac58-c35a73b6c95f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-5eca11d0-e203-4c36-94dd-68ac5124735c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5eca11d0-e203-4c36-94dd-68ac5124735c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-efc7b8c6-cc44-47ce-b00d-1e55c3fcef75' class='xr-var-data-in' type='checkbox'><label for='data-efc7b8c6-cc44-47ce-b00d-1e55c3fcef75' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-675e7b89-9d6f-4dad-aa14-3a7d485fc364' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-675e7b89-9d6f-4dad-aa14-3a7d485fc364' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6977c140-1774-463d-a7c4-f52b80ea96d9' class='xr-var-data-in' type='checkbox'><label for='data-6977c140-1774-463d-a7c4-f52b80ea96d9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-b5e347ad-55f6-436f-9500-5bb6b8278854' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b5e347ad-55f6-436f-9500-5bb6b8278854' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6e7b6028-c760-4ddc-812a-74b7a7a93e91' class='xr-var-data-in' type='checkbox'><label for='data-6e7b6028-c760-4ddc-812a-74b7a7a93e91' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-544bf7b3-00c3-4326-9d40-5a4fb9b1fd86' class='xr-section-summary-in' type='checkbox' checked /><label for='section-544bf7b3-00c3-4326-9d40-5a4fb9b1fd86' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-d4154328-d2ed-4c65-abbf-547071ddc85c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d4154328-d2ed-4c65-abbf-547071ddc85c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f7dcce79-4b3f-4540-adfd-08a9d6af90fa' class='xr-var-data-in' type='checkbox'><label for='data-f7dcce79-4b3f-4540-adfd-08a9d6af90fa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-71d54abc-afb4-4533-8efd-605042a20ebe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-71d54abc-afb4-4533-8efd-605042a20ebe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ec6b00f4-56e0-408a-8bc8-e0e1e1c89229' class='xr-var-data-in' type='checkbox'><label for='data-ec6b00f4-56e0-408a-8bc8-e0e1e1c89229' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-e9da2221-909c-47ff-ba96-2ac94d232617' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e9da2221-909c-47ff-ba96-2ac94d232617' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-adea8981-ae66-4e09-bb0b-9ef0249eee0c' class='xr-var-data-in' type='checkbox'><label for='data-adea8981-ae66-4e09-bb0b-9ef0249eee0c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-066062c9-a3aa-4c5d-b80f-c9ccefa589df' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-066062c9-a3aa-4c5d-b80f-c9ccefa589df' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-990164fa-4bd6-4c11-9215-1e6d8e9b8412' class='xr-var-data-in' type='checkbox'><label for='data-990164fa-4bd6-4c11-9215-1e6d8e9b8412' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.7423 -0.8588 ... -0.2163 0.1348</div><input id='attrs-a073a939-f0f3-4093-a58a-cfcafa3206fb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a073a939-f0f3-4093-a58a-cfcafa3206fb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-75f9d420-0807-4dfa-a1e4-ee94fadee702' class='xr-var-data-in' type='checkbox'><label for='data-75f9d420-0807-4dfa-a1e4-ee94fadee702' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.74232793, -0.85876023,  0.02189627],
           [ 0.44528485,  0.68755345, -0.24940187],
           [ 0.22426361,  0.25875398,  0.40908791],
           [ 0.81285642, -0.35401788,  0.06283186],
           [ 0.99366811,  0.03857657, -1.14654771],
           [ 1.3031248 , -0.21553727,  0.14927694],
           [ 1.52363443, -0.34791007,  0.15975631],
           [ 1.64150844,  0.74356583,  0.30027241],
           [ 0.02205415,  0.00363442, -0.82329143],
           [ 0.76801572, -0.21630414,  0.13481513]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>152.3 904.5 ... 1.334e+04 2.469e+04</div><input id='attrs-4e6952f8-c5b5-4838-a704-cf597f186cb0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4e6952f8-c5b5-4838-a704-cf597f186cb0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7531ac6e-d813-4ae9-990a-93f9321083e5' class='xr-var-data-in' type='checkbox'><label for='data-7531ac6e-d813-4ae9-990a-93f9321083e5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 1.52338009e+02,  9.04481790e+02,  1.65032188e+04],
            [ 2.73459612e+01, -2.08436861e+01,  1.64703214e+04],
            [ 2.13440820e+02,  1.35695686e+03,  1.65189125e+04],
            [ 2.50889145e+02,  1.63430827e+03,  1.65284042e+04],
            [-5.44261409e+01, -6.26021790e+02,  1.64482251e+04],
            [ 2.13611877e+02,  1.35822368e+03,  1.65189561e+04],
            [-1.42115962e+01, -3.28420820e+02,  1.64591484e+04],
            [ 8.08764963e+01,  3.75404463e+02,  1.64845407e+04],
            [ 1.42983159e+01, -1.17416442e+02,  1.64668261e+04],
            [ 5.37748149e+01,  1.74782235e+02,  1.64773660e+04]],
    
           [[ 7.76039815e+02,  4.78660959e+03,  1.89251151e+04],
            [ 6.51194761e+02,  3.86219915e+03,  1.88927884e+04],
            [ 8.37133934e+02,  5.23903055e+03,  1.89407751e+04],
            [ 8.74597523e+02,  5.51647698e+03,  1.89503261e+04],
            [ 5.69612225e+02,  3.25820111e+03,  1.88714282e+04],
            [ 8.37305025e+02,  5.24029758e+03,  1.89408188e+04],
            [ 6.09724345e+02,  3.55516448e+03,  1.88819537e+04],
            [ 7.04641150e+02,  4.25792349e+03,  1.89066810e+04],
            [ 6.38172424e+02,  3.76578394e+03,  1.88893913e+04],
    ...
            [ 1.51845627e+03,  9.26032383e+03,  2.22604513e+04],
            [ 1.70417901e+03,  1.06358079e+04,  2.23075977e+04],
            [ 1.74166382e+03,  1.09133865e+04,  2.23172310e+04],
            [ 1.43713727e+03,  8.65796628e+03,  2.22401141e+04],
            [ 1.70435015e+03,  1.06370753e+04,  2.23076416e+04],
            [ 1.47710700e+03,  8.95404326e+03,  2.22500869e+04],
            [ 1.57178568e+03,  9.65531997e+03,  2.22738897e+04],
            [ 1.50546912e+03,  9.16412764e+03,  2.22571908e+04],
            [ 1.54477636e+03,  9.45527269e+03,  2.22670736e+04]],
    
           [[ 2.26679879e+03,  1.40655900e+04,  2.47138810e+04],
            [ 2.14230507e+03,  1.31433667e+04,  2.46829183e+04],
            [ 2.32787213e+03,  1.45178816e+04,  2.47294603e+04],
            [ 2.36537220e+03,  1.47955552e+04,  2.47391529e+04],
            [ 2.06117563e+03,  1.25421892e+04,  2.46633172e+04],
            [ 2.32804330e+03,  1.45191492e+04,  2.47295043e+04],
            [ 2.10104294e+03,  1.28376286e+04,  2.46728922e+04],
            [ 2.19555034e+03,  1.35378390e+04,  2.46960301e+04],
            [ 2.12934323e+03,  1.30473280e+04,  2.46797561e+04],
            [ 2.16857966e+03,  1.33380323e+04,  2.46893640e+04]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.131 -1.309 ... -1.02 -0.9262</div><input id='attrs-3286f9c6-305b-481a-81d0-d8a1f9ef702a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3286f9c6-305b-481a-81d0-d8a1f9ef702a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e771bbb0-5a10-4277-bcdb-9466c6138ab6' class='xr-var-data-in' type='checkbox'><label for='data-e771bbb0-5a10-4277-bcdb-9466c6138ab6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.1308011 , -1.30932152, -0.91909123],
           [-0.93467347, -0.98176331, -0.95458511],
           [-0.94845513, -0.95661131, -1.00355878],
           [-1.29465948, -0.94700075, -0.92129015],
           [-1.08974199, -0.99825982, -1.60822936],
           [-1.74286138, -0.93365463, -0.93038344],
           [-1.58675451, -1.15865566, -0.92852933],
           [-1.91471408, -1.04399044, -0.96066507],
           [-0.9502843 , -0.96760084, -1.27432629],
           [-1.05843845, -1.02038092, -0.92619223]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.158 -1.227 ... -0.8818 -0.9294</div><input id='attrs-f2a65a26-fbc4-43e2-9712-dd7541872379' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f2a65a26-fbc4-43e2-9712-dd7541872379' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ea7da672-e0a8-4db1-b48f-7a5f8c55b1ca' class='xr-var-data-in' type='checkbox'><label for='data-ea7da672-e0a8-4db1-b48f-7a5f8c55b1ca' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.15823485, -1.22696359, -0.92035125],
           [-0.98208442, -1.09482957, -0.95144778],
           [-0.90784262, -0.89169189, -1.00377405],
           [-1.21308779, -0.92090389, -0.92209598],
           [-1.37693709, -0.85951263, -1.57793672],
           [-1.73176271, -0.88144329, -0.93123945],
           [-2.04381574, -0.91912509, -0.93324782],
           [-2.23008521, -1.13477486, -0.96529403],
           [-0.88322888, -0.85851187, -1.25929208],
           [-1.17779625, -0.88178548, -0.92936176]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.21e+03 7.485e+03 ... 2.058e+04</div><input id='attrs-82a4258d-37bd-4551-b82e-f5edfa83bd88' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-82a4258d-37bd-4551-b82e-f5edfa83bd88' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b7c62713-3802-4785-9793-1aeee7422950' class='xr-var-data-in' type='checkbox'><label for='data-b7c62713-3802-4785-9793-1aeee7422950' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1209.56839699,  7485.03589302, 20608.54989065],
           [ 1084.82551642,  6561.26149045, 20576.61985005],
           [ 1270.65647431,  7937.41924242, 20624.18641259],
           [ 1308.13067301,  8214.93171727, 20633.7785647 ],
           [ 1003.37474666,  5958.08369124, 20555.77115414],
           [ 1270.82758938,  7938.68642267, 20624.23021237],
           [ 1043.41567129,  6254.60386984, 20566.02030297],
           [ 1138.21341592,  6956.62172858, 20590.28538183],
           [ 1071.82077167,  6464.95579141, 20573.29106667],
           [ 1111.17723605,  6756.40724902, 20583.36501638]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1172 14.09 ... 0.9965 0.1246</div><input id='attrs-9c7425f6-9edd-4900-8773-20b05cbfdc91' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9c7425f6-9edd-4900-8773-20b05cbfdc91' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-af431cac-217f-4b15-9727-dc3745d52deb' class='xr-var-data-in' type='checkbox'><label for='data-af431cac-217f-4b15-9727-dc3745d52deb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.17178916e-01,  1.40939294e+01,  1.55120188e-01,
             3.98464399e-01,  1.35593840e+00, -6.30001307e-02,
             1.17178916e-01,  6.26199637e+02,  4.38152477e-03,
             8.85849246e-01,  8.82821084e-01,  8.47175234e-01,
             2.46336177e+00],
           [ 1.57907582e-01,  5.69177001e+00,  1.06808574e-01,
             4.10911653e-01,  1.33302534e+00, -8.59131904e-02,
             1.57907582e-01,  3.90104440e+03,  2.56092410e-01,
             1.42121630e-17,  8.42092418e-01,  8.87553829e-01,
             1.70186849e+00],
           [ 5.48907230e-04,  4.45229187e-01,  1.63841143e-01,
             9.77076318e-02,  1.41864320e+00, -2.95332805e-04,
             5.48907230e-04,  2.49225388e+03,  6.47556344e-02,
             3.36782367e-02,  9.99451093e-01,  9.96466642e-01,
             1.24616254e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d1c47dce-095f-473e-a97f-549aa1d5c438' class='xr-section-summary-in' type='checkbox' checked /><label for='section-d1c47dce-095f-473e-a97f-549aa1d5c438' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7f91452aa3e0&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: 489, &#x27;0&#x27;: 587}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 83, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;0&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



Or, if we want to select only the WM-hypointensities, we can use:

.. code:: python

   norm_data.sel(response_vars="WM-hypointensities")

This will return a new NormData object with only the WM-hypointensities.

.. code:: ipython3

    norm_data.sel(response_vars="WM-hypointensities").statistics




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;statistics&#x27; (statistic: 13)&gt; Size: 104B
    array([ 1.17178916e-01,  1.40939294e+01,  1.55120188e-01,  3.98464399e-01,
            1.35593840e+00, -6.30001307e-02,  1.17178916e-01,  6.26199637e+02,
            4.38152477e-03,  8.85849246e-01,  8.82821084e-01,  8.47175234e-01,
            2.46336177e+00])
    Coordinates:
      * statistic      (statistic) &lt;U8 416B &#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;
        response_vars  &lt;U22 88B &#x27;WM-hypointensities&#x27;</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-obj-name'>&#x27;statistics&#x27;</div><ul class='xr-dim-list'><li><span class='xr-has-index'>statistic</span>: 13</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-7d0b97ba-082f-4bdf-bca0-cbd5e6892ed9' class='xr-array-in' type='checkbox' checked><label for='section-7d0b97ba-082f-4bdf-bca0-cbd5e6892ed9' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.1172 14.09 0.1551 0.3985 1.356 ... 0.8858 0.8828 0.8472 2.463</span></div><div class='xr-array-data'><pre>array([ 1.17178916e-01,  1.40939294e+01,  1.55120188e-01,  3.98464399e-01,
            1.35593840e+00, -6.30001307e-02,  1.17178916e-01,  6.26199637e+02,
            4.38152477e-03,  8.85849246e-01,  8.82821084e-01,  8.47175234e-01,
            2.46336177e+00])</pre></div></div></li><li class='xr-section-item'><input id='section-7cc577b9-56c6-4ba7-848b-7cd85d082040' class='xr-section-summary-in' type='checkbox' checked /><label for='section-7cc577b9-56c6-4ba7-848b-7cd85d082040' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-3a8782bb-0c0b-450a-a728-9dcf1481c844' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3a8782bb-0c0b-450a-a728-9dcf1481c844' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-33ca975c-6803-4530-a0b3-7c9b7af5aa4e' class='xr-var-data-in' type='checkbox'><label for='data-33ca975c-6803-4530-a0b3-7c9b7af5aa4e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-75851a64-7102-486a-80c6-e2b26ec1c73e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-75851a64-7102-486a-80c6-e2b26ec1c73e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-94a243f9-04f3-4182-b252-c628dfe1fc50' class='xr-var-data-in' type='checkbox'><label for='data-94a243f9-04f3-4182-b252-c628dfe1fc50' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li></ul></div></li></ul></div></div>



Now we can use the to_dataframe method to cast that selection back to a
pandas dataframe.

.. code:: ipython3

    new_df = norm_data.sel(response_vars="WM-hypointensities").to_dataframe()
    new_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    
        .dataframe thead tr:last-of-type th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>logp</th>
          <th>Yhat</th>
          <th colspan="2" halign="left">batch_effects</th>
          <th>subject_ids</th>
          <th colspan="5" halign="left">centiles</th>
        </tr>
        <tr>
          <th></th>
          <th>age</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>sex</th>
          <th>site</th>
          <th>subject_ids</th>
          <th>(WM-hypointensities, 0.05)</th>
          <th>(WM-hypointensities, 0.25)</th>
          <th>(WM-hypointensities, 0.5)</th>
          <th>(WM-hypointensities, 0.75)</th>
          <th>(WM-hypointensities, 0.95)</th>
        </tr>
        <tr>
          <th>observations</th>
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
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.63</td>
          <td>1686.7</td>
          <td>0.742328</td>
          <td>-1.158235</td>
          <td>1209.568397</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>0</td>
          <td>152.338009</td>
          <td>776.039815</td>
          <td>1209.568397</td>
          <td>1643.096979</td>
          <td>2266.798785</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.34</td>
          <td>1371.1</td>
          <td>0.445285</td>
          <td>-0.982084</td>
          <td>1084.825516</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>1</td>
          <td>27.345961</td>
          <td>651.194761</td>
          <td>1084.825516</td>
          <td>1518.456272</td>
          <td>2142.305072</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.20</td>
          <td>1414.8</td>
          <td>0.224264</td>
          <td>-0.907843</td>
          <td>1270.656474</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>2</td>
          <td>213.440820</td>
          <td>837.133934</td>
          <td>1270.656474</td>
          <td>1704.179014</td>
          <td>2327.872129</td>
        </tr>
        <tr>
          <th>3</th>
          <td>31.39</td>
          <td>1830.6</td>
          <td>0.812856</td>
          <td>-1.213088</td>
          <td>1308.130673</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>3</td>
          <td>250.889145</td>
          <td>874.597523</td>
          <td>1308.130673</td>
          <td>1741.663823</td>
          <td>2365.372201</td>
        </tr>
        <tr>
          <th>4</th>
          <td>13.58</td>
          <td>1642.4</td>
          <td>0.993668</td>
          <td>-1.376937</td>
          <td>1003.374747</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>4</td>
          <td>-54.426141</td>
          <td>569.612225</td>
          <td>1003.374747</td>
          <td>1437.137268</td>
          <td>2061.175634</td>
        </tr>
      </tbody>
    </table>
    </div>



This should give you a pretty good overview of how to work with
NormData. Most of the functionality is built on top of xarray, so if you
want to learn more about xarray, you can check out the `xarray
documentation <https://docs.xarray.dev/en/stable/>`__. However, the
Xarray.DataSet class does not officially support being extended, so the
API does not work completely as expected.

If you have any suggestions for improvements, please let us know!

Pre-processing and split datasets
---------------------------------

Sometimes we have a dataset that is pre-split into train and test, and
we want to use that exact data split to fit the model. We can then load
the data into two NormData objects, but we have to make sure that the
two datasets are compatible. This will ensure that the fitted model is
applicable to both of them.

.. code:: ipython3

    # Download an example dataset:
    data = pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    )
    # Create an arbitrary split as a placeholder for a predefined split ()
    train, test = train_test_split(data, test_size=100)

.. code:: ipython3

    # specify the column names to use
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = ["WM-hypointensities", "Left-Lateral-Ventricle", "Brain-Stem"]
    
    # create NormData objects
    norm_train = NormData.from_dataframe(
        name="train",
        dataframe=train,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
    )
    norm_test = NormData.from_dataframe(
        name="test",
        dataframe=test,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
    )


.. code:: text

    Process: 2590 - 2026-10-27 14:33:27 - Dataset "train" created.
        - 978 observations
        - 978 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 2590 - 2026-10-27 14:33:27 - Dataset "test" created.
        - 100 observations
        - 100 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (18)
        


.. code:: ipython3

    # Should print false, because the train and test split do not contain the same sites
    print(norm_train.check_compatibility(norm_test))


.. code:: text

    True


.. code:: ipython3

    norm_train.check_compatibility(norm_test)




.. code:: text

    True



.. code:: ipython3

    norm_train.make_compatible(norm_test)

.. code:: ipython3

    norm_train.check_compatibility(norm_test)




.. code:: text

    True


