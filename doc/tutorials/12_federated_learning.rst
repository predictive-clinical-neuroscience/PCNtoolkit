Federated normative modeling
============================

In multi-site neuroimaging studies, data often cannot leave the hospital
or institution where it was collected, due to privacy regulations such
as GDPR. **Federated learning (FL)** addresses this constraint: each
site trains a model locally and shares only the trained model
parameters; never the raw data.

This tutorial demonstrates a FL workflow for normative modeling using
the PCNtoolkit. For more details you can read the paper below:

   | Kia SM, Huijsdens H, Rutherford S, de Boer A, Dinga R, Wolfers T,
     et al. (2022)
   | *Closing the life-cycle of normative modeling using federated
     hierarchical Bayesian regression.*
   | PLoS ONE 17(12): e0278776.
     https://doi.org/10.1371/journal.pone.0278776

What we will do
---------------

**Classic normative modelling workflow**

1. *Fit a model* on all data together (let’s call it *baseline model* as
   later we will compare it with the extended model produced from the FL
   workflow)

**FL workflow**

2. *Split* the data into a large central dataset and two smaller ones
3. *Fit a central model* on the central dataset only
4. *Extend* the central model to each of the two smaller datasets

**Comparison classic vs FL workflow**

5. *Compare* the extended model to the baseline model

The functions that we will use
------------------------------

+-------------------------------------+-----------------------------------+
| Function                            | Role                              |
+=====================================+===================================+
| ``NormativeModel.fit_predict()``    | Fit and predict the baseline and  |
|                                     | central model                     |
+-------------------------------------+-----------------------------------+
| ``NormativeModel.extend_predict()`` | Extend the central model with     |
|                                     | data from a remote location +     |
|                                     | synthetic data (generated from    |
|                                     | the central model) and then       |
|                                     | predict on the test data from a   |
|                                     | remote location.                  |
+-------------------------------------+-----------------------------------+

Imports
-------

.. code:: ipython3

    import logging
    import warnings
    
    import matplotlib.pyplot as plt
    import numpy as np
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
        plot_centiles_advanced,
        plot_qq,
    )
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(
        action="ignore", category=FutureWarning
    )
    pd.options.mode.chained_assignment = None
    pcntoolkit.util.output.Output.set_show_messages(
        False
    )

Load data
---------

We use the
`fcon1000 <https://fcon_1000.projects.nitrc.org/fcpClassic/FcpTable.html>`__
dataset that is included in PCNtoolkit. This dataset contains derived
structural MRI phenotypes from 1,078 subjects collected across 23 sites,
including cortical thickness measures, subcortical and ventricular
volumes, and global brain-volume estimates.

For this tutorial, we select a single response variable: the
WM-hypointensitieswhich is a measure related to damaged or diseased
tissue within the brain’s white matter.

.. code:: ipython3

    # Download the dataset
    norm_data: NormData = load_fcon1000()
    
    # Select only the white matter hypointensities feature
    features_to_model = ["WM-hypointensities"]
    norm_data = norm_data.sel(
        {"response_vars": features_to_model}
    )
    
    # Show all available sites
    all_sites = np.unique(
        norm_data.batch_effects.sel(
            batch_effect_dims="site"
        ).values
    )
    print(
        f"Total sites: {len(all_sites)}"
    )
    print(f"Sites: {all_sites}")


.. parsed-literal::

    Total sites: 23
    Sites: ['AnnArbor_a' 'AnnArbor_b' 'Atlanta' 'Baltimore' 'Bangor' 'Beijing_Zang'
     'Berlin_Margulies' 'Cambridge_Buckner' 'Cleveland' 'ICBM' 'Leiden_2180'
     'Leiden_2200' 'Milwaukee_b' 'Munchen' 'NewYork_a' 'NewYork_a_ADHD'
     'Newark' 'Oulu' 'Oxford' 'PaloAlto' 'Pittsburgh' 'Queensland'
     'SaintLouis']
    

Split data
----------

We split the data into:

- A large central dataset (19 sites)
- Two smaller datasets (each dataset has 2 sites)

In a FL scenario the large model would be owned by a central location
(e.g., a hospital in the Netherlands) and the smaller ones by remote
locations 1 and 2 (e.g, a hospital in France and in the USA). All these
locations don’t want to share their data due to privacy. For this
reason, they use the FL workflow.

.. code:: ipython3

    # Pick 2 sites for each remote location
    location1_sites = list(all_sites[:2])
    location2_sites = list(all_sites[2:4])
    print(
        f"Location 1 sites: {location1_sites}"
    )
    print(
        f"Location 2 sites: {location2_sites}"
    )
    
    # Split off location 1
    location1_data, remaining = (
        norm_data.batch_effects_split(
            {"site": location1_sites},
            names=("location1", "remaining"),
        )
    )
    
    # Split off location 2 
    location2_data, central_data = (
        remaining.batch_effects_split(
            {"site": location2_sites},
            names=("location2", "central"),
        )
    )
    
    # Create train/test splits for each location
    train_central, test_central = (
        central_data.train_test_split()
    )
    train_location1, test_location1 = (
        location1_data.train_test_split()
    )
    train_location2, test_location2 = (
        location2_data.train_test_split()
    )
    
    # Global train/test for the baseline model
    train_all, test_all = (
        norm_data.train_test_split()
    )
    
    print(
        f"\nCentral: "
        f"{train_central.X.shape[0]} train, "
        f"{test_central.X.shape[0]} test"
    )
    print(
        f"Location 1: "
        f"{train_location1.X.shape[0]} train, "
        f"{test_location1.X.shape[0]} test"
    )
    print(
        f"Location 2: "
        f"{train_location2.X.shape[0]} train, "
        f"{test_location2.X.shape[0]} test"
    )
    print(
        f"All data: "
        f"{train_all.X.shape[0]} train, "
        f"{test_all.X.shape[0]} test"
    )


.. parsed-literal::

    Location 1 sites: [np.str_('AnnArbor_a'), np.str_('AnnArbor_b')]
    Location 2 sites: [np.str_('Atlanta'), np.str_('Baltimore')]
    
    Central: 776 train, 195 test
    Location 1: 44 train, 12 test
    Location 2: 40 train, 11 test
    All data: 862 train, 216 test
    

Visualize the data
------------------

.. code:: ipython3

    feature = features_to_model[0]
    datasets = {
        "Central location": train_central,
        "Location 1": train_location1,
        "Location 2": train_location2,
    }
    
    fig, axes = plt.subplots(
        3, 2, figsize=(15, 12)
    )
    
    # for every dataset
    for i, (name, data) in enumerate(
        datasets.items()
    ):
        df = data.to_dataframe()
        # Count plot
        sns.countplot(
            data=df,
            y=("batch_effects", "site"),
            hue=("batch_effects", "sex"),
            ax=axes[i, 0],
            orient="h",
        )
        axes[i, 0].legend(title="Sex")
        axes[i, 0].set_title(
            f"{name}"
        )
        axes[i, 0].set_xlabel("Count")
        axes[i, 0].set_ylabel("Site")
    
        # Scatter plot
        sns.scatterplot(
            data=df,
            x=("X", "age"),
            y=("Y", feature),
            hue=("batch_effects", "site"),
            style=("batch_effects", "sex"),
            ax=axes[i, 1],
        )
        axes[i, 1].legend([], [])
        axes[i, 1].set_title(
            f"{name}"
        )
        axes[i, 1].set_xlabel("Age")
        axes[i, 1].set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()



.. image:: 12_federated_learning_files/12_federated_learning_8_0.png


Configure the HBR model
-----------------------

We define a shared model configuration that will be used for both
baseline and FL model. This ensures a fair comparison. We use a Normal
likelihood HBR with B-spline basis functions.

.. code:: ipython3

    mu = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
        intercept=make_prior(
            random=True,
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
            sigma=make_prior(dist_name="Normal", dist_params=(0.0, 1.0), mapping="softplus", mapping_params=(0.0, 3.0)),
        ),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )
    
    likelihood = NormalLikelihood(mu, sigma)
    
    template_hbr = HBR(
        name="template",
        cores=16,
        progressbar=False,
        draws=1500,
        tune=500,
        chains=4,
        nuts_sampler="nutpie",
        likelihood=likelihood,
    )
    

--------------

Part 1: Baseline model
----------------------

In a non-FL scenario, we would pool all the data from all the 23 sites
into a single dataset and train one model.

.. code:: ipython3

    baseline_model = NormativeModel(
        template_regression_model=template_hbr,
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=False,
        save_dir=(
            "resources/federated/baseline"
        ),
        inscaler="standardize",
        outscaler="standardize",
    );
    
    # Use the data from all 23 sites, before any splitting happened.
    baseline_model.fit_predict(
        train_all, test_all);

Part 2: FL with ``extend()``
----------------------------

Now we simulate the FL scenario. None of the locations (central,
location 1 and 2) share their data with each other. Only model
parameters are exchanged.

Step 1: Train the central model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The central location trains an HBR model using only its own 19 sites.

.. code:: ipython3

    central_model = NormativeModel(
        template_regression_model=template_hbr,
        savemodel=True,
        evaluate_model=True,
        saveresults=False,
        saveplots=False,
        save_dir=(
            "resources/federated/central"
        ),
        inscaler="standardize",
        outscaler="standardize",
    );
    
    central_model.fit_predict(train_central, test_central);

.. code:: ipython3

    # Centile curves for the central model
    plot_centiles_advanced(
        central_model,
        scatter_data=train_central,
        batch_effects="all",
        show_legend=False
    )



.. image:: 12_federated_learning_files/12_federated_learning_15_0.png




.. parsed-literal::

    [<Figure size 640x480 with 1 Axes>]



Step 2: Extend the central model to remote locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Location 1 receives the central model json files that are saved in
``resources/federated/central`` and calls ``extend_predict()`` locally
using its own private data.

``extend_predict()`` runs both ``extend()`` and ``predict()``.
``extend()`` synthesizes data from the central model’s learned
distribution, merges it with the real local data, and refits a full
model.

No real data is exchanged only model parameters.

.. code:: ipython3

    # extend() synthesizes random data, so we set a seed to make sure in this tutorial the results are always the same
    import numpy as np
    np.random.seed(42)
    
    # Location 1 loads the central model from disk
    central_model = NormativeModel.load("resources/federated/central")
    
    # Location 1 extends the central model
    # with their private data.
    extended_location_1 = central_model.extend_predict(
        train_location1,
        test_location1,
        save_dir=(
            "resources/federated/extended_location_1"
        ),
    );


::


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\file_manager.py:219, in CachingFileManager._acquire_with_cache_info(self, needs_lock)
        218 try:
    --> 219     file = self._cache[self._key]
        220 except KeyError:
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\lru_cache.py:56, in LRUCache.__getitem__(self, key)
         55 with self._lock:
    ---> 56     value = self._cache[key]
         57     self._cache.move_to_end(key)
    

    KeyError: [<class 'h5netcdf.core.File'>, ('c:\\Users\\kontsi\\Documents\\GitHub\\PCNtoolkit-local\\examples\\resources\\federated\\extended_location_1\\model\\WM-hypointensities\\idata.nc',), 'a', (('decode_vlen_strings', True), ('driver', None), ('format', 'NETCDF4'), ('invalid_netcdf', None)), '2f331e2f-1e29-4518-b917-d07c43d5c932']

    
    During handling of the above exception, another exception occurred:
    

    OSError                                   Traceback (most recent call last)

    Cell In[22], line 10
          6 central_model = NormativeModel.load("resources/federated/central")
          8 # Location 1 extends the central model
          9 # with their private data.
    ---> 10 extended_location_1 = central_model.extend_predict(
         11     train_location1,
         12     test_location1,
         13     save_dir=(
         14         "resources/federated/extended_location_1"
         15     ),
         16 );
    

    File ~\Documents\GitHub\PCNtoolkit-local\pcntoolkit\normative_model.py:284, in NormativeModel.extend_predict(self, extend_data, predict_data, save_dir, n_synth_samples)
        278 def extend_predict(
        279     self, extend_data: NormData, predict_data: NormData, save_dir: str | None = None, n_synth_samples: int | None = None
        280 ) -> NormativeModel:
        281     """
        282     Extends the model to a new dataset and predicts the data.
        283     """
    --> 284     new_model = self.extend(extend_data, save_dir, n_synth_samples)
        285     new_model.predict(predict_data)
        286     return new_model
    

    File ~\Documents\GitHub\PCNtoolkit-local\pcntoolkit\normative_model.py:275, in NormativeModel.extend(self, data, save_dir, n_synth_samples)
        261     save_dir = self.save_dir + "_extend"
        263 new_model = NormativeModel(
        264     copy.deepcopy(self.template_regression_model),
        265     savemodel=True,
       (...)    272     save_dir=save_dir,
        273 )
    --> 275 new_model.fit(merged_data)
        276 return new_model
    

    File ~\Documents\GitHub\PCNtoolkit-local\pcntoolkit\normative_model.py:148, in NormativeModel.fit(self, data)
        146 self.postprocess(data)
        147 if self.savemodel:  # Make sure model is saved
    --> 148     self.save()
        149 self.predict(data)
    

    File ~\Documents\GitHub\PCNtoolkit-local\pcntoolkit\normative_model.py:461, in NormativeModel.save(self, path)
        459 os.makedirs(regmodel_path, exist_ok=True)
        460 reg_model_dict = {}
    --> 461 reg_model_dict["model"] = model.to_dict(regmodel_path)
        462 reg_model_dict["outscaler"] = self.outscalers[responsevar].to_dict()
        463 with open(os.path.join(regmodel_path, "regression_model.json"), "w", encoding="utf-8") as f:
    

    File ~\Documents\GitHub\PCNtoolkit-local\pcntoolkit\regression_model\hbr.py:348, in HBR.to_dict(self, path)
        346 if self.is_fitted and (path is not None):
        347     idata_path = os.path.join(path, "idata.nc")
    --> 348     self.save_idata(idata_path)
        349     my_dict["idata_path"] = idata_path
        350 if self.is_fitted:
    

    File ~\Documents\GitHub\PCNtoolkit-local\pcntoolkit\regression_model\hbr.py:448, in HBR.save_idata(self, path)
        446 if self.is_fitted:
        447     if hasattr(self, "idata"):
    --> 448         self.idata.to_netcdf(path, groups=["posterior"])
        449     else:
        450         raise ValueError(Output.error(Errors.ERROR_HBR_FITTED_BUT_NO_IDATA))
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\arviz\data\inference_data.py:520, in InferenceData.to_netcdf(self, filename, compress, groups, engine, base_group, overwrite_existing)
        514 if compress:
        515     kwargs["encoding"] = {
        516         var_name: {"zlib": True}
        517         for var_name, values in data.variables.items()
        518         if _compressible_dtype(values.dtype)
        519     }
    --> 520 data.to_netcdf(filename, mode=mode, group=f"{base_group}/{group}", **kwargs)
        521 data.close()
        522 mode = "a"
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\core\dataset.py:2123, in Dataset.to_netcdf(self, path, mode, format, group, engine, encoding, unlimited_dims, compute, invalid_netcdf, auto_complex)
       2120     encoding = {}
       2121 from xarray.backends.writers import to_netcdf
    -> 2123 return to_netcdf(  # type: ignore[return-value]  # mypy cannot resolve the overloads:(
       2124     self,
       2125     path,
       2126     mode=mode,
       2127     format=format,
       2128     group=group,
       2129     engine=engine,
       2130     encoding=encoding,
       2131     unlimited_dims=unlimited_dims,
       2132     compute=compute,
       2133     multifile=False,
       2134     invalid_netcdf=invalid_netcdf,
       2135     auto_complex=auto_complex,
       2136 )
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\writers.py:422, in to_netcdf(dataset, path_or_file, mode, format, group, engine, encoding, unlimited_dims, compute, multifile, invalid_netcdf, auto_complex)
        419 else:
        420     target = normalized_path  # type: ignore[assignment]
    --> 422 store = get_writable_netcdf_store(
        423     target,
        424     engine,
        425     mode=mode,
        426     format=format,
        427     autoclose=autoclose,
        428     invalid_netcdf=invalid_netcdf,
        429     auto_complex=auto_complex,
        430 )
        431 if group is not None:
        432     store = store.get_child_store(group)
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\writers.py:78, in get_writable_netcdf_store(target, engine, format, mode, autoclose, invalid_netcdf, auto_complex)
         75 if auto_complex is not None:
         76     kwargs["auto_complex"] = auto_complex
    ---> 78 return store_open(target, mode=mode, format=format, **kwargs)
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\h5netcdf_.py:242, in H5NetCDFStore.open(cls, filename, mode, format, group, lock, autoclose, invalid_netcdf, phony_dims, decode_vlen_strings, driver, driver_kwds, storage_options)
        235 manager_cls = (
        236     CachingFileManager
        237     if isinstance(filename, str) and not is_remote_uri(filename)
        238     else PickleableFileManager
        239 )
        240 manager = manager_cls(h5netcdf.File, filename, mode=mode, kwargs=kwargs)
    --> 242 return cls(
        243     manager,
        244     group=group,
        245     format=format,
        246     mode=mode,
        247     lock=lock,
        248     autoclose=autoclose,
        249 )
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\h5netcdf_.py:152, in H5NetCDFStore.__init__(self, manager, group, mode, format, lock, autoclose)
        149 self.format = format or "NETCDF4"
        150 # todo: utilizing find_root_and_group seems a bit clunky
        151 #  making filename available on h5netcdf.Group seems better
    --> 152 self._filename = find_root_and_group(self.ds)[0].filename
        153 self.is_remote = is_remote_uri(self._filename)
        154 self.lock = ensure_lock(lock)
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\h5netcdf_.py:260, in H5NetCDFStore.ds(self)
        258 @property
        259 def ds(self):
    --> 260     return self._acquire()
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\h5netcdf_.py:252, in H5NetCDFStore._acquire(self, needs_lock)
        251 def _acquire(self, needs_lock=True):
    --> 252     with self._manager.acquire_context(needs_lock) as root:
        253         ds = _nc4_require_group(
        254             root, self._group, self._mode, create_group=_h5netcdf_create_group
        255         )
        256     return ds
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\contextlib.py:137, in _GeneratorContextManager.__enter__(self)
        135 del self.args, self.kwds, self.func
        136 try:
    --> 137     return next(self.gen)
        138 except StopIteration:
        139     raise RuntimeError("generator didn't yield") from None
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\file_manager.py:207, in CachingFileManager.acquire_context(self, needs_lock)
        204 @contextmanager
        205 def acquire_context(self, needs_lock: bool = True) -> Iterator[T_File]:
        206     """Context manager for acquiring a file."""
    --> 207     file, cached = self._acquire_with_cache_info(needs_lock)
        208     try:
        209         yield file
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\xarray\backends\file_manager.py:225, in CachingFileManager._acquire_with_cache_info(self, needs_lock)
        223     kwargs = kwargs.copy()
        224     kwargs["mode"] = self._mode
    --> 225 file = self._opener(*self._args, **kwargs)
        226 if self._mode == "w":
        227     # ensure file doesn't get overridden when opened again
        228     self._mode = "a"
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\h5netcdf\core.py:1907, in File.__init__(self, path, mode, format, invalid_netcdf, phony_dims, backend, **kwargs)
       1905     else:  # default h5py
       1906         self._h5py = h5py
    -> 1907         self.__h5file, self._preexisting_file, self._close_h5file = _open_h5py(
       1908             path, mode, **kwargs
       1909         )
       1911 except Exception:
       1912     self._closed = True
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\h5netcdf\core.py:1800, in _open_h5py(path, mode, **kwargs)
       1796 if isinstance(path, str):
       1797     exists = path.startswith(("http", "s3://")) or (
       1798         os.path.exists(path) and mode != "w"
       1799     )
    -> 1800     h5file = h5py.File(path, mode, **kwargs)
       1801     return h5file, exists, True
       1802 elif isinstance(path, h5py.File):
       1803     # already-open file
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\h5py\_hl\files.py:566, in File.__init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, track_times, **kwds)
        557     fapl = make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0,
        558                      locking, page_buf_size, min_meta_keep, min_raw_keep,
        559                      alignment_threshold=alignment_threshold,
        560                      alignment_interval=alignment_interval,
        561                      meta_block_size=meta_block_size,
        562                      **kwds)
        563     fcpl = make_fcpl(track_order=track_order, track_times=track_times,
        564                      fs_strategy=fs_strategy, fs_persist=fs_persist,
        565                      fs_threshold=fs_threshold, fs_page_size=fs_page_size)
    --> 566     fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
        568 if isinstance(libver, tuple):
        569     self._libver = libver
    

    File c:\Users\kontsi\AppData\Local\anaconda3\envs\.ptk-dev\Lib\site-packages\h5py\_hl\files.py:247, in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
        245     fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
        246 elif mode == 'w':
    --> 247     fid = h5f.create(name, h5f.ACC_TRUNC, fapl=fapl, fcpl=fcpl)
        248 elif mode == 'a':
        249     # Open in append mode (read/write).
        250     # If that fails, create a new file only if it won't clobber an
        251     # existing one (ACC_EXCL)
        252     try:
    

    File h5py/_objects.pyx:54, in h5py._objects.with_phil.wrapper()
    

    File h5py/_objects.pyx:55, in h5py._objects.with_phil.wrapper()
    

    File h5py/h5f.pyx:124, in h5py.h5f.create()
    

    OSError: Unable to synchronously create file (unable to truncate a file which is already open)


.. code:: ipython3

    # Visualize the extended model
    plot_centiles_advanced(
        extended_location_1,
        scatter_data=train_location1,
        batch_effects="all",
    )



.. image:: 12_federated_learning_files/12_federated_learning_18_0.png




.. parsed-literal::

    [<Figure size 640x480 with 1 Axes>]



The extended model from location 1 knows about both the central sites
(via synthetic data) and its own local sites.

Now location 1 shares its extended model parameters with location 2.
Location 2 extends the model further with their own data.

.. code:: ipython3

    # extend() synthesizes random data, so we set a seed to make sure in this tutorial the results are always the same
    import numpy as np
    np.random.seed(42)
    
    # Location 2 loads the extended model from disk
    extended_location_1 = NormativeModel.load("resources/federated/extended_location_1")
    
    # Location 2 extends the model
    # with their private data.
    extended_location_1_and_2 = extended_location_1.extend_predict(
        train_location2,
        test_location2,
        save_dir=(
            "resources/federated/extended_location_1_and_2"
        ),
    )

.. code:: ipython3

    plot_centiles_advanced(
        extended_location_1_and_2,
        scatter_data=train_location2,
        batch_effects="all",
    )



.. image:: 12_federated_learning_files/12_federated_learning_22_0.png




.. parsed-literal::

    [<Figure size 640x480 with 1 Axes>]



The extended model from location 2 knows about both the central and
location 1 sites (via synthetic data) and its own local sites.

--------------

Extended vs baseline model
--------------------------

We now compare the 2 models:

- **baseline**: all data were in one location
- **extended**: data were split in 3 locations

Centile curves
~~~~~~~~~~~~~~

.. code:: ipython3

    # baseline model centiles
    print("=== baseline model ===")
    plot_centiles_advanced(
        baseline_model,
        scatter_data=test_all,
        batch_effects="all",
        show_legend=False
    )
    
    # Extended model centiles
    print("\n=== Extended model ===")
    plot_centiles_advanced(
        extended_location_1_and_2,
        scatter_data=test_all,
        batch_effects="all",
        show_legend=False
    )


.. parsed-literal::

    === baseline model ===
    


.. image:: 12_federated_learning_files/12_federated_learning_25_1.png


.. parsed-literal::

    
    === Extended model ===
    


.. image:: 12_federated_learning_files/12_federated_learning_25_3.png




.. parsed-literal::

    [<Figure size 640x480 with 1 Axes>]



Conclusions
-----------

The two models perform very similarly. So the FL workflow, where the
data are from different locations, performs similar to the baseline
workflow, where all the data are in one location.

A small difference in the centile plots is that, compared to the
baseline, the centiles of the extended model show a downward shift for
old ages (> 80 years), reaching even negative values for WM
hypointensities. This probably happens because there are almost no real
data beyond ~80 years, and the synthetic data have little information in
this range.

Because of this lack of data beyond ~80 years, we should be cautious
with the baseline model as well; the upward trend observed there is not
necessarily more accurate.
