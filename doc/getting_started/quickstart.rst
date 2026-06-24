Quickstart guide
================
    

Installation
------------

.. code-block:: bash
    
    pip install pcntoolkit


Example usage
-------------

.. code-block:: python

    from pcntoolkit import {load_fcon, BLR, NormativeModel}

    fcon1000 = load_fcon()

    train, test = fcon1000.train_test_split()

    # Create a BLR model with heteroskedastic noise
    model = NormativeModel(BLR(heteroskedastic=True), 
                        inscaler='standardize', 
                        outscaler='standardize')

    model.fit_predict(train, test)

