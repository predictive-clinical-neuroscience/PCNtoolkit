Quickstart guide
================
    

Installation
------------

.. code-block:: bash
    
    pip install pcntoolkit


Example usage
-------------

.. code-block:: python

    from pcntoolkit import load_fcon1000, BLR, NormativeModel

    fcon1000 = load_fcon1000()

    train, test = fcon1000.train_test_split()

    # Create a BLR model with heteroskedastic noise
    model = NormativeModel(BLR(heteroskedastic=True), 
                        inscaler='standardize', 
                        outscaler='standardize')

    model.fit_predict(train, test)
