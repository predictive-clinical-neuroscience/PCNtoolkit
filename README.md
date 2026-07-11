# <img src="https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit/dev/doc/_static/pcn-icon.png" alt="PCNtoolkit logo" height="52" align="top" style="vertical-align: middle; margin-bottom: 20px;" /> PCNtoolkit

[![Downloads][downloads-badge]][downloads-link]
[![DOI][doi-badge]][doi-link]
[![License: GPL v3][license-badge]][license-link]

PCNtoolkit is an open-source Python package for Normative Modelling of neuroimaging data.

## Deprecation warning

With version 1.X.X (June 2025), PCNtoolkit was rewritten to be object-oriented, making it more extendable and maintainable. As a result, version 0.X.X is not compatible with 1.X.X and is no longer actively maintained.

## Installation

```bash
pip install pcntoolkit
```

## Example usage

```python
from pcntoolkit import {load_fcon, BLR, NormativeModel}

fcon1000 = load_fcon()

train, test = fcon1000.train_test_split()

# Create a BLR model with heteroskedastic noise
model = NormativeModel(BLR(heteroskedastic=True), 
                       inscaler='standardize', 
                       outscaler='standardize')

model.fit_predict(train, test)
```

## Help and support

### Users
The best place to learn Normative Modelling and the PCNtoolkit is our [website documentation](https://pcntoolkit.readthedocs.io/en/latest/).

Feel free to ask your questions and engage in discussions with the community on the [NeuroStars online forum](https://neurostars.org/tags/pcntoolkit). Please add the tag *pcntoolkit* to your post.

### Contributors

Contributions are always welcome! To start see our [website contributing guidelines](https://pcntoolkit.readthedocs.io/en/stable/pages/contributing.html).

You can find more in depth guidelines in our [GitHub Wiki](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/wiki).


## License

PCNtoolkit is released under the [GPL-3.0-only](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/blob/dev/LICENSE) license.

<!-- Badge and link definitions -->
[downloads-badge]: https://img.shields.io/pypi/dm/pcntoolkit.svg?label=PyPI
[downloads-link]: https://pypi.org/project/pcntoolkit/

[license-badge]: https://img.shields.io/badge/License-GPLv3-blue.svg
[license-link]: https://www.gnu.org/licenses/gpl-3.0

[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.7498917.svg
[doi-link]: https://doi.org/10.5281/zenodo.7498917

