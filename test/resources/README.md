# Pre-trained models

Pretrained models in test/resources/pretrained_* are produced by an actual training
run under PCNtoolkit 1.1.2 with a single response variable. They test
pcntoolkit/util/migration.py which was introduced in v1.1.2.

**Do not regenerate these with a newer PCNtoolkit.** Otehrwise the tests will be unusable

## Provenance

Generated on Linux.

Both models: one covariate `covariate_0`, one response variable
`response_var_0`, batch effect `site` with values `0` and `1`, standardize
in/outscalers.

| | Configuration | Size |
|---|---|---|
| `pretrained_blr` | l-bfgs-b, 50 iterations, no warp | 8 KB |
| `pretrained_hbr` | Normal likelihood, pymc sampler, draws=15, tune=5, chains=2 | 68 KB |

HBR sampling is deliberately minimal to keep `idata.nc` small enough to commit.