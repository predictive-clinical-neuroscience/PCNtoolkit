import math

import arviz as az
import pymc as pm

from pcntoolkit.normative_model import NormativeModel


def compare_hbr_models(models: dict[str, str]):
    """Compares HBR models

    Args:
        models (dict[str, str]): dictionary of [model name, path]

    Returns:
        dictionary of (responsevar, comparison): [str, dataframe]
    """

    loaded_models: dict[str, NormativeModel] = {}
    for k, v in models.items():
        try:
            m = NormativeModel.load(v)
            savemodels, saveplots, saveresults = m.savemodel, m.saveplots, m.saveresults
            m.savemodel = False
            m.saveplots = False
            m.saveresults = False
            n_samples = int(
                2
                * math.exp(
                    sum([math.log(len(uni)) for uni in m.unique_batch_effects.values()])
                )
            )
            m.predict(m.synthesize(n_samples=n_samples))
            m.savemodel = savemodels
            m.saveplots = saveplots
            m.saveresults = saveresults
            loaded_models[k] = m
        except Exception as e:
            print("Cannot load model at location:", v, e)

    comparisons = {}
    for respvar in loaded_models[k].response_vars:
        traces = {}
        for name, model in loaded_models.items():
            with model[respvar].pymc_model:
                pm.compute_log_likelihood(model[respvar].idata)
            traces[name] = model[respvar].idata
        comparisons[respvar] = az.compare(traces)
    return comparisons
