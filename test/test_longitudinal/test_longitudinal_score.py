import numpy as np
import pandas as pd
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.longitudinal_score.longitudinal_score import LongitudinalScore


def _longitudinal_dataframe(*, duplicate_visits: bool = False) -> pd.DataFrame:
    visits = [1, 2, 1, 2] if not duplicate_visits else [1, 1, 1, 1]
    return pd.DataFrame(
        {
            "sub_id": ["a", "a", "b", "b"],
            "visit": visits,
            "age": [20.0, 22.0, 30.0, 32.0],
            "site": ["s1", "s1", "s1", "s1"],
            "sex": ["F", "F", "M", "M"],
            "metric_a": [1.0, 1.1, 2.0, 2.1],
        }
    )


def _predicted_norm_data(dataframe: pd.DataFrame) -> NormData:
    data = NormData.from_dataframe(
        "longitudinal",
        dataframe,
        covariates=["age"],
        batch_effects=["site", "sex"],
        response_vars=["metric_a"],
        subject_ids="sub_id",
        visits="visit",
    )
    data["Yhat"] = (["observations", "response_vars"], data.Y.values.copy())
    data["Z"] = (["observations", "response_vars"], np.zeros(data.Y.shape))
    return data


def test_get_visits_requires_visits_on_norm_data():
    df = _longitudinal_dataframe()
    data = NormData.from_dataframe(
        "longitudinal",
        df,
        covariates=["age"],
        batch_effects=["site", "sex"],
        response_vars=["metric_a"],
        subject_ids="sub_id",
    )
    data["Yhat"] = (["observations", "response_vars"], data.Y.values.copy())
    data["Z"] = (["observations", "response_vars"], np.zeros(data.Y.shape))

    with pytest.raises(ValueError, match="no visit labels"):
        data.get_visits()


def test_get_visits_rejects_non_numeric_labels():
    df = _longitudinal_dataframe()
    df["visit"] = df["visit"].map({1: "baseline", 2: "followup"})
    data = _predicted_norm_data(df)

    with pytest.raises(ValueError, match="must be numeric"):
        data.get_visits()


def test_check_is_longitudinal_rejects_duplicate_visit_labels():
    data = _predicted_norm_data(_longitudinal_dataframe(duplicate_visits=True))

    with pytest.raises(ValueError, match="identical visit labels"):
        LongitudinalScore._check_is_longitudinal(data)


def test_check_is_longitudinal_accepts_distinct_visits():
    data = _predicted_norm_data(_longitudinal_dataframe())
    LongitudinalScore._check_is_longitudinal(data)
