import pandas as pd
import pytest

from pd_model.features.build_features import create_features


@pytest.fixture
def sample_df():
    data = {
        "AGE": [30, 40],
        "PAY_0": [2, -1],
        "PAY_2": [0, 0],
        "PAY_3": [-1, 0],
        "PAY_4": [4, 0],
        "PAY_5": [0, 0],
        "PAY_6": [1, 0],
        "BILL_AMT1": [100.0, 200.0],
        "BILL_AMT2": [200.0, 300.0],
        "BILL_AMT3": [300.0, 400.0],
        "BILL_AMT4": [400.0, 500.0],
        "BILL_AMT5": [500.0, 600.0],
        "BILL_AMT6": [600.0, 700.0],
        "PAY_AMT1": [10.0, 20.0],
        "PAY_AMT2": [20.0, 30.0],
        "PAY_AMT3": [30.0, 40.0],
        "PAY_AMT4": [40.0, 50.0],
        "PAY_AMT5": [50.0, 60.0],
        "PAY_AMT6": [60.0, 70.0],
    }

    return pd.DataFrame(data)


def test_created_features(sample_df):
    result = create_features(sample_df)

    expected_columns = [
        "DELAY_COUNT",
        "MAX_DELAY",
        "AVG_POSITIVE_DELAY",
        "AVG_BILL_AMT",
        "MAX_BILL_AMT",
        "AVG_PAY_AMT",
        "AGE_GROUP",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_delay_features(sample_df):
    result = create_features(sample_df)

    assert result.loc[0, "DELAY_COUNT"] == 3
    assert result.loc[0, "MAX_DELAY"] == 4
    assert result.loc[0, "AVG_POSITIVE_DELAY"] == pytest.approx(7 / 3)


def test_bill_features(sample_df):
    result = create_features(sample_df)

    assert result.loc[0, "AVG_BILL_AMT"] == 350.0
    assert result.loc[0, "MAX_BILL_AMT"] == 600.0


def test_payment_features(sample_df):
    result = create_features(sample_df)

    assert result.loc[0, "AVG_PAY_AMT"] == 35.0


def test_no_positive_delays(sample_df):
    sample_df.loc[1, ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] = -1

    result = create_features(sample_df)

    assert result.loc[1, "DELAY_COUNT"] == 0
    assert result.loc[1, "MAX_DELAY"] == 0
    assert result.loc[1, "AVG_POSITIVE_DELAY"] == 0


def test_age_group(sample_df):
    result = create_features(sample_df)

    assert result.loc[0, "AGE_GROUP"] == "26-35"
    assert result.loc[1, "AGE_GROUP"] == "36-45"
