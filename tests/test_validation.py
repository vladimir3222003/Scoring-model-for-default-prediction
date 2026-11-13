import pandas as pd
import pytest

from pd_model.data.validation import validate_data


@pytest.fixture
def valid_df():
    data = {
        "ID": [1, 2],
        "LIMIT_BAL": [50000.0, 100000.0],
        "SEX": [1, 2],
        "EDUCATION": [2, 1],
        "MARRIAGE": [1, 2],
        "AGE": [30, 25],
        "PAY_0": [0, 1],
        "PAY_2": [-1, 0],
        "PAY_3": [0, 0],
        "PAY_4": [0, 0],
        "PAY_5": [0, 0],
        "PAY_6": [0, 0],
        "BILL_AMT1": [10000.0, 20000.0],
        "BILL_AMT2": [9000.0, 18000.0],
        "BILL_AMT3": [8000.0, 16000.0],
        "BILL_AMT4": [7000.0, 14000.0],
        "BILL_AMT5": [6000.0, 12000.0],
        "BILL_AMT6": [5000.0, 10000.0],
        "PAY_AMT1": [2000.0, 3000.0],
        "PAY_AMT2": [2000.0, 3000.0],
        "PAY_AMT3": [2000.0, 3000.0],
        "PAY_AMT4": [2000.0, 3000.0],
        "PAY_AMT5": [2000.0, 3000.0],
        "PAY_AMT6": [2000.0, 3000.0],
        "default.payment.next.month": [0, 1],
    }

    return pd.DataFrame(data)


def test_valid_data(valid_df):
    result = validate_data(valid_df)
    assert result.equals(valid_df)


def test_null_value(valid_df):
    valid_df.loc[0, "AGE"] = None

    with pytest.raises(ValueError):
        validate_data(valid_df)


def test_invalid_education(valid_df):
    valid_df.loc[0, "EDUCATION"] = 10

    with pytest.raises(ValueError):
        validate_data(valid_df)


def test_negative_payment(valid_df):
    valid_df.loc[0, "PAY_AMT1"] = -100.0

    with pytest.raises(ValueError):
        validate_data(valid_df)


def test_duplicate_id(valid_df):
    valid_df.loc[1, "ID"] = valid_df.loc[0, "ID"]

    with pytest.raises(ValueError):
        validate_data(valid_df)
