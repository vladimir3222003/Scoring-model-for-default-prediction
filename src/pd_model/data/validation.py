from pathlib import Path

import great_expectations as gx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPECTATIONS_DIR = PROJECT_ROOT / "data" / "expectations"

SUITE_NAME = "credit_data_suite"
SOURCE_NAME = "credit_data_source"
ASSET_NAME = "credit_data_asset"
BATCH_NAME = "credit_data_batch"
VALIDATION_NAME = "credit_data_validation"


REQUIRED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default.payment.next.month",
]

PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

BILL_COLUMNS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]

PAYMENT_COLUMNS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

INTEGER_COLUMNS = [
    "ID",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_COLUMNS,
    "default.payment.next.month",
]

NUMERIC_COLUMNS = ["LIMIT_BAL", *BILL_COLUMNS, *PAYMENT_COLUMNS]


def get_validation_definition():
    EXPECTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    context = gx.get_context(mode="file", project_root_dir=str(EXPECTATIONS_DIR))

    try:
        data_source = context.data_sources.get(SOURCE_NAME)
    except LookupError:
        data_source = context.data_sources.add_pandas(name=SOURCE_NAME)

    try:
        data_asset = data_source.get_asset(ASSET_NAME)
    except LookupError:
        data_asset = data_source.add_dataframe_asset(name=ASSET_NAME)

    try:
        batch_definition = data_asset.get_batch_definition(BATCH_NAME)
    except LookupError:
        batch_definition = data_asset.add_batch_definition_whole_dataframe(BATCH_NAME)

    suite = gx.ExpectationSuite(name=SUITE_NAME)

    for column in REQUIRED_COLUMNS:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="ID"))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="LIMIT_BAL", min_value=1)
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2])
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="EDUCATION", value_set=[0, 1, 2, 3, 4, 5, 6]
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="MARRIAGE", value_set=[0, 1, 2, 3]
        )
    )

    # Возраст
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="AGE", min_value=18, max_value=100
        )
    )

    pay_statuses = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    for column in PAY_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column, value_set=pay_statuses
            )
        )

    for column in PAYMENT_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=0)
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="default.payment.next.month", value_set=[0, 1]
        )
    )

    suite = context.suites.add_or_update(suite)

    validation_definition = gx.ValidationDefinition(
        name=VALIDATION_NAME, data=batch_definition, suite=suite
    )

    validation_definition = context.validation_definitions.add_or_update(
        validation_definition
    )

    return validation_definition


def validate_data(df):
    validation_definition = get_validation_definition()

    validation_results = validation_definition.run(batch_parameters={"dataframe": df})

    if not validation_results.success:
        raise ValueError("Great Expectations: проверка качества данных не пройдена.")

    wrong_integer_types = [
        column
        for column in INTEGER_COLUMNS
        if not pd.api.types.is_integer_dtype(df[column])
    ]

    wrong_numeric_types = [
        column
        for column in NUMERIC_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if wrong_integer_types:
        raise TypeError(
            f"Столбцы должны иметь целочисленный тип: " f"{wrong_integer_types}"
        )

    if wrong_numeric_types:
        raise TypeError(f"Столбцы должны иметь числовой тип: " f"{wrong_numeric_types}")

    return df
