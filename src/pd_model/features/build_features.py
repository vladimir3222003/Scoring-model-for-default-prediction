import pandas as pd


def create_features(df):
    df = df.copy()

    def delay_count(column):
        j = 0
        for pay_i in column:
            if pay_i > 0:
                j += 1
        return j

    pay_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    bill_columns = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]

    payment_columns = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    df["DELAY_COUNT"] = df[pay_columns].apply(delay_count, axis=1)

    positive_delays = df[pay_columns].where(df[pay_columns] > 0)

    df["MAX_DELAY"] = positive_delays.max(axis=1).fillna(0).astype(int)

    df["AVG_POSITIVE_DELAY"] = positive_delays.mean(axis=1).fillna(0)

    df["AVG_BILL_AMT"] = df[bill_columns].mean(axis=1)

    df["MAX_BILL_AMT"] = df[bill_columns].max(axis=1)

    df["AVG_PAY_AMT"] = df[payment_columns].mean(axis=1)

    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[18, 26, 36, 46, 56, 66, 101],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        right=False,
    )

    return df
