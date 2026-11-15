from pathlib import Path

import joblib
import pandas as pd

from pd_model.monitoring.psi import calculate_psi

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "pd_model_gradient_boosting.joblib"

REPORT_DIR = ROOT / "reports" / "monitoring"
REPORT_PATH = REPORT_DIR / "psi_report.csv"


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    model = joblib.load(MODEL_PATH)
    feature_columns = model.feature_names_in_.tolist()

    X_train = train[feature_columns]
    X_test = test[feature_columns]

    monitored_features = [
        "LIMIT_BAL",
        "AGE",
        "DELAY_COUNT",
        "MAX_DELAY",
        "AVG_BILL_AMT",
        "AVG_PAY_AMT",
    ]

    results = []

    for feature in monitored_features:
        psi_value = calculate_psi(X_train[feature].values, X_test[feature].values)

        results.append({"feature": feature, "psi": psi_value})

    # вероятности дефолта
    train_pd = model.predict_proba(X_train)[:, 1]
    test_pd = model.predict_proba(X_test)[:, 1]

    pd_psi = calculate_psi(train_pd, test_pd)

    results.append({"feature": "Predicted_PD", "psi": pd_psi})

    report = pd.DataFrame(results)

    def interpret_psi(value):
        if value < 0.1:
            return "Низкий дрейф"
        elif value < 0.25:
            return "Умеренный дрейф"
        else:
            return "Высокий дрейф"

    report["status"] = report["psi"].apply(interpret_psi)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("\nPSI Monitoring Report:")
    print(report.to_string(index=False))

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
