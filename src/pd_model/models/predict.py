from functools import lru_cache
from pathlib import Path

import joblib

from pd_model.features.build_features import create_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "pd_model_gradient_boosting.joblib"


@lru_cache(maxsize=1)
def load_model(model_path=str(MODEL_PATH)):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    return joblib.load(model_path)


def predict_default(df):
    model = load_model()

    # Создаём те же признаки, что и при обучении
    df = create_features(df)

    X = df.drop(columns=["default.payment.next.month", "ID"], errors="ignore")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "predictions": predictions.tolist(),
        "default_probabilities": probabilities.tolist(),
    }
