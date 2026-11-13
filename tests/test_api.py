from fastapi.testclient import TestClient

from pd_model.api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict(monkeypatch):
    def mock_predict_default(df):
        return {"predictions": [1], "default_probabilities": [0.6449]}

    monkeypatch.setattr("pd_model.api.app.predict_default", mock_predict_default)

    payload = {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["default_probability"] <= 1


def test_predict_invalid_data():
    payload = {"LIMIT_BAL": -100, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
