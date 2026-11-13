from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

from pd_model.models.predict import predict_default

app = FastAPI(
    title="Credit Default Prediction API",
    description="API для прогнозирования вероятности дефолта клиента",
    version="1.0.0",
)


class ClientData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    LIMIT_BAL: float = Field(ge=0)

    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)

    PAY_0: int = Field(ge=-2, le=8)
    PAY_2: int = Field(ge=-2, le=8)
    PAY_3: int = Field(ge=-2, le=8)
    PAY_4: int = Field(ge=-2, le=8)
    PAY_5: int = Field(ge=-2, le=8)
    PAY_6: int = Field(ge=-2, le=8)

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    default_probability: float


@app.get("/")
def root():
    return {"message": "Credit Default Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData):
    client_data = pd.DataFrame([client.model_dump()])

    result = predict_default(client_data)

    return {
        "prediction": result["predictions"][0],
        "default_probability": result["default_probabilities"][0],
    }
