from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def create_pipeline(numerical_features, categorical_features, model_type):

    if model_type == "logistic_regression":
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    elif model_type == "gradient_boosting":
        numerical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

    else:
        raise ValueError(f"Неизвестная модель: {model_type}")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=(model_type == "logistic_regression"),
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, random_state=42)

    else:
        classifier = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    return pipeline
