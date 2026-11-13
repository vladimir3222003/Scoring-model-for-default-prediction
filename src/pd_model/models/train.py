from pathlib import Path
import json
import joblib

import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

from pd_model.models.pipeline import create_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TARGET = "default.payment.next.month"

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "AGE_GROUP"]


def load_split_data():
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Файл не найден: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Файл не найден: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return metrics, y_proba


def log_search_candidates(grid_search, model_name):
    params_list = grid_search.cv_results_["params"]
    mean_scores = grid_search.cv_results_["mean_test_score"]
    std_scores = grid_search.cv_results_["std_test_score"]

    for i, params in enumerate(params_list):
        with mlflow.start_run(run_name=f"{model_name}_candidate_{i + 1}"):
            mlflow.set_tag("model_type", model_name)
            mlflow.log_params({key: str(value) for key, value in params.items()})

            mlflow.log_metric("cv_roc_auc_mean", float(mean_scores[i]))

            mlflow.log_metric("cv_roc_auc_std", float(std_scores[i]))


def log_best_model(
    model, model_name, best_params, cv_score, metrics, roc_path, comparison_path
):
    with mlflow.start_run(run_name=f"{model_name}_best"):
        mlflow.set_tag("model_type", model_name)

        mlflow.log_params({key: str(value) for key, value in best_params.items()})

        mlflow.log_metric("cv_roc_auc", float(cv_score))

        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

        # Сохраняем модель в MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

        # Сохраняем графики и таблицу
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(comparison_path))


def train_model():

    # Подключаем через sqlite
    mlflow_db_path = PROJECT_ROOT / "mlflow.db"

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.as_posix()}")

    mlflow.set_experiment("Credit_Default_Prediction")

    train_df, test_df = load_split_data()

    X_train = train_df.drop(columns=[TARGET, "ID"])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET, "ID"])
    y_test = test_df[TARGET]

    numerical_features = [
        column for column in X_train.columns if column not in CATEGORICAL_FEATURES
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # логистическая регрессия

    lr_pipeline = create_pipeline(
        numerical_features, CATEGORICAL_FEATURES, model_type="logistic_regression"
    )

    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"],
    }

    lr_search = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=lr_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    print("\nОбучение Logistic Regression...")

    lr_search.fit(X_train, y_train)

    best_lr = lr_search.best_estimator_

    lr_metrics, lr_proba = calculate_metrics(best_lr, X_test, y_test)

    log_search_candidates(lr_search, "logistic_regression")

    # градиент бустинг

    gb_pipeline = create_pipeline(
        numerical_features, CATEGORICAL_FEATURES, model_type="gradient_boosting"
    )

    gb_param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [2, 3],
    }

    gb_search = GridSearchCV(
        estimator=gb_pipeline,
        param_grid=gb_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    # взвешиваем классы из-за дисбаланса
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    print("\nОбучение Gradient Boosting...")

    gb_search.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    best_gb = gb_search.best_estimator_

    gb_metrics, gb_proba = calculate_metrics(best_gb, X_test, y_test)

    log_search_candidates(gb_search, "gradient_boosting")

    # сравниваем

    results = pd.DataFrame(
        [lr_metrics, gb_metrics], index=["Logistic Regression", "Gradient Boosting"]
    )

    print("\nЛучшие параметры Logistic Regression:")
    print(lr_search.best_params_)

    print("\nЛучший CV ROC-AUC Logistic Regression:")
    print(lr_search.best_score_)

    print("\nЛучшие параметры Gradient Boosting:")
    print(gb_search.best_params_)

    print("\nЛучший CV ROC-AUC Gradient Boosting:")
    print(gb_search.best_score_)

    print("\nСравнение на тестовой выборке:")
    print(results.round(4))

    # ROC

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_proba)

    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)

    roc_path = figures_dir / "roc_curve_comparison.png"
    comparison_path = reports_dir / "model_comparison.csv"

    plt.figure(figsize=(8, 6))

    plt.plot(
        lr_fpr,
        lr_tpr,
        label=f"Logistic Regression " f"(AUC = {lr_metrics['roc_auc']:.4f})",
    )

    plt.plot(
        gb_fpr,
        gb_tpr,
        label=f"Gradient Boosting " f"(AUC = {gb_metrics['roc_auc']:.4f})",
    )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Credit Default Prediction")
    plt.legend()
    plt.grid()

    plt.savefig(roc_path, dpi=300, bbox_inches="tight")

    plt.close()

    results.to_csv(comparison_path)

    # сохраняем

    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"

    metrics = {
        "lr_roc_auc": float(lr_metrics["roc_auc"]),
        "lr_precision": float(lr_metrics["precision"]),
        "lr_recall": float(lr_metrics["recall"]),
        "lr_f1": float(lr_metrics["f1"]),
        "gb_roc_auc": float(gb_metrics["roc_auc"]),
        "gb_precision": float(gb_metrics["precision"]),
        "gb_recall": float(gb_metrics["recall"]),
        "gb_f1": float(gb_metrics["f1"]),
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    lr_path = models_dir / "pd_model.joblib"
    gb_path = models_dir / "pd_model_gradient_boosting.joblib"

    joblib.dump(best_lr, lr_path)
    joblib.dump(best_gb, gb_path)

    log_best_model(
        best_lr,
        "logistic_regression",
        lr_search.best_params_,
        lr_search.best_score_,
        lr_metrics,
        roc_path,
        comparison_path,
    )

    log_best_model(
        best_gb,
        "gradient_boosting",
        gb_search.best_params_,
        gb_search.best_score_,
        gb_metrics,
        roc_path,
        comparison_path,
    )

    print("\nМодели сохранены:")
    print(lr_path)
    print(gb_path)

    print(f"\nROC-кривая: {roc_path}")
    print(f"Сравнение моделей: {comparison_path}")
    print(f"Метрики моделей: {metrics_path}")
    return best_lr, best_gb, results


if __name__ == "__main__":
    train_model()
