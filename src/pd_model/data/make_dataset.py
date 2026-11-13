from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from pd_model.data.validation import validate_data
from pd_model.features.build_features import create_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TARGET = "default.payment.next.month"


def load_raw_data():
    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "credit_card_default.csv"

    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"Используем локальный датасет: {df.shape}")
        return df

    dataset_path = kagglehub.dataset_download(
        "uciml/default-of-credit-card-clients-dataset"
    )

    csv_files = list(Path(dataset_path).rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("CSV-файл не найден в датасете Kaggle")

    df = pd.read_csv(csv_files[0])
    df.to_csv(output_path, index=False)

    print(f"Датасет загружен: {df.shape}")

    return df


def make_dataset():

    df = load_raw_data()

    df = validate_data(df)

    df = create_features(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Файлы сохранены в: {output_dir}")

    return train_df, test_df


if __name__ == "__main__":
    make_dataset()
