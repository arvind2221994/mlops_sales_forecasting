import os
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "superkart-sales-forecasting")
DATASET_REPO_ID = f"{HF_USERNAME}/{HF_DATASET_NAME}"


def load_data_from_hf():
    """Loads train and test splits directly from Hugging Face Dataset Hub."""
    print(f"Loading train and test splits from HF Dataset: {DATASET_REPO_ID}")
    train_ds = load_dataset(DATASET_REPO_ID, split="train").to_pandas()
    test_ds = load_dataset(DATASET_REPO_ID, split="test").to_pandas()
    return train_ds, test_ds


def build_preprocessor(X):
    """Creates a scikit-learn ColumnTransformer for preprocessing."""
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    cat_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", fill_value="Unknown"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )
    return preprocessor


def evaluate_regression(y_true, y_pred):
    """Calculates RMSE, MAE, and R2 scores."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}