import os
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import train_test_split

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_NAME = os.getenv("HF_DATASET_NAME", "superkart-sales-forecasting")
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

PROCESSED_DATA_DIR = "./data/processed"


def prepare_data():
    if not HF_TOKEN or not HF_USERNAME:
        raise ValueError(
            "HF_TOKEN or HF_USERNAME not found in environment/.env file."
        )

    login(token=HF_TOKEN)

    print(f"Loading raw dataset from Hugging Face space: {REPO_ID}...")
    hf_dataset = load_dataset(REPO_ID, split="train")
    df = hf_dataset.to_pandas()

    # Data Cleaning & Preprocessing
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace(
            {"low sugar": "Low Sugar", "reg": "Regular", "no sugar": "No Sugar"}
        )

    if "Product_Weight" in df.columns:
        df["Product_Weight"] = df["Product_Weight"].fillna(
            df["Product_Weight"].median()
        )

    if "Store_Size" in df.columns:
        df["Store_Size"] = df["Store_Size"].fillna("Unknown")

    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])

    # Train/Test Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save locally
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Push updated splits to Hugging Face
    processed_dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    processed_dataset_dict.push_to_hub(repo_id=REPO_ID, private=False)

    print(f"Data Preparation Complete! Dataset space: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    prepare_data()