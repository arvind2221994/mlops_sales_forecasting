import os
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_NAME = os.getenv("HF_DATASET_NAME", "superkart-sales-forecasting")
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

DATA_DIR = "./data"
CSV_FILE_PATH = os.path.join(DATA_DIR, "SuperKart.csv")


def register_dataset_to_hf():
    if not HF_TOKEN or not HF_USERNAME:
        raise ValueError(
            "HF_TOKEN or HF_USERNAME not found in environment/.env file."
        )

    login(token=HF_TOKEN)

    print(f"Loading data from {CSV_FILE_PATH}...")
    df = pd.read_csv(CSV_FILE_PATH)

    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({"train": dataset})

    print(f"Pushing dataset to Hugging Face Hub: {REPO_ID}...")
    dataset_dict.push_to_hub(repo_id=REPO_ID, private=False)

    print(
        f"Data Registration Complete! Access your dataset at: https://huggingface.co/datasets/{REPO_ID}"
    )


if __name__ == "__main__":
    register_dataset_to_hf()