import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
SPACE_NAME = os.getenv("HF_SPACE_NAME", "superkart-sales-forecasting-ui")
SPACE_REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

DEPLOYMENT_DIR = "./deployment"


def deploy_to_huggingface_space():
    if not HF_TOKEN or not HF_USERNAME:
        raise ValueError(
            "HF_TOKEN or HF_USERNAME missing from environment/.env file."
        )

    login(token=HF_TOKEN)
    api = HfApi()

    # Create Streamlit Space on Hugging Face (Works on Free Tier)
    print(f"Creating Hugging Face Streamlit Space: {SPACE_REPO_ID}...")
    api.create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="streamlit",  # Uses Streamlit engine instead of Docker
        exist_ok=True,
        private=False,
    )

    # Upload files
    print(f"Uploading files from {DEPLOYMENT_DIR} to HF Space...")
    api.upload_folder(
        folder_path=DEPLOYMENT_DIR,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        ignore_patterns=["deploy_to_hf.py", "Dockerfile", "__pycache__/*"],
    )

    print("\n" + "=" * 60)
    print("Deployment files successfully pushed!")
    print(
        f"Access your live application at: https://huggingface.co/spaces/{SPACE_REPO_ID}"
    )
    print("=" * 60)


if __name__ == "__main__":
    deploy_to_huggingface_space()