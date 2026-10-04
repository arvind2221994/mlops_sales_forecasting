import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

from dotenv import load_dotenv
from huggingface_hub import HfApi, login
import joblib
import pandas as pd
from utils import evaluate_regression, load_data_from_hf

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
BEST_MODEL_REPO = f"{HF_USERNAME}/superkart-best-sales-model"

model_files = {
    "Decision Tree": MODELS_DIR / "decision_tree_model.joblib",
    "Bagging": MODELS_DIR / "bagging_model.joblib",
    "Random Forest": MODELS_DIR / "random_forest_model.joblib",
    "AdaBoost": MODELS_DIR / "adaboost_model.joblib",
    "Gradient Boosting": MODELS_DIR / "gradient_boosting_model.joblib",
    "XGBoost": MODELS_DIR / "xgboost_model.joblib",
}

if not any(path.is_file() for path in model_files.values()):
    training_commands = "\n".join(
        f"  python scripts/{script_name}"
        for script_name in (
            "train_decision_tree.py",
            "train_bagging.py",
            "train_random_forest.py",
            "train_adaboost.py",
            "train_gradient_boosting.py",
            "train_xgboost.py",
        )
    )
    raise FileNotFoundError(
        f"No trained model artifacts were found in {MODELS_DIR}.\n"
        "Train at least one model before running the comparison:\n"
        f"{training_commands}"
    )

login(token=HF_TOKEN)

train_df, test_df = load_data_from_hf()
target_col = "Product_Store_Sales_Total"
drop_cols = ["Product_Id", "Store_Id", target_col]

X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
y_test = test_df[target_col]

results = []

print("--- Comparative Analysis ---")
for name, path in model_files.items():
    if not path.is_file():
        print(f"Skipping {name}: Artifact not found at {path}")
        continue

    model = joblib.load(path)

    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    metrics["Model"] = name
    results.append(metrics)

comparison_df = pd.DataFrame(results)[["Model", "RMSE", "MAE", "R2"]]
comparison_df = comparison_df.sort_values(by="RMSE", ascending=True)

print("\n" + comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]["Model"]
print(
    f"\nFinal Best Model Selected: {best_model_name} (Lowest RMSE: {comparison_df.iloc[0]['RMSE']:.2f})"
)

REPORTS_DIR.mkdir(exist_ok=True)
comparison_csv = REPORTS_DIR / "model_comparison.csv"
comparison_report = REPORTS_DIR / "model_training_summary.md"
comparison_df.to_csv(comparison_csv, index=False)

report_lines = [
    "# Automated Model Comparison",
    "",
    "| Rank | Model | RMSE | MAE | R2 |",
    "|---:|---|---:|---:|---:|",
]
for rank, row in enumerate(comparison_df.itertuples(index=False), start=1):
    report_lines.append(
        f"| {rank} | {row.Model} | {row.RMSE:.6f} | "
        f"{row.MAE:.6f} | {row.R2:.6f} |"
    )
report_lines.extend(
    [
        "",
        "## Selected Model",
        "",
        f"**{best_model_name}** was selected because it achieved the lowest "
        f"RMSE ({comparison_df.iloc[0]['RMSE']:.6f}).",
        "",
    ]
)
comparison_report.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Comparison CSV saved to: {comparison_csv}")
print(f"Comparison report saved to: {comparison_report}")

best_path = model_files[best_model_name]
api = HfApi()
api.create_repo(repo_id=BEST_MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj=str(best_path),
    path_in_repo="model.joblib",
    repo_id=BEST_MODEL_REPO,
    repo_type="model",
)
print(
    f"Best model ({best_model_name}) successfully registered at: https://huggingface.co/{BEST_MODEL_REPO}"
)