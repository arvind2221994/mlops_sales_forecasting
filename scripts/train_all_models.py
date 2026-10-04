import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_JOBS = (
    ("Decision Tree", "train_decision_tree.py", "decision_tree_model.joblib"),
    ("Bagging", "train_bagging.py", "bagging_model.joblib"),
    ("Random Forest", "train_random_forest.py", "random_forest_model.joblib"),
    ("AdaBoost", "train_adaboost.py", "adaboost_model.joblib"),
    (
        "Gradient Boosting",
        "train_gradient_boosting.py",
        "gradient_boosting_model.joblib",
    ),
    ("XGBoost", "train_xgboost.py", "xgboost_model.joblib"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train all sales forecasting models in sequence."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run compare_models.py after every model trains successfully.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)

    failures = []
    completed_artifacts = []

    print(f"Model artifacts will be saved in: {MODELS_DIR}")

    for index, (model_name, script_name, artifact_name) in enumerate(
        TRAINING_JOBS, start=1
    ):
        script_path = SCRIPTS_DIR / script_name
        artifact_path = MODELS_DIR / artifact_name

        print(f"\n[{index}/{len(TRAINING_JOBS)}] Training {model_name}")
        print(f"Running: {sys.executable} {script_path}")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode != 0:
            failures.append(
                (model_name, f"training exited with code {result.returncode}")
            )
            continue

        if not artifact_path.is_file():
            failures.append(
                (model_name, f"expected artifact was not created: {artifact_path}")
            )
            continue

        completed_artifacts.append((model_name, artifact_path))
        print(f"Saved {model_name} artifact: {artifact_path}")

    print("\n=== Training Summary ===")
    for model_name, artifact_path in completed_artifacts:
        print(f"[SUCCESS] {model_name}: {artifact_path}")
    for model_name, reason in failures:
        print(f"[FAILED]  {model_name}: {reason}")

    if failures:
        print(
            f"\n{len(failures)} model(s) failed. Comparison was not started."
        )
        return 1

    print(f"\nAll {len(completed_artifacts)} models trained successfully.")

    if not args.compare:
        print("Run comparison with: python scripts\\compare_models.py")
        return 0

    print("\n=== Comparative Analysis ===")
    comparison = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "compare_models.py")],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return comparison.returncode


if __name__ == "__main__":
    raise SystemExit(main())
