import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"

from dotenv import load_dotenv
from huggingface_hub import HfApi, login
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from utils import build_preprocessor, evaluate_regression, load_data_from_hf

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
MODEL_REPO_ID = f"{HF_USERNAME}/superkart-gradient-boosting"

login(token=HF_TOKEN)

train_df, test_df = load_data_from_hf()
target_col = "Product_Store_Sales_Total"
drop_cols = ["Product_Id", "Store_Id", target_col]

X_train = train_df.drop(
    columns=[c for c in drop_cols if c in train_df.columns]
)
y_train = train_df[target_col]
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
y_test = test_df[target_col]

preprocessor = build_preprocessor(X_train)
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42)),
    ]
)

param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__learning_rate": [0.03, 0.1],
    "regressor__max_depth": [3, 5],
}

print("Tuning Gradient Boosting...")
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
metrics = evaluate_regression(y_test, y_pred)
print(f"Gradient Boosting Metrics: {metrics}")

MODELS_DIR.mkdir(exist_ok=True)
model_path = MODELS_DIR / "gradient_boosting_model.joblib"
joblib.dump(best_model, model_path)

api = HfApi()
api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj=str(model_path),
    path_in_repo="model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)
print(f"Model successfully registered at: https://huggingface.co/{MODEL_REPO_ID}")