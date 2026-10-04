# SuperKart Sales Forecasting: Model Training Report

## Motivation

When selecting the best model for predicting sales revenue
(`Product_Store_Sales_Total`), we must evaluate how tree-based algorithms and
ensemble techniques handle the specific tabular characteristics of retail sales
datasets.

## Model Comparison

| Model | Theoretical Suitability | Strengths for SuperKart Sales Data | Limitations / Risk Factors |
|---|---|---|---|
| **Decision Tree** | Low (Baseline) | Simple, non-parametric, and requires minimal preprocessing for non-linear relations. | High risk of overfitting; highly sensitive to noise and small variations in retail features. |
| **Bagging** | Moderate | Reduces the variance of complex decision trees by averaging over bootstrap samples. | Does not address bias; tree models trained in parallel remain correlated. |
| **Random Forest** | High | De-correlates trees by selecting random feature subsets at each split; robust to noise and outliers. | Can be computationally heavier; may struggle with extreme target extrapolation. |
| **AdaBoost** | Moderate–High | Sequentially focuses on hard-to-predict store and product instances by boosting weights. | Highly sensitive to noisy targets or outliers in sales data; risk of overfitting noisy data. |
| **Gradient Boosting (GBM)** | High | Minimizes loss functions through gradient descent; excels at capturing subtle, non-linear interactions. | Requires careful hyperparameter tuning (`learning_rate`, `n_estimators`, and `max_depth`) to prevent overfitting. |
| **XGBoost** | **Highest (Recommended)** | Uses regularization ($L_1$/$L_2$) to prevent overfitting, handles missing values naturally, and uses second-order gradients. | Requires hyperparameter optimization for the best performance. |


## Overview

This report documents the training and comparison of six regression models for
predicting `Product_Store_Sales_Total`:

1. Decision Tree
2. Bagging
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. XGBoost

All six models trained successfully. Random Forest produced the lowest root mean
squared error (RMSE) and was selected as the final model.

## Training Process

The training workflow performs the following steps for each model:

1. Loads the train and test splits from the
   [SuperKart Hugging Face dataset](https://huggingface.co/datasets/arvind2221994/superkart-sales-forecasting).
2. Removes `Product_Id` and `Store_Id` from the model features and uses
   `Product_Store_Sales_Total` as the prediction target.
3. Builds a preprocessing and regression pipeline.
4. Uses `GridSearchCV` with five-fold cross-validation and negative RMSE scoring
   to select the best hyperparameters.
5. Evaluates the best estimator against the test split using RMSE, mean absolute
   error (MAE), and coefficient of determination (R2).
6. Saves the fitted pipeline as a `.joblib` artifact in the shared `models`
   directory.
7. Registers the trained model in its corresponding Hugging Face model
   repository.

The shared model directory is:

```text
C:\Users\arvnarayanan\Documents\mlops_sales_forecasting\models
```

## Commands

### Activate the Python environment

Run the following command from the project root:

```powershell
.\.venv-x64\Scripts\Activate.ps1
```

### Train all models

```powershell
python scripts\train_all_models.py
```

This command trains every model sequentially and verifies that all six model
artifacts were created.

### Train all models and compare automatically

```powershell
python scripts\train_all_models.py --compare
```

The `--compare` option runs the comparative analysis after all models train
successfully.

### Compare existing model artifacts

```powershell
python scripts\compare_models.py
```

This command loads the saved model artifacts, evaluates them on the same test
split, ranks them by RMSE, and registers the best model on Hugging Face.

## Evaluation Metrics

| Metric | Selection Direction | Interpretation |
|---|---|---|
| RMSE | Lower is better | Penalizes larger prediction errors more heavily. |
| MAE | Lower is better | Measures the average absolute prediction error. |
| R2 | Higher is better | Measures the proportion of target variance explained by the model. |

## Comparison Results

| Rank | Model | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|
| 1 | **Random Forest** | **280.841298** | **109.730805** | **0.930876** |
| 2 | Bagging | 284.309950 | 111.844678 | 0.929158 |
| 3 | Gradient Boosting | 284.661536 | 123.146624 | 0.928983 |
| 4 | XGBoost | 285.973916 | 116.279626 | 0.928326 |
| 5 | Decision Tree | 337.833003 | 153.840066 | 0.899974 |
| 6 | AdaBoost | 449.957095 | 346.419979 | 0.822561 |

## Final Model Selection

**Random Forest was selected as the final model.**

It achieved:

- **RMSE:** 280.841298
- **MAE:** 109.730805
- **R2:** 0.930876

Random Forest ranked first on all three reported metrics. It had the lowest RMSE
and MAE while achieving the highest R2. The comparison process uses the lowest
RMSE as the primary selection criterion.

The final model was registered at:

[SuperKart Best Sales Model](https://huggingface.co/arvind2221994/superkart-best-sales-model)

## Generated Model Artifacts

| Model | Local Artifact | Hugging Face Repository |
|---|---|---|
| Decision Tree | `models\decision_tree_model.joblib` | [superkart-decision-tree](https://huggingface.co/arvind2221994/superkart-decision-tree) |
| Bagging | `models\bagging_model.joblib` | [superkart-bagging](https://huggingface.co/arvind2221994/superkart-bagging) |
| Random Forest | `models\random_forest_model.joblib` | [superkart-random-forest](https://huggingface.co/arvind2221994/superkart-random-forest) |
| AdaBoost | `models\adaboost_model.joblib` | [superkart-adaboost](https://huggingface.co/arvind2221994/superkart-adaboost) |
| Gradient Boosting | `models\gradient_boosting_model.joblib` | [superkart-gradient-boosting](https://huggingface.co/arvind2221994/superkart-gradient-boosting) |
| XGBoost | `models\xgboost_model.joblib` | [superkart-xgboost](https://huggingface.co/arvind2221994/superkart-xgboost) |

The selected Random Forest artifact was also uploaded as `model.joblib` to the
final best-model repository.
