# Convert the Project Folder into a GitHub Repository

This guide is formatted for use in a Google Colab notebook. Add the explanatory
text to **Text cells** and commands to **Code cells**.

> Do not place GitHub tokens, passwords, `.env` contents, or other credentials
> directly in the notebook. In particular, do not commit the project's `.env`
> file.

## 1. Open the project folder

### Text cell

Change the notebook's working directory to the folder containing the project.
Unlike `!cd`, the `%cd` notebook command persists for later cells.

### Code cell

```python
%cd /content/mlops_sales_forecasting
```

Replace `/content/mlops_sales_forecasting` if the project is stored elsewhere.

## 2. Update `.gitignore`

### Text cell

The project already ignores `.env`, Python cache files, data files, and MLflow
run data. Add `.venv-x64/` so the local Windows virtual environment is not
uploaded to GitHub.

### Code cell

```python
from pathlib import Path

gitignore = Path(".gitignore")
entry = ".venv-x64/"
current_rules = gitignore.read_text(encoding="utf-8").splitlines()

if entry not in current_rules:
    with gitignore.open("a", encoding="utf-8") as file:
        file.write("\n# Local virtual environment\n.venv-x64/\n")
```

## 3. Initialize the Git repository

### Text cell

Initialize Git inside the project folder and name the primary branch `main`.

### Code cell

```python
!git init
!git branch -M main
```

## 4. Configure the commit author

### Text cell

Set the name and email that should appear in the Git commit history. Use an
email associated with the GitHub account, or the account's GitHub-provided
`noreply` email address.

### Code cell

```python
!git config user.name "YOUR NAME"
!git config user.email "YOUR_GITHUB_EMAIL"
```

Replace the placeholder values before running the cell.

## 5. Confirm sensitive and generated files are ignored

### Text cell

Use `git check-ignore` to verify that the environment file and local virtual
environment will not be committed. Each successful check prints the ignored
path.

### Code cell

```python
!git check-ignore .env
!git check-ignore .venv-x64/
```

## 6. Stage and review the project files

### Text cell

Stage the project, then inspect the staged file list before committing.
Verify that `.env`, `.venv-x64/`, local data, and other secrets are absent.

### Code cell

```python
!git add .
!git status
```

If a sensitive file appears in the output, do not continue. Add it to
`.gitignore`, remove it from the staging area with
`!git restore --staged PATH`, and check the status again.

## 7. Create the first commit

### Text cell

Save the staged project files in the repository's initial commit.

### Code cell

```python
!git commit -m "Initial commit: sales forecasting MLOps project"
```

## 8. Install GitHub CLI

### Text cell

Install the GitHub CLI (`gh`) in the Colab runtime. Colab runtimes are
temporary, so this installation may need to be repeated when a new runtime is
created.

### Code cell

```python
!sudo apt-get update -qq
!sudo apt-get install -y gh
!gh --version
```

## 9. Log in to GitHub

### Text cell

Authenticate GitHub CLI using the browser-based login flow. The command
displays a one-time code and a GitHub URL. Open the URL, enter the code, approve
access, and then return to the notebook.

### Code cell

```python
!gh auth login --hostname github.com --git-protocol https --web
```

Confirm that authentication succeeded:

### Code cell

```python
!gh auth status
```

Authentication is stored only in the current Colab runtime. A new runtime may
require another login.

## 10. Create the public GitHub repository and push

### Text cell

Create a public repository named `mlops_sales_forecasting` under the logged-in
GitHub account. GitHub CLI also adds the new repository as the `origin` remote
and pushes the `main` branch.

### Code cell

```python
!gh repo create mlops_sales_forecasting --public --source=. --remote=origin --push
```

## 11. Verify the remote and push

### Text cell

Confirm that `origin` points to GitHub, the local branch tracks
`origin/main`, and no local changes remain.

### Code cell

```python
!git remote -v
!git branch -vv
!git status
!gh repo view
```

## 12. Push later changes

### Text cell

After editing the project, stage the changes, review them, create another
commit, and push it to GitHub.

### Code cell

```python
!git add .
!git status
!git commit -m "Describe the changes"
!git push
```

---

## GitHub Actions Workflow Setup

The following steps configure and activate the automated MLOps workflow after
the repository has been created and pushed to GitHub.

### 1. Verify the workflow file

GitHub Actions discovers workflows only inside `.github/workflows`. Confirm that
the pipeline exists at:

```text
.github/workflows/pipeline.yml
```

The pipeline validates pull requests and performs training, comparison, model
publication, artifact retention, and report generation on relevant updates to
`main`.

### 2. Configure GitHub Actions variables

Open the GitHub repository and go to:

```text
Settings → Secrets and variables → Actions → Variables
```

Add the following repository variables:

| Variable | Value |
|---|---|
| `HF_USERNAME` | `arvind2221994` |
| `HF_DATASET_NAME` | `superkart-sales-forecasting` |

These values are not credentials, so they can be stored as repository
variables.

### 3. Configure the Hugging Face secret

On the same GitHub page, select **Secrets**, and add:

| Secret | Value |
|---|---|
| `HF_TOKEN` | A Hugging Face access token with permission to update the dataset and model repositories |

Do not place the token in `pipeline.yml`, `.env.example`, documentation, or
source code. GitHub injects it into the training job at runtime.

### 4. Allow workflow write access

The pipeline commits generated CSV and Markdown comparison reports to `main`.
Go to:

```text
Settings → Actions → General → Workflow permissions
```

Select **Read and write permissions**, then save the setting. If branch
protection blocks direct pushes, either permit GitHub Actions to push generated
reports or change the reporting step to create a pull request.

### 5. Commit and push the workflow

From the project root, stage the workflow, report-generation change, and setup
documentation:

```powershell
git add .github\workflows\pipeline.yml
git add scripts\compare_models.py
git add GITHUB_SETUP.md
git status
git commit -m "Add automated MLOps training workflow"
git push origin main
```

After the push, open the repository's **Actions** tab and confirm that
**MLOps Training Pipeline** appears in the workflow list.

### 6. Start the MLOps pipeline

The workflow is stored at:

```text
.github/workflows/pipeline.yml
```

It runs automatically when relevant Python or dependency files are pushed to
`main`. Pull requests run validation only. To start training manually:

1. Open the repository's **Actions** tab.
2. Select **MLOps Training Pipeline**.
3. Select **Run workflow**.
4. Choose the `main` branch.
5. Select **Run workflow** again.

The training job uses the prepared Hugging Face train and test splits. It does
not run `prepare_data.py`, because repeatedly splitting the existing train split
would change and shrink the dataset.

### 7. Review pipeline outputs

After a successful run:

- Open the workflow run to review training logs.
- Download the `model-training-<run-number>` artifact for the generated model
  files and reports.
- Confirm that the individual and best-model Hugging Face repositories were
  updated.
- Review `reports/model_comparison.csv`.
- Review `reports/model_training_summary.md`.
- Confirm that the generated report commit was pushed to `main`.

If configuration validation fails, verify that `HF_TOKEN`, `HF_USERNAME`, and
`HF_DATASET_NAME` use the exact names shown above.
