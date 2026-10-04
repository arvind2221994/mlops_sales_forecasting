# Migrating the SuperKart App to Streamlit Community Cloud

## Overview

The SuperKart Sales Revenue Predictor was moved from a planned Hugging Face
Space deployment to Streamlit Community Cloud. The application continues to
store and retrieve its trained Random Forest model from Hugging Face Model Hub;
only the user-interface hosting platform changed.

The deployed application is available at:

[SuperKart Sales Predictor](https://mlopssalesforecasting-sfvvmx7aozxtwfgjuwcv4f.streamlit.app/)

GitHub repository creation and source-control setup are documented separately in
`GITHUB_SETUP.md` and are intentionally excluded from this guide.

## Architecture

```text
User
  |
  v
Streamlit Community Cloud
  |
  | Downloads model.joblib
  v
Hugging Face Model Hub
arvind2221994/superkart-best-sales-model
```

Streamlit Community Cloud runs the interface in `deployment/app.py`. During
startup, the application downloads `model.joblib` from the public Hugging Face
model repository, loads the fitted pipeline, and caches it for subsequent
predictions.

## Why the Deployment Target Changed

The original deployment script attempted to create a Hugging Face Space with:

```python
space_sdk="streamlit"
```

Hugging Face no longer accepts `streamlit` as a Space SDK. The supported SDK
values are `gradio`, `docker`, and `static`. Docker Spaces require a paid plan,
while a static Space cannot run Python, Streamlit, or a scikit-learn model.

Streamlit Community Cloud was therefore selected because it runs the existing
Streamlit application directly and does not require a Docker image or a rewrite
to Gradio.

## Files Used by Streamlit Community Cloud

| File | Purpose |
|---|---|
| `deployment/app.py` | Defines the user interface, downloads the model, and generates predictions. |
| `deployment/requirements.txt` | Lists the Python packages installed during deployment. |
| `deployment/deploy_to_hf.py` | Legacy Hugging Face Space deployment script; it is not used by Streamlit Community Cloud. |

## Application Preparation

### 1. Keep the trained model in Hugging Face Model Hub

The comparison workflow selected Random Forest as the best model and uploaded
the artifact as:

```text
arvind2221994/superkart-best-sales-model/model.joblib
```

The model repository remains the source of truth for the production artifact.
Moving the interface to Streamlit Community Cloud did not require moving or
duplicating the model.

### 2. Configure the application to download the model

The Streamlit application constructs the model repository ID from the configured
Hugging Face username:

```python
HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
MODEL_REPO_ID = f"{HF_USERNAME}/superkart-best-sales-model"
```

It downloads the selected model with:

```python
model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename="model.joblib",
    repo_type="model",
)
```

Because the model repository is public, deployment does not require an
`HF_TOKEN`.

### 3. Cache the loaded model

The model-loading function uses Streamlit's resource cache:

```python
@st.cache_resource
def load_model():
    ...
```

This prevents the 40.9 MB model artifact from being downloaded and loaded again
on every Streamlit rerun.

### 4. Declare deployment dependencies

The deployment requirements include Streamlit, Hugging Face Hub, Pandas,
NumPy, joblib, scikit-learn, XGBoost, and python-dotenv. Streamlit Community
Cloud installs these packages automatically when it builds the application.

For maximum serialization compatibility, the runtime versions of `joblib`,
NumPy, Pandas, and scikit-learn should match the versions used to train and save
the model.

## Streamlit Community Cloud Configuration

After the source was available in GitHub, the following configuration was used
when creating the Streamlit application:

| Setting | Value |
|---|---|
| Repository | GitHub repository containing this project |
| Branch | `main` |
| Main file path | `deployment/app.py` |
| Python version | Python 3.12 |

The application was created from the Streamlit Community Cloud workspace:

```text
https://share.streamlit.io
```

## Environment Configuration

The application requires the Hugging Face username to construct the model
repository ID. In the Streamlit application settings, the following value was
added under **Advanced settings → Secrets**:

```toml
HF_USERNAME = "arvind2221994"
```

Tokens and `.env` files must not be committed to source control. If the model
repository becomes private later, `HF_TOKEN` should be added through Streamlit
secrets rather than placed in the application source.

## Build and Startup Process

When the application is deployed or restarted, Streamlit Community Cloud:

1. Checks out the configured GitHub branch.
2. Locates `deployment/app.py` as the entry point.
3. Installs packages from `deployment/requirements.txt`.
4. Starts the Streamlit server.
5. Reads `HF_USERNAME` from the deployment environment.
6. Downloads `model.joblib` from Hugging Face Model Hub.
7. Loads and caches the fitted Random Forest pipeline.
8. Renders the product and store input form.

No local `docker build` command or Hugging Face Space creation script is needed.

## Deployment Verification

The deployment was considered successful after verifying the following:

- The public Streamlit URL opened successfully.
- The SuperKart prediction form rendered.
- The application displayed **Model loaded successfully from Hugging Face
  Hub!**
- Product and store inputs were available.
- The prediction button submitted the form.
- The application returned a numeric predicted sales total.

The model-loaded message confirms that the hosted application can reach
Hugging Face, download the artifact, and deserialize the fitted pipeline.

## Updating the Application

Application updates no longer use `deploy_to_hf.py`. Changes are deployed by
pushing updates to the configured GitHub branch. Streamlit Community Cloud
detects the new commit and rebuilds or reruns the application automatically.

After an update:

1. Open the Streamlit application.
2. Review the build and runtime logs if the app does not start.
3. Confirm that the model-loaded message appears.
4. Submit a representative prediction.
5. Verify that a numeric result is displayed.

## Troubleshooting

| Symptom | Likely Cause | Resolution |
|---|---|---|
| Model repository resolves to `your-hf-username` | `HF_USERNAME` is missing from Streamlit secrets. | Add `HF_USERNAME = "arvind2221994"` and reboot the app. |
| Model download returns `401` or `403` | The model is private or the deployment lacks access. | Add an authorized `HF_TOKEN` through Streamlit secrets. |
| `ModuleNotFoundError` during startup | A package is missing from `deployment/requirements.txt`. | Add the package and redeploy. |
| Warning or failure while loading `model.joblib` | Runtime library versions differ from training versions. | Pin joblib, NumPy, Pandas, and scikit-learn to the training versions. |
| App redirects visitors to authentication | The app is not publicly shared or workspace access is restricted. | Review the application sharing settings and test in a signed-out browser. |
| App sleeps after inactivity | Normal Community Cloud resource behavior. | Reopen the URL and allow the application to restart. |

## Result

The migration preserved the existing Streamlit interface and Hugging Face model
registry while replacing Hugging Face Spaces with Streamlit Community Cloud as
the application host. This avoided Docker and paid Hugging Face compute without
changing the model-training or model-comparison workflow.
