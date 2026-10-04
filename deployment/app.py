import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
import streamlit as st

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
MODEL_REPO_ID = f"{HF_USERNAME}/superkart-best-sales-model"

st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 SuperKart Sales Revenue Predictor")
st.markdown("Enter product and store details below to predict total sales revenue.")


@st.cache_resource
def load_model():
    """Downloads and caches the trained model from Hugging Face Model Hub."""
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID, filename="model.joblib", repo_type="model"
    )
    return joblib.load(model_path)


try:
    model = load_model()
    st.success("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input Form
with st.form("prediction_form"):
    st.subheader("Product Details")
    col1, col2 = st.columns(2)

    with col1:
        product_mrp = st.number_input(
            "Product MRP ($)", min_value=0.0, value=150.0, step=1.0
        )
        product_weight = st.number_input(
            "Product Weight", min_value=0.0, value=12.5, step=0.1
        )
        product_sugar = st.selectbox(
            "Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
        )

    with col2:
        product_area = st.number_input(
            "Allocated Display Area Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.005,
        )
        product_type = st.selectbox(
            "Product Category",
            [
                "Dairy",
                "Soft Drinks",
                "Meat",
                "Fruits and Vegetables",
                "Household",
                "Snack Foods",
                "Baking Goods",
                "Frozen Foods",
                "Canned",
                "Health and Hygiene",
            ],
        )

    st.subheader("Store Details")
    col3, col4 = st.columns(2)

    with col3:
        store_year = st.number_input(
            "Establishment Year",
            min_value=1900,
            max_value=2026,
            value=1999,
            step=1,
        )
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])

    with col4:
        store_city = st.selectbox(
            "City Tier", ["Tier 1", "Tier 2", "Tier 3"]
        )
        store_type = st.selectbox(
            "Store Type",
            [
                "Supermarket Type 1",
                "Supermarket Type 2",
                "Departmental Store",
                "Food Mart",
            ],
        )

    submit_button = st.form_submit_button("Predict Sales Total")

if submit_button:
    input_data = pd.DataFrame(
        [
            {
                "Product_Weight": product_weight,
                "Product_Sugar_Content": product_sugar,
                "Product_Allocated_Area": product_area,
                "Product_Type": product_type,
                "Product_MRP": product_mrp,
                "Store_Establishment_Year": store_year,
                "Store_Size": store_size,
                "Store_Location_City_Type": store_city,
                "Store_Type": store_type,
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.metric(
        label="Predicted Store Sales Total",
        value=f"${round(float(prediction), 2):,}",
    )