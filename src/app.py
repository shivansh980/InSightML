import streamlit as st
import pandas as pd
import os

from model_loader import ModelLoaderFactory
from shap_explainer import ShapExplainerContext
from report_generator import PDFReportGenerator

# Streamlit App Title
st.title("📊 InSightML - Model Interpretability Tool")

# Upload Model File
model_file = st.file_uploader("Upload your model file (Pickle or Keras)", type=["pkl", "h5"])
data_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if model_file and data_file:
    # Save model file locally
    model_path = os.path.join("temp_models", model_file.name)
    os.makedirs("temp_models", exist_ok=True)  # Ensure directory exists
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())

    # Save dataset locally
    data_path = os.path.join("temp_data", data_file.name)
    os.makedirs("temp_data", exist_ok=True)  # Ensure directory exists
    with open(data_path, "wb") as f:
        f.write(data_file.getbuffer())

    # Load dataset
    data = pd.read_csv(data_path)
    st.write("📂 Dataset Preview:", data.head())

    # Load Model
    loader = ModelLoaderFactory.get_model_loader(model_path)
    model = loader.load_model(model_path)  # Now we pass a valid file path

    # Generate SHAP Explanations
    explainer = ShapExplainerContext(model, data)
    shap_values = explainer.explain_model()
    
    # Generate Report
    report_path = "output_report.pdf"
    report = PDFReportGenerator(model, data, shap_values, output_path=report_path)
    report.generate_pdf_report()

    report_path = "reports/output_report.pdf"
    # Provide download link
    with open(report_path, "rb") as file:
        st.download_button("📥 Download Report", file, file_name="Model_Report.pdf")

st.sidebar.write("🔍 InSightML helps analyze ML models using SHAP and generate reports.")
