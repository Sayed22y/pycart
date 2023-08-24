import pandas as pd
import streamlit as st
from pycaret.classification import *





def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None
    
    
def perform_eda(data):
    try:
        setup(data, target='target_variable', silent=True)
        st.success("Exploratory Data Analysis completed.")
    except Exception as e:
        st.error(f"An error occurred during EDA: {e}")

def train_models():
    try:
        compare_models()
        st.success("Model training completed.")
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")

def deploy_model(data, selected_models):
    try:
        model = create_model(selected_models)
        st.header("Model Deployment")
        st.write("Deployed model:", model)
        st.write("Sample prediction:", predict_model(model, data.head(1)))
    except Exception as e:
        st.error(f"An error occurred during model deployment: {e}")
def main():
    st.title("Machine Learning Package Demo")

    # Upload data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        if data is not None:
            # Perform EDA
            st.header("Exploratory Data Analysis")
            perform_eda(data)

            # Train models
            if st.button("Train Models"):
                st.header("Model Training")
                train_models()

            # Model deployment
            if st.button("Deploy Model"):
                selected_models = st.multiselect("Select models for deployment", get_all_models())
                if selected_models:
                    deploy_model(data, selected_models)
