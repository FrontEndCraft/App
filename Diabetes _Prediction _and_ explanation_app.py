import streamlit as st
from joblib import load
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import numpy as np
import shap 
import streamlit.components.v1 as components


# Load the dataset
# Load the model
model = load(r'C:\Users\HP\Downloads\zip_file\best_xgb.joblib')
Diabetes = pd.read_csv(r"C:\Users\HP\Downloads\zip_file\diabetes_prediction_dataset.csv")
scaler = load(r"C:\Users\HP\Downloads\zip_file\scaler.joblib")
image =r"C:\Users\HP\Downloads\zip_file\Diabetespicture.jpg"

# # # Path to your project files (relative paths are advised)
# MODEL_PATH = './models/best_xgb.joblib'
# DATA_PATH = './data/diabetes_prediction_dataset.csv'
# SCALER_PATH = './scalers/scaler.joblib'
# IMAGE_PATH = './images/Diabetespicture.jpg'

# @st.cache(allow_output_mutation=True)
# def load_resources():
#     model = load(MODEL_PATH)
#     Diabetes = pd.read_csv(DATA_PATH)
#     scaler = load(SCALER_PATH)
#     return model, Diabetes, scaler
# model, scaler, Diabetes = load_resources()

# Preprocessing Functions
def encode_features(df):
    encoder = LabelEncoder()
    for col in ['gender', 'smoking_history']:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))
    return df
    
# Apply encoding
Diabetes_encoded = encode_features(Diabetes.copy())
X = Diabetes_encoded.iloc[:, :-1]
y = Diabetes_encoded.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SHAP Explainer
explainer_shap = shap.TreeExplainer(model)
def generate_shap_explanation(data_for_prediction):
    scaled_data = scaler.transform([data_for_prediction])
    shap_values = explainer_shap.shap_values(scaled_data)
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    expected_value = explainer_shap.expected_value if np.isscalar(explainer_shap.expected_value) else explainer_shap.expected_value[1]
    
    # Generate the force plot
    shap_plot = shap.force_plot(expected_value, shap_values[0], scaled_data[0], feature_names=X_train.columns.tolist(), show=False)
    
    # Convert the plot to HTML
    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
    
    # Use Streamlit components to render HTML
    components.html(shap_html, height=300)

# LIME Explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=['Non-Diabetic', 'Diabetic'],
    kernel_width=5,
    discretize_continuous=True,
    mode='classification',
    random_state=42
)

def generate_lime_explanation(data_for_prediction):
    explanation = explainer.explain_instance(data_for_prediction, model.predict_proba, num_features=10)
    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)

def predict(features):
    # Ensure the features are in the correct order as the model expects
    feature_order = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    features_list = [features[feature] for feature in feature_order]  # Convert dictionary to list in the correct order
    
    # Convert to 2D array-like structure as scaler and model expect
    scaled_features = scaler.transform([features_list])
    prediction_proba = model.predict_proba(scaled_features)[0]
    prediction = model.predict(scaled_features)[0]
    return prediction, prediction_proba


def collect_user_inputs():
    return {
        'gender': st.sidebar.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female'),
        'age': st.sidebar.slider('Age', min_value=1, max_value=100, value=50),
        'hypertension': st.sidebar.selectbox('Hypertension', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes'),
        'heart_disease': st.sidebar.selectbox('Heart Disease', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes'),
        'smoking_history': st.sidebar.selectbox('Smoking History', options=[0, 1, 2, 3, 4], format_func=lambda x: ['Never', 'Unknown', 'Current', 'Former', 'Ever'][x]),
        'bmi': st.sidebar.number_input('BMI', value=27.32, format="%.2f"),
        'HbA1c_level': st.sidebar.number_input('HbA1c Level', value=5.53, format="%.2f"),
        'blood_glucose_level': st.sidebar.number_input('Blood Glucose Level', value=138, format="%d")
    }

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app section", ["Home", "Predict and Explain Diabetes Risk"])
    
    if app_mode == "Home":
        st.title("Welcome to the Diabetes Risk Prediction and Explanation App")
        st.image(image, caption="Exploring Diabetes Health and Wellness", use_column_width=True)
    elif app_mode == "Predict and Explain Diabetes Risk":
        st.title("Predict and Explain Diabetes Risk")
        user_inputs = collect_user_inputs()
        
        if st.button('Predict and Explain Risk'):
            prediction, prediction_proba = predict(user_inputs)
            predicted_class = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
            st.success(f'Prediction: {predicted_class} (Probability: {prediction_proba[1]:.2f})')
            
            st.write("Generating SHAP Explanation...")
            generate_shap_explanation(list(user_inputs.values()))
            
            st.write("Generating LIME Explanation...")
            generate_lime_explanation(np.array(list(user_inputs.values())).astype(float))

if __name__ == '__main__':
    main()
