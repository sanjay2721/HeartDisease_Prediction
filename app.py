import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import lime
import lime.lime_tabular
import shap
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Explainable AI for Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin-bottom: 20px;
    }
    .risk-factor {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('logistic_regression_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        app_config = joblib.load('app_config.pkl')
        return lr_model, rf_model, app_config
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please run train_model.py first to generate the model files.")
        return None, None, None

lr_model, rf_model, app_config = load_models()

# App title
st.markdown('<h1 class="main-header">🫀 Explainable AI for Heart Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
This app predicts the likelihood of heart disease using machine learning models trained on the Cleveland dataset, 
with model interpretation using SHAP and LIME for explainable AI.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Random Forest"],
    help="Choose which machine learning model to use for predictions"
)

st.sidebar.header("👤 Patient Information")

def user_input_features():
    # Create two columns for better layout
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.subheader("Numerical Features")
        age = st.slider('Age', 20, 100, 50)
        trestbps = st.slider('Resting BP (mm Hg)', 90, 200, 120)
        chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
        thalach = st.slider('Max Heart Rate', 60, 220, 150)
        oldpeak = st.slider('ST Depression', 0.0, 6.2, 1.0, 0.1)
    
    with col2:
        st.subheader("Categorical Features")
        sex = st.radio('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        fbs = st.radio('Fasting BS > 120', ['No', 'Yes'])
        restecg = st.selectbox('Resting ECG', 
                              ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        exang = st.radio('Exercise Angina', ['No', 'Yes'])
        slope = st.selectbox('ST Slope', 
                            ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.slider('Major Vessels', 0, 3, 0)
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # Convert to numerical values
    sex_num = 1 if sex == 'Male' else 0
    cp_num = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp]
    fbs_num = 1 if fbs == 'Yes' else 0
    restecg_num = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}[restecg]
    exang_num = 1 if exang == 'Yes' else 0
    slope_num = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}[slope]
    thal_num = {'Normal': 3, 'Fixed Defect': 6, 'Reversible Defect': 7}[thal]
    
    data = {
        'age': age,
        'sex': sex_num,
        'cp': cp_num,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_num,
        'restecg': restecg_num,
        'thalach': thalach,
        'exang': exang_num,
        'oldpeak': oldpeak,
        'slope': slope_num,
        'ca': ca,
        'thal': thal_num
    }
    
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display user input
st.header("📋 Patient Input Features")
st.dataframe(input_df.style.format("{:.1f}"), use_container_width=True)

# Prediction button
if st.sidebar.button('🔍 Predict Heart Disease', type='primary', use_container_width=True):
    if lr_model is not None and rf_model is not None:
        try:
            # Select model
            model = lr_model if model_choice == "Logistic Regression" else rf_model
            
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Display results
            st.header("🎯 Prediction Results")
            
            # Create result boxes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                result = "🫀 Heart Disease Detected" if prediction[0] == 1 else "✅ No Heart Disease"
                st.metric("Prediction", result)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                confidence = max(prediction_proba[0])
                st.metric("Confidence", f"{confidence:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.metric("Model Used", model_choice)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability chart
            st.subheader("📊 Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = ['No Heart Disease', 'Heart Disease']
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.bar(categories, prediction_proba[0], color=colors, alpha=0.8)
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.set_title('Prediction Probabilities')
            
            # Add value labels
            for bar, value in zip(bars, prediction_proba[0]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            
            # Risk factors analysis
            st.header("⚠️ Risk Factors Analysis")
            
            risk_factors = []
            thresholds = {
                'age': ('>60', 60, input_df['age'].iloc[0] > 60),
                'thalach': ('<120', 120, input_df['thalach'].iloc[0] < 120),
                'oldpeak': ('>2.0', 2.0, input_df['oldpeak'].iloc[0] > 2.0),
                'chol': ('>240', 240, input_df['chol'].iloc[0] > 240),
                'trestbps': ('>140', 140, input_df['trestbps'].iloc[0] > 140)
            }
            
            for feature, (condition, threshold, is_risk) in thresholds.items():
                if is_risk:
                    value = input_df[feature].iloc[0]
                    risk_factors.append(f"{feature.upper()} ({value}) - {condition} (threshold: {threshold})")
            
            if risk_factors:
                st.warning("**Significant Risk Factors Detected:**")
                for factor in risk_factors:
                    st.markdown(f'<div class="risk-factor">• {factor}</div>', unsafe_allow_html=True)
            else:
                st.success("**✅ No significant risk factors detected based on input values.**")
            
            # Explainable AI Section
            st.header("🤖 Explainable AI (XAI) Insights")
            st.markdown("""
            This section provides explanations for the model's prediction using SHAP and LIME techniques, 
            making the AI decision-making process transparent and interpretable.
            """)
            
            # Create tabs for different XAI methods
            tab1, tab2 = st.tabs(["SHAP Explanation", "LIME Explanation"])
            
            with tab1:
                st.subheader("SHAP (SHapley Additive exPlanations)")
                st.markdown("""
                SHAP values show how much each feature contributes to the prediction, either positively or negatively.
                """)
                
                try:
                    # Get the preprocessor and model from the pipeline
                    preprocessor = model.named_steps['preprocessor']
                    model_for_shap = model.named_steps['classifier']
                    
                    # Transform the input data
                    input_transformed = preprocessor.transform(input_df)
                    
                    # Get feature names after transformation
                    numerical_features = app_config['numerical_cols']
                    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(app_config['categorical_cols'])
                    feature_names = list(numerical_features) + list(categorical_features)
                    
                    # Create a SHAP explainer
                    if model_choice == "Logistic Regression":
                        explainer = shap.LinearExplainer(model_for_shap, input_transformed, feature_names=feature_names)
                    else:
                        explainer = shap.TreeExplainer(model_for_shap)
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(input_transformed)
                    
                    # Plot SHAP values
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    if model_choice == "Logistic Regression":
                        shap.summary_plot(shap_values, input_transformed, feature_names=feature_names, plot_type="bar", show=False)
                    else:
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            # For binary classification, we usually use the values for class 1
                            shap.summary_plot(shap_values[1], input_transformed, feature_names=feature_names, plot_type="bar", show=False)
                        else:
                            shap.summary_plot(shap_values, input_transformed, feature_names=feature_names, plot_type="bar", show=False)
                    
                    plt.title(f"SHAP Feature Importance - {model_choice}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("""
                    **Interpretation:**
                    - Features with positive SHAP values increase the risk of heart disease
                    - Features with negative SHAP values decrease the risk of heart disease
                    - The length of the bar shows the magnitude of the feature's impact
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error in SHAP explanation: {str(e)}")
                    st.info("This might be due to model type or data format issues.")
            
            with tab2:
                st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
                st.markdown("""
                LIME explains individual predictions by approximating the model locally with an interpretable model.
                """)
                
                try:
                    # Get the preprocessor from the pipeline
                    preprocessor = model.named_steps['preprocessor']
                    
                    # Transform the input data
                    input_transformed = preprocessor.transform(input_df)
                    
                    # Get feature names after transformation
                    numerical_features = app_config['numerical_cols']
                    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(app_config['categorical_cols'])
                    feature_names = list(numerical_features) + list(categorical_features)
                    
                    # Create a custom predict function that handles the preprocessing
                    def predict_proba_wrapper(x):
                        # The input to LIME is already in the transformed space
                        # We need to use the model's predict_proba directly
                        if model_choice == "Logistic Regression":
                            return model.named_steps['classifier'].predict_proba(x)
                        else:
                            return model.named_steps['classifier'].predict_proba(x)
                    
                    # Create a LIME explainer with the transformed training data
                    X_train_transformed = preprocessor.transform(app_config['X_train'])
                    
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        training_data=X_train_transformed, 
                        feature_names=feature_names,
                        class_names=['No Heart Disease', 'Heart Disease'],
                        mode='classification',
                        verbose=False,
                        random_state=42
                    )
                    
                    # Explain the instance
                    exp = explainer.explain_instance(
                        input_transformed[0], 
                        predict_proba_wrapper, 
                        num_features=10
                    )
                    
                    # Display the explanation
                    fig = exp.as_pyplot_figure()
                    plt.title(f"LIME Explanation - {model_choice}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("""
                    **Interpretation:**
                    - The plot shows which features are contributing to this specific prediction
                    - Orange bars indicate features increasing the risk
                    - Blue bars indicate features decreasing the risk
                    - The values show the actual feature values for this patient
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error in LIME explanation: {str(e)}")
                    st.info("This might be due to data format or LIME configuration issues.")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Note: Some explainability features require additional setup. Make sure all dependencies are installed.")
    else:
        st.error("❌ Models not loaded properly. Please run train_model.py first.")

# Information section
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Dataset:** Cleveland Heart Disease  
**Models:** Logistic Regression & Random Forest  
**XAI Methods:** SHAP & LIME  

**Note:** This is for educational purposes only.
""")