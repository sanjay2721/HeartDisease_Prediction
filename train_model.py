import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import lime
import lime.lime_tabular
import shap

print("=== Heart Disease Prediction Model Training ===")

# --- SETUP DATA PATH ---
DATA_PATH = "data/processed.cleveland.data"

# --- LOAD DATA ---
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, header=None, na_values=['?'])
print(f"Dataset loaded successfully. Shape: {df.shape}")

# --- PREPROCESS DATA ---
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

if df.shape[1] == len(column_names):
    df.columns = column_names
    print("Columns renamed successfully.")
else:
    raise ValueError(f"Expected {len(column_names)} columns, got {df.shape[1]}")

# Convert data types
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

df['ca'] = pd.to_numeric(df['ca'], errors='coerce').astype('Int64')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce').astype('Int64')

# Binarize target
df['target'] = pd.to_numeric(df['target'], errors='coerce').astype('Int64')
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values
for col in ['ca', 'thal']:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed missing values in '{col}' with mode: {mode_val}")

print("Data preprocessing completed successfully!")

# --- FEATURE ENGINEERING AND MODEL TRAINING ---
print("\n2. Feature engineering and model training...")

X = df.drop('target', axis=1)
y = df['target']

numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# --- TRAIN LOGISTIC REGRESSION ---
print("\n3. Training Logistic Regression model...")
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")

# --- TRAIN RANDOM FOREST ---
print("\n4. Training Random Forest model...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# --- PREPARE FOR EXPLAINABILITY ---
print("\n5. Preparing for model explainability...")

# Fit the preprocessor on the training data
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)

# Get feature names after transformation
numeric_features = numerical_cols
categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = list(numeric_features) + list(categorical_features)

print(f"Original features: {len(X.columns)}")
print(f"Transformed features: {len(all_feature_names)}")

# --- SAVE MODELS AND DATA ---
print("\n6. Saving models and data...")

joblib.dump(lr_pipeline, 'logistic_regression_model.pkl')
joblib.dump(rf_pipeline, 'random_forest_model.pkl')

app_data = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'all_columns': X.columns.tolist(),
    'feature_names': all_feature_names,
    'X_train': X_train,  # Save training data for LIME
    'X_test': X_test,    # Save test data for evaluation
    'y_train': y_train,
    'y_test': y_test
}
joblib.dump(app_data, 'app_config.pkl')

print("""
✅ All models and data saved successfully!
Files created:
- logistic_regression_model.pkl
- random_forest_model.pkl
- app_config.pkl
""")

# --- FINAL EVALUATION ---
print("\n7. Final model evaluation:")
print(f"\nLogistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))

print(f"\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))

print("\n=== Model Training Complete ===")
print("You can now run the Streamlit app with: streamlit run app.py")