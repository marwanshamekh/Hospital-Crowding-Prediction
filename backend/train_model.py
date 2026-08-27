"""
Training and Export Pipeline for Hospital Crowding Prediction
Reproduces EXACTLY the preprocessing, feature engineering, and Decision Tree training from Hospital.ipynb.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


def train_and_export_model():
    # 1. Load Dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "Hospital.csv")
    if not os.path.exists(dataset_path):
        dataset_path = "Hospital.csv"
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")

    # 2. Data Preprocessing (Cell 9 & 10 in Hospital.ipynb)
    # Remove Patient_ID
    df_processed = df.drop(columns=['Patient_ID'])
    
    # Convert Date to datetime and extract Month and DayOfWeek
    df_processed['Date'] = pd.to_datetime(df_processed['Date'])
    df_processed['Month'] = df_processed['Date'].dt.month
    df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
    df_processed.drop(columns=['Date'], inplace=True)

    # 3. Feature Engineering (Cell 11 in Hospital.ipynb)
    df_processed['Bed_Occupancy_Rate'] = df_processed['Occupied_Beds'] / df_processed['Hospital_Capacity']
    df_processed['Total_Staff'] = df_processed['Available_Doctors'] + df_processed['Available_Nurses']
    df_processed['Staff_to_Patient_Ratio'] = df_processed['Total_Staff'] / (df_processed['Patient_Arrivals'] + 1e-6)

    # 4. One-Hot Encoding (Cell 12 in Hospital.ipynb)
    df_processed = pd.get_dummies(df_processed, columns=['Department', 'Patient_Type'], drop_first=True)

    # 5. Target Encoding (Cell 13 in Hospital.ipynb)
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    reverse_target_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    df_processed['Crowding_Level'] = df_processed['Crowding_Level'].map(target_mapping)

    # 6. Separate features and target (Cell 15 in Hospital.ipynb)
    X = df_processed.drop(columns=['Crowding_Level'])
    y = df_processed['Crowding_Level']

    feature_names = X.columns.tolist()
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature list (exact order): {feature_names}")

    # 7. Train / Test Split (80/20 stratified, random_state=42) (Cell 15 in Hospital.ipynb)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Train Decision Tree Classifier (Cell 18 & 20 in Hospital.ipynb)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 9. Model Evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    cv_f1_mean = float(cv_scores.mean())
    cv_f1_std = float(cv_scores.std())

    print("\n" + "="*50)
    print(f"Decision Tree Model Evaluation:")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}% ({test_accuracy:.4f})")
    print(f"Weighted F1-Score: {test_f1:.4f}")
    print(f"5-Fold CV F1-Score: {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")
    print("="*50)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    # 10. Feature Importances
    importances = model.feature_importances_
    feature_importance_dict = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
    sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop Feature Importances:")
    for feat, imp in sorted_importance:
        print(f"  {feat:30s}: {imp * 100:6.2f}%")

    # 11. Export Model and Metadata
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "hospital_crowding_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to: {model_path}")

    metadata = {
        "model_type": "DecisionTreeClassifier",
        "algorithm": "Decision Tree (CART)",
        "random_state": 42,
        "test_accuracy": round(float(test_accuracy), 4),
        "test_f1_weighted": round(float(test_f1), 4),
        "cv_5fold_f1_mean": round(cv_f1_mean, 4),
        "cv_5fold_f1_std": round(cv_f1_std, 4),
        "target_mapping": target_mapping,
        "reverse_target_mapping": {str(k): v for k, v in reverse_target_mapping.items()},
        "classes": ["Low", "Medium", "High"],
        "feature_names": feature_names,
        "feature_importances": feature_importance_dict,
        "sorted_feature_importances": [{"feature": k, "importance": v} for k, v in sorted_importance],
        "training_records": int(X_train.shape[0]),
        "testing_records": int(X_test.shape[0]),
        "total_records": int(df.shape[0]),
        "engineered_features_count": len(feature_names)
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata successfully saved to: {metadata_path}")

    return model, metadata


if __name__ == "__main__":
    train_and_export_model()
