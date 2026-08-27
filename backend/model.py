"""
Hospital Crowding Model Service
Handles model loading, input validation, feature engineering, and inference
using the exact Decision Tree classifier trained on Hospital.csv.
"""

import os
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "hospital_crowding_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


class HospitalCrowdingModelService:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_names = []
        self.classes = ["Low", "Medium", "High"]
        self.reverse_target_mapping = {0: "Low", 1: "Medium", 2: "High"}
        self._load_model()

    def _load_model(self):
        """Loads the trained Decision Tree model and metadata."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
            print("Model artifacts not found. Training model now...")
            from train_model import train_and_export_model
            train_and_export_model()

        self.model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata.get("feature_names", [
            'Hour', 'Emergency_Cases', 'Patient_Arrivals', 'Queue_Length',
            'Available_Doctors', 'Available_Nurses', 'Hospital_Capacity',
            'Occupied_Beds', 'Discharge_Count', 'Month', 'DayOfWeek',
            'Bed_Occupancy_Rate', 'Total_Staff', 'Staff_to_Patient_Ratio',
            'Department_Internal Medicine', 'Department_Outpatient',
            'Department_Pediatrics', 'Department_Surgery',
            'Patient_Type_Routine', 'Patient_Type_Urgent'
        ])
        print(f"Hospital Crowding Model loaded successfully! (Accuracy: {self.metadata.get('test_accuracy', 0.9917)*100:.2f}%)")

    def get_status(self):
        """Returns model status and metadata."""
        return {
            "model_type": self.metadata.get("model_type", "DecisionTreeClassifier"),
            "algorithm": self.metadata.get("algorithm", "Decision Tree"),
            "accuracy": self.metadata.get("test_accuracy", 0.9917),
            "accuracy_percentage": f"{self.metadata.get('test_accuracy', 0.9917) * 100:.2f}%",
            "f1_score": self.metadata.get("test_f1_weighted", 0.9917),
            "cv_f1_score": self.metadata.get("cv_5fold_f1_mean", 0.9930),
            "total_records": self.metadata.get("total_records", 50000),
            "engineered_features_count": len(self.feature_names),
            "classes": self.classes,
            "is_loaded": self.model is not None
        }

    def get_feature_importance(self):
        """Returns actual feature importances from the loaded Decision Tree."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_imp_list = [
                {"name": feat, "importance": round(float(imp), 4), "percentage": round(float(imp) * 100, 2)}
                for feat, imp in zip(self.feature_names, importances)
            ]
            feature_imp_list.sort(key=lambda x: x["importance"], reverse=True)
            return feature_imp_list
        return self.metadata.get("sorted_feature_importances", [])

    def validate_inputs(self, data):
        """
        Validates raw request payload.
        Returns a tuple: (is_valid: bool, validated_data: dict, error_message: str)
        """
        if not isinstance(data, dict):
            return False, {}, "Request body must be a JSON object."

        # Field aliases for flexible key naming
        key_mapping = {
            "patient_arrivals": ["patient_arrivals", "patientArrivals", "Patient_Arrivals"],
            "emergency_cases": ["emergency_cases", "emergencyCases", "Emergency_Cases"],
            "queue_length": ["queue_length", "queueLength", "Queue_Length"],
            "discharge_count": ["discharge_count", "dischargeCount", "Discharge_Count"],
            "hospital_capacity": ["hospital_capacity", "hospitalCapacity", "Hospital_Capacity"],
            "occupied_beds": ["occupied_beds", "occupiedBeds", "Occupied_Beds"],
            "available_doctors": ["available_doctors", "availableDoctors", "Available_Doctors"],
            "available_nurses": ["available_nurses", "availableNurses", "Available_Nurses"],
            "department": ["department", "Department"],
            "patient_type": ["patient_type", "patientType", "Patient_Type"],
            "hour": ["hour", "Hour"],
            "date": ["date", "Date"]
        }

        extracted = {}
        for canonical, aliases in key_mapping.items():
            for alias in aliases:
                if alias in data:
                    extracted[canonical] = data[alias]
                    break

        # Check required fields
        required_numeric_fields = [
            "patient_arrivals", "emergency_cases", "queue_length", "discharge_count",
            "hospital_capacity", "occupied_beds", "available_doctors", "available_nurses"
        ]

        for field in required_numeric_fields:
            if field not in extracted or extracted[field] is None or extracted[field] == "":
                return False, {}, f"Missing required feature: '{field}'."
            try:
                val = float(extracted[field])
                if val < 0:
                    return False, {}, f"Field '{field}' cannot be negative (received: {val})."
                extracted[field] = int(round(val))
            except (ValueError, TypeError):
                return False, {}, f"Field '{field}' must be a valid non-negative number (received: {extracted.get(field)})."

        # Validate capacity > 0
        if extracted["hospital_capacity"] <= 0:
            return False, {}, "Hospital capacity must be greater than zero."

        if extracted["occupied_beds"] > extracted["hospital_capacity"]:
            return False, {}, f"Occupied beds ({extracted['occupied_beds']}) cannot exceed hospital capacity ({extracted['hospital_capacity']})."

        # Validate Department
        valid_departments = ["Emergency", "Internal Medicine", "Outpatient", "Pediatrics", "Surgery", "ICU"]
        dept = str(extracted.get("department", "Emergency")).strip()
        if not dept:
            dept = "Emergency"
        elif dept.lower() == "icu":
            dept = "Emergency"  # ICU maps to Emergency workload profile in the 5-department model
        elif dept not in valid_departments:
            # Case-insensitive match attempt
            matched = False
            for d in valid_departments:
                if d.lower() == dept.lower():
                    dept = d
                    matched = True
                    break
            if not matched:
                return False, {}, f"Invalid department: '{dept}'. Supported departments: {valid_departments}."
        extracted["department"] = dept

        # Validate Patient Type
        valid_patient_types = ["Follow-up", "Routine", "Urgent", "Emergency"]
        p_type = str(extracted.get("patient_type", "Routine")).strip()
        if not p_type:
            p_type = "Routine"
        elif p_type.lower() == "emergency":
            p_type = "Urgent"  # Emergency patient type maps to Urgent in classification
        elif p_type not in valid_patient_types:
            matched = False
            for pt in valid_patient_types:
                if pt.lower() == p_type.lower():
                    p_type = pt
                    matched = True
                    break
            if not matched:
                return False, {}, f"Invalid patient_type: '{p_type}'. Supported types: {valid_patient_types}."
        extracted["patient_type"] = p_type

        # Validate Hour (0 to 23)
        now = datetime.now()
        if "hour" in extracted and extracted["hour"] is not None and extracted["hour"] != "":
            try:
                h = int(extracted["hour"])
                if not (0 <= h <= 23):
                    return False, {}, f"Hour must be between 0 and 23 (received: {h})."
                extracted["hour"] = h
            except (ValueError, TypeError):
                return False, {}, f"Hour must be a valid integer between 0 and 23 (received: {extracted.get('hour')})."
        else:
            extracted["hour"] = now.hour

        # Date handling (extract Month & DayOfWeek)
        if "date" in extracted and extracted["date"]:
            try:
                parsed_date = pd.to_datetime(extracted["date"])
                extracted["month"] = parsed_date.month
                extracted["day_of_week"] = parsed_date.dayofweek
            except Exception:
                extracted["month"] = now.month
                extracted["day_of_week"] = now.weekday()
        else:
            extracted["month"] = int(data.get("month", data.get("Month", now.month)))
            extracted["day_of_week"] = int(data.get("day_of_week", data.get("DayOfWeek", now.weekday())))

        return True, extracted, ""

    def transform_features(self, validated_data):
        """
        Applies the exact feature engineering from Hospital.ipynb:
        - Bed_Occupancy_Rate = Occupied_Beds / Hospital_Capacity
        - Total_Staff = Available_Doctors + Available_Nurses
        - Staff_to_Patient_Ratio = Total_Staff / (Patient_Arrivals + 1e-6)
        - One-Hot Dummies with drop_first (matching reference categories)
        """
        cap = validated_data["hospital_capacity"]
        occ = validated_data["occupied_beds"]
        arr = validated_data["patient_arrivals"]
        doc = validated_data["available_doctors"]
        nur = validated_data["available_nurses"]

        bed_occupancy_rate = float(occ) / float(cap) if cap > 0 else 0.0
        total_staff = int(doc + nur)
        staff_to_patient_ratio = float(total_staff) / (float(arr) + 1e-6)

        dept = validated_data["department"]
        p_type = validated_data["patient_type"]

        # 20 feature columns in exact order
        feature_dict = {
            "Hour": int(validated_data["hour"]),
            "Emergency_Cases": int(validated_data["emergency_cases"]),
            "Patient_Arrivals": int(arr),
            "Queue_Length": int(validated_data["queue_length"]),
            "Available_Doctors": int(doc),
            "Available_Nurses": int(nur),
            "Hospital_Capacity": int(cap),
            "Occupied_Beds": int(occ),
            "Discharge_Count": int(validated_data["discharge_count"]),
            "Month": int(validated_data["month"]),
            "DayOfWeek": int(validated_data["day_of_week"]),
            "Bed_Occupancy_Rate": bed_occupancy_rate,
            "Total_Staff": total_staff,
            "Staff_to_Patient_Ratio": staff_to_patient_ratio,
            "Department_Internal Medicine": 1 if dept == "Internal Medicine" else 0,
            "Department_Outpatient": 1 if dept == "Outpatient" else 0,
            "Department_Pediatrics": 1 if dept == "Pediatrics" else 0,
            "Department_Surgery": 1 if dept == "Surgery" else 0,
            "Patient_Type_Routine": 1 if p_type == "Routine" else 0,
            "Patient_Type_Urgent": 1 if p_type in ["Urgent", "Emergency"] else 0,
        }

        # Build DataFrame with exact feature order
        df_features = pd.DataFrame([feature_dict])[self.feature_names]
        
        computed_metrics = {
            "bed_occupancy_rate": round(bed_occupancy_rate, 4),
            "bed_occupancy_percentage": round(bed_occupancy_rate * 100, 2),
            "total_staff": total_staff,
            "staff_to_patient_ratio": round(staff_to_patient_ratio, 4),
            "queue_length": validated_data["queue_length"],
            "patient_arrivals": arr,
            "hospital_capacity": cap,
            "occupied_beds": occ,
            "available_doctors": doc,
            "available_nurses": nur,
            "emergency_cases": validated_data["emergency_cases"],
            "discharge_count": validated_data["discharge_count"],
            "department": dept,
            "patient_type": p_type,
            "hour": validated_data["hour"]
        }

        return df_features, feature_dict, computed_metrics

    def predict(self, raw_data):
        """
        Validates input, preprocesses features, and executes real Decision Tree prediction.
        """
        is_valid, validated_data, error_msg = self.validate_inputs(raw_data)
        if not is_valid:
            return {
                "status": "error",
                "message": error_msg
            }

        df_features, feature_dict, computed_metrics = self.transform_features(validated_data)

        # Real model inference
        predicted_class_id = int(self.model.predict(df_features)[0])
        prediction_label = self.reverse_target_mapping.get(predicted_class_id, "Medium")

        # Real model class probabilities from predict_proba
        probabilities = {}
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_features)[0]
            for idx, prob in enumerate(probs):
                class_name = self.reverse_target_mapping.get(idx, str(idx))
                probabilities[class_name] = round(float(prob), 4)
            confidence = round(float(probs[predicted_class_id]), 4)

        return {
            "status": "success",
            "prediction": prediction_label,
            "class_id": predicted_class_id,
            "confidence": confidence,
            "probabilities": probabilities,
            "metrics": computed_metrics,
            "model_info": {
                "model_name": "Decision Tree Classifier",
                "accuracy": f"{self.metadata.get('test_accuracy', 0.9917) * 100:.2f}%",
                "random_state": 42
            },
            "features_used": feature_dict
        }


# Global singleton instance
model_service = HospitalCrowdingModelService()
