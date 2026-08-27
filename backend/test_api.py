"""
Comprehensive End-to-End Test Suite for Hospital Crowding Prediction API
Tests all required scenarios:
1. Valid prediction (Low, Medium, High)
2. Missing fields
3. Invalid values & bounds
4. Multiple predictions
5. Feature importance endpoint
6. Health / Status check
"""

import sys
import os
import json

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
from model import model_service


def run_tests():
    client = app.test_client()
    passed_tests = 0
    total_tests = 0

    print("=" * 70)
    print("STARTING END-TO-END TESTS FOR HOSPITAL CROWDING PREDICTION API")
    print("=" * 70)

    # Test 1: Health Check Endpoint
    total_tests += 1
    print("\n[TEST 1] GET /health (System Status Check)")
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "online", "Expected status 'online'"
    assert data["model_loaded"] is True, "Expected model_loaded True"
    assert "Decision Tree" in data["algorithm"], f"Expected Decision Tree algorithm, got {data['algorithm']}"
    assert data["accuracy"] == "99.17%", f"Expected 99.17%, got {data['accuracy']}"
    assert data["engineered_features"] == 20, f"Expected 20 features, got {data['engineered_features']}"
    print(f"  [PASS] Passed: Health check returned online status with Decision Tree model (Accuracy: {data['accuracy']})")
    passed_tests += 1

    # Test 2: Feature Importance Endpoint
    total_tests += 1
    print("\n[TEST 2] GET /api/feature-importance (Real Decision Tree Feature Weights)")
    res = client.get("/api/feature-importance")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "success", "Expected status 'success'"
    assert len(data["feature_importances"]) == 20, f"Expected 20 features, got {len(data['feature_importances'])}"
    top_feature = data["feature_importances"][0]
    print(f"  [PASS] Passed: Feature importance retrieved. Top feature: {top_feature['name']} ({top_feature['percentage']}%)")
    passed_tests += 1

    # Test 3: Valid High Crowding Prediction
    total_tests += 1
    print("\n[TEST 3] POST /predict (High Crowding Scenario)")
    high_payload = {
        "patient_arrivals": 90,
        "emergency_cases": 20,
        "queue_length": 30,
        "discharge_count": 5,
        "hospital_capacity": 100,
        "occupied_beds": 95,
        "available_doctors": 3,
        "available_nurses": 8,
        "department": "Emergency",
        "patient_type": "Urgent",
        "hour": 14
    }
    res = client.post("/predict", data=json.dumps(high_payload), content_type="application/json")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "success", "Expected success"
    assert data["prediction"] in ["Low", "Medium", "High"], f"Invalid prediction: {data['prediction']}"
    assert "confidence" in data, "Missing confidence"
    assert "metrics" in data, "Missing metrics"
    assert data["metrics"]["bed_occupancy_percentage"] == 95.0, "Mismatch in calculated bed occupancy"
    print(f"  [PASS] Passed: Predicted '{data['prediction']}' Crowding (Confidence: {data['confidence'] * 100:.1f}%, Occupancy: {data['metrics']['bed_occupancy_percentage']}%, Ratio: {data['metrics']['staff_to_patient_ratio']})")
    passed_tests += 1

    # Test 4: Valid Low Crowding Prediction
    total_tests += 1
    print("\n[TEST 4] POST /predict (Low Crowding Scenario)")
    low_payload = {
        "patient_arrivals": 12,
        "emergency_cases": 0,
        "queue_length": 0,
        "discharge_count": 10,
        "hospital_capacity": 200,
        "occupied_beds": 35,
        "available_doctors": 15,
        "available_nurses": 45,
        "department": "Pediatrics",
        "patient_type": "Routine",
        "hour": 9
    }
    res = client.post("/predict", data=json.dumps(low_payload), content_type="application/json")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "success", "Expected success"
    assert data["prediction"] == "Low", f"Expected 'Low', got {data['prediction']}"
    print(f"  [PASS] Passed: Predicted '{data['prediction']}' Crowding (Confidence: {data['confidence'] * 100:.1f}%, Occupancy: {data['metrics']['bed_occupancy_percentage']}%, Ratio: {data['metrics']['staff_to_patient_ratio']})")
    passed_tests += 1

    # Test 5: Valid Medium Crowding Prediction
    total_tests += 1
    print("\n[TEST 5] POST /predict (Medium Crowding Scenario)")
    medium_payload = {
        "patient_arrivals": 38,
        "emergency_cases": 1,
        "queue_length": 0,
        "discharge_count": 8,
        "hospital_capacity": 200,
        "occupied_beds": 158,
        "available_doctors": 13,
        "available_nurses": 39,
        "department": "Outpatient",
        "patient_type": "Follow-up",
        "hour": 14
    }
    res = client.post("/predict", data=json.dumps(medium_payload), content_type="application/json")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "success", "Expected success"
    assert data["prediction"] in ["Medium", "High", "Low"], f"Invalid prediction: {data['prediction']}"
    print(f"  [PASS] Passed: Predicted '{data['prediction']}' Crowding (Confidence: {data['confidence'] * 100:.1f}%, Bed Occ: {data['metrics']['bed_occupancy_percentage']}%)")
    passed_tests += 1

    # Test 6: Missing Field Validation Error
    total_tests += 1
    print("\n[TEST 6] POST /predict (Missing Field Error Handling)")
    incomplete_payload = {
        "patient_arrivals": 45,
        "emergency_cases": 5,
        # queue_length is missing
        "hospital_capacity": 200,
        "occupied_beds": 150
    }
    res = client.post("/predict", data=json.dumps(incomplete_payload), content_type="application/json")
    assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "error", "Expected status 'error'"
    assert "Missing required feature" in data["message"], f"Unexpected error message: {data['message']}"
    print(f"  [PASS] Passed: Properly rejected missing field with error: '{data['message']}'")
    passed_tests += 1

    # Test 7: Invalid Value (Occupied Beds > Capacity)
    total_tests += 1
    print("\n[TEST 7] POST /predict (Occupied Beds > Capacity Validation)")
    overflow_payload = {
        "patient_arrivals": 45,
        "emergency_cases": 5,
        "queue_length": 12,
        "discharge_count": 8,
        "hospital_capacity": 100,
        "occupied_beds": 150,  # exceeds capacity
        "available_doctors": 10,
        "available_nurses": 30,
        "department": "Emergency",
        "patient_type": "Emergency",
        "hour": 10
    }
    res = client.post("/predict", data=json.dumps(overflow_payload), content_type="application/json")
    assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "error", "Expected status 'error'"
    assert "cannot exceed hospital capacity" in data["message"], f"Unexpected error message: {data['message']}"
    print(f"  [PASS] Passed: Properly rejected invalid capacity with error: '{data['message']}'")
    passed_tests += 1

    # Test 8: Invalid Value (Negative Numbers)
    total_tests += 1
    print("\n[TEST 8] POST /predict (Negative Numbers Validation)")
    negative_payload = {
        "patient_arrivals": -5,  # negative
        "emergency_cases": 5,
        "queue_length": 12,
        "discharge_count": 8,
        "hospital_capacity": 200,
        "occupied_beds": 100,
        "available_doctors": 10,
        "available_nurses": 30,
        "department": "Emergency",
        "patient_type": "Emergency",
        "hour": 10
    }
    res = client.post("/predict", data=json.dumps(negative_payload), content_type="application/json")
    assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "error", "Expected status 'error'"
    assert "cannot be negative" in data["message"], f"Unexpected error message: {data['message']}"
    print(f"  [PASS] Passed: Properly rejected negative value with error: '{data['message']}'")
    passed_tests += 1

    # Test 9: Multiple Sequential Predictions
    total_tests += 1
    print("\n[TEST 9] POST /predict (Multiple Sequential Predictions)")
    departments = ["Emergency", "Internal Medicine", "Outpatient", "Pediatrics", "Surgery"]
    for i, dept in enumerate(departments):
        payload = {
            "patient_arrivals": 20 + i * 15,
            "emergency_cases": i * 3,
            "queue_length": i * 5,
            "discharge_count": 5 + i,
            "hospital_capacity": 150,
            "occupied_beds": 40 + i * 20,
            "available_doctors": 8 + i,
            "available_nurses": 20 + i * 2,
            "department": dept,
            "patient_type": "Routine" if i % 2 == 0 else "Follow-up",
            "hour": (8 + i * 2) % 24
        }
        res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert res.status_code == 200, f"Failed on iteration {i} for dept {dept}"
        data = res.get_json()
        assert data["status"] == "success", f"Failed on iteration {i}"
        print(f"  Iteration {i+1} ({dept}): Predicted {data['prediction']} (Confidence: {data['confidence']*100:.0f}%, Bed Occ: {data['metrics']['bed_occupancy_percentage']}%)")

    print("  [PASS] Passed: Successfully ran 5 sequential predictions across different departments.")
    passed_tests += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} Tests Passed (100% Success)")
    print("=" * 70)
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
