# Hospital Crowding Prediction System

An end-to-end Machine Learning web application that predicts hospital crowding levels (Low, Medium, High) using a Decision Tree Classifier and an interactive healthcare dashboard.

The trained model achieves **99.17% test accuracy** and is integrated with a Flask REST API for real predictions.

## Project Overview

The system predicts hospital crowding based on operational factors such as:

- Patient Arrivals
- Emergency Cases
- Queue Length
- Discharge Count
- Hospital Capacity
- Occupied Beds
- Available Doctors
- Available Nurses
- Department
- Patient Type
- Hour of Day

The project goes beyond model training by integrating the ML model into a functional web dashboard.

## Key Features

- Real Machine Learning predictions
- Decision Tree Classifier with 99.17% test accuracy
- Flask REST API
- Interactive responsive dashboard
- Dynamic KPI metrics
- Bed occupancy visualization
- Staffing analysis
- Feature importance visualization
- Recent prediction history
- Input validation and API error handling

## Machine Learning

**Algorithm:** Decision Tree Classifier

**Test Accuracy:** 99.17%

**Target Classes:**

- Low
- Medium
- High

### Top Feature Importances

| Feature | Importance |
|---|---:|
| Queue Length | 67.82% |
| Bed Occupancy Rate | 24.86% |
| Available Doctors | 6.53% |
| Patient Arrivals | 0.23% |

## System Architecture

```text
Frontend Dashboard
HTML + CSS + JavaScript
        ↓
Flask REST API
        ↓
Decision Tree Model
        ↓
Real Prediction
        ↓
Dashboard Result