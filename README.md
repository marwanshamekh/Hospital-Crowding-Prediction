# Hospital Crowding Prediction System

An end-to-end **Machine Learning web application** that predicts hospital crowding levels (**Low, Medium, High**) using a Decision Tree Classifier and an interactive healthcare dashboard.

The trained model achieves **99.17% test accuracy** and is integrated with a **Flask REST API** for real predictions.

---

## 📌 Project Overview

The system predicts hospital crowding based on operational factors such as:

* Patient Arrivals
* Emergency Cases
* Queue Length
* Discharge Count
* Hospital Capacity
* Occupied Beds
* Available Doctors
* Available Nurses
* Department
* Patient Type
* Hour of Day

The project integrates the ML model into a functional web dashboard.

---

## 🚀 Key Features

* Real Machine Learning predictions
* Decision Tree Classifier — **99.17% accuracy**
* Flask REST API
* Interactive responsive dashboard
* Dynamic KPI metrics
* Bed occupancy & staffing visualization
* Feature importance visualization
* Prediction history
* Input validation & API error handling

---

## 🧠 Machine Learning

**Algorithm:** Decision Tree Classifier

**Test Accuracy:** 99.17%

**Target Classes:**

* Low
* Medium
* High

### Top Feature Importances

| Feature                | Importance |
| ---------------------- | ---------: |
| Queue Length           |     67.82% |
| Bed Occupancy Rate     |     24.86% |
| Available Doctors      |      6.53% |
| Patient Arrivals       |      0.23% |
| Staff-to-Patient Ratio |      0.12% |

---

## 🏗️ System Architecture

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
```

---

## 🛠️ Technologies

**Machine Learning:** Python, Pandas, NumPy, Scikit-learn

**Backend:** Python, Flask, REST API

**Frontend:** HTML, CSS, JavaScript

**Tools:** Git, GitHub, VS Code

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/marwanshamekh/Hospital-Crowding-Prediction.git
cd Hospital-Crowding-Prediction
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run the backend

```bash
python backend/app.py
```

The API will run at:

```text
http://127.0.0.1:5000
```

### 4. Open the frontend

Run the frontend using **VS Code Live Server**.

Make sure the Flask backend is running before using predictions.

