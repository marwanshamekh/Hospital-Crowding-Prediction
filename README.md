# Hospital Crowding Prediction

A Machine Learning project for predicting hospital crowding levels based on hospital capacity, patient arrivals, staff availability, queue length, and other operational features.

## Project Overview

Hospital overcrowding can negatively affect patient experience, waiting times, and the efficiency of healthcare services.

This project aims to predict the hospital's **Crowding Level** using historical hospital operational data and machine learning classification models.

The target variable contains three classes:

* **Low**
* **Medium**
* **High**

## Dataset

The project uses a dataset named `Hospital.csv`.

The dataset contains information related to hospital operations, including:

* Hospital capacity
* Occupied beds
* Patient arrivals
* Available doctors
* Available nurses
* Queue length
* Department
* Patient type
* Date
* Crowding level

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `Patient_ID` column.
2. Converted the `Date` column to datetime.
3. Extracted:

   * Month
   * Day of Week
4. Removed the original `Date` column.
5. Created additional operational features.
6. Applied One-Hot Encoding to categorical variables.
7. Converted the target variable into numerical classes:

   * Low → 0
   * Medium → 1
   * High → 2

## Feature Engineering

Several features were created to better represent hospital workload and capacity.

### Bed Occupancy Rate

```text
Occupied Beds / Hospital Capacity
```

This represents the percentage of hospital beds currently occupied.

### Total Staff

```text
Available Doctors + Available Nurses
```

This combines the available medical workforce into one feature.

### Staff-to-Patient Ratio

```text
Total Staff / Patient Arrivals
```

This provides an indication of available staff relative to the number of arriving patients.

## Machine Learning Models

Four classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

The data was divided into training and testing sets using an 80/20 split with stratification.

Standardization was applied to the Logistic Regression model.

## Model Evaluation

The models were evaluated using:

* Accuracy
* Weighted F1-Score
* Classification Report
* Confusion Matrix
* 5-Fold Cross Validation

Feature importance was also analyzed for the selected Decision Tree model.

## Best Model

The Decision Tree Classifier was selected as the best-performing model based on the evaluation performed in the notebook.

> **Note:** Model performance numbers should be updated after running the notebook with the current dataset.

## Business Insights

The model can potentially help hospitals:

* Monitor crowding levels.
* Identify periods of high congestion.
* Improve patient routing.
* Support hospital capacity planning.
* Better allocate available medical staff.
* Monitor operational indicators such as queue length and bed occupancy.

## Project Structure

```text
Hospital-Crowding-Prediction/
│
├── Hospital.csv
├── Hospital1.ipynb
├── hospital_crowding_model.pkl
├── scaler.pkl
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 3. Open the notebook

```bash
jupyter notebook Hospital1.ipynb
```

### 4. Run the notebook cells

Make sure `Hospital.csv` is located in the same directory as the notebook.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

## Future Improvements

Possible improvements include:

* Hyperparameter tuning.
* Testing additional classification algorithms.
* Handling class imbalance if present.
* Creating a real-time prediction interface.
* Deploying the model using Streamlit or Flask.
* Adding automated data validation and preprocessing pipelines.
