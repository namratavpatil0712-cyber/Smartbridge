# Hypertension Prediction Project

## Project Scenarios

**Scenario 1: Hypertensive Patient Monitoring**

A patient with diagnosed hypertension visits a healthcare facility where the Hypertension Prediction system is deployed. By entering their clinical parameters including blood pressure readings, symptoms, medication status, and lifestyle factors, the system provides an immediate risk assessment and classification. Healthcare providers can use this prediction to make informed decisions about treatment adjustments and patient monitoring frequency.

**Scenario 2: Preventive Health Screening**

A healthcare facility conducts routine health screenings for at‑risk populations. Using the Hypertension Prediction system, medical staff can quickly assess individuals who may be developing hypertension before it becomes critical. The system identifies patients in early stages (Stage‑1) who could benefit from lifestyle interventions and preventive measures.

**Scenario 3: Emergency Department Triage**

In emergency department settings, the Hypertension Prediction system assists medical staff in rapidly triaging patients with cardiovascular complaints. By analyzing vital signs and patient history, the system can identify hypertensive crises that require immediate intervention, helping prioritize critical cases.

---

## Project Flow

1. User enters patient demographic and blood test parameters through the interface.
2. System preprocesses the input data using the trained scaler.
3. Integrated machine learning model analyzes the standardized features.
4. Prediction result (Anemia/No Anemia) is displayed with confidence metrics.
5. System provides clinical interpretation and recommendations.

---

## Project Activities

1. **Data Collection & Preparation**
   - Collect the anemia dataset
   - Data preparation and cleaning

2. **Exploratory Data Analysis**
   - Descriptive statistics
   - Visual analysis
   - Univariate and bivariate analysis

3. **Model Building**
   - Training the model with multiple algorithms
   - Testing different classifiers

4. **Performance Testing & Model Selection**
   - Testing model with multiple evaluation metrics
   - Comparing model accuracy across different algorithms
   - Selecting the best performing model

5. **Model Deployment**
   - Save the best model
   - Integrate with web framework

---

## Project Structure

```
HYPERTENSION PREDICTION/
├── app.py              # Flask application backend
├── logreg_model.pkl    # Trained model file
├── static/             # Static assets (CSS, JavaScript, uploaded images)
│   └── style.css       # Application styling
└── templates/          # HTML templates for Flask application
    └── index.html      # Landing page
```

Use this structure as the basis for organizing the project files.

---

*File created and maintained by Neeraj Jadhav.*


## Project Overview
This project predicts hypertension stages using machine learning models.

The dataset includes patient health information such as:
- Gender
- Age
- Medical History
- Medication Status
- Systolic Pressure
- Diastolic Pressure
- Symptoms

The goal is to analyze patient data and build machine learning models to predict hypertension stages.

---

## Dataset Features

Gender  
Age  
History  
Patient  
TakeMedication  
Severity  
BreathShortness  
VisualChanges  
NoseBleeding  
Whendiagnoused  
Systolic  
Diastolic  
ControlledDiet  
Stages (Target)

---

## Data Preprocessing

### Label Encoding

Gender: Male = 0 , Female = 1  
Binary Features: No = 0 , Yes = 1  

Age Groups:
18-34 = 1  
35-50 = 2  
51-64 = 3  
65+ = 4  

Severity:
Mild = 0  
Moderate = 1  
Severe = 2  

Target Stages:
Normal = 0  
Stage-1 = 1  
Stage-2 = 2  
Crisis = 3  

---

## Feature Scaling

MinMaxScaler was applied to ordinal features to improve model performance.

---

## Exploratory Data Analysis

### Gender Distribution
The dataset contains an almost equal number of male and female patients.

### Hypertension Stage Distribution
Stage-1 hypertension is the most common stage in the dataset.

### Correlation Analysis
A strong positive correlation exists between systolic and diastolic pressure.

### Medication vs Severity
Patients taking medication are mostly in higher severity stages.

### Age vs Hypertension Stage
Older age groups show higher hypertension stages.

---

## Data Splitting

Training Set: 1460 samples (80%)

Testing Set: 365 samples (20%)

Stratified sampling ensures balanced class representation.

---

## Machine Learning Models Used

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Ridge Classifier

Gaussian Naive Bayes

---

## Model Accuracy Comparison

Logistic Regression : 96.99%

Decision Tree : 99.73%

Random Forest : 100.0%

SVM : 99.73%

KNN : 97.26%

Ridge Classifier : 95.07%

Gaussian Naive Bayes : 88.49%

---

## Confusion Matrix

Confusion Matrix was generated using the Random Forest model to evaluate classification performance.


## Created by Namrata Patil
---

# Comprehensive Model Comparison

## Overfitting Analysis and Model Selection Rationale

### Why Logistic Regression Was Selected

Although some machine learning models achieved perfect accuracy, careful analysis was required to ensure the model would generalize well to real-world medical data.

### Critical Analysis of High-Performing Models

#### Perfect Accuracy Models (100%) - Overfitting Indicators

The following models achieved **100% test accuracy**:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

While perfect accuracy may appear ideal, it is often a **sign of overfitting**, especially in healthcare datasets.

These models may have memorized patterns from the training data rather than learning generalizable relationships.

### Overfitting Consequences

- Poor performance on unseen patient data
- Lack of adaptability to real-world clinical variations
- Risk of incorrect clinical decisions
- Reduced reliability in medical environments

### Key Performance Indicators

| Metric | Value |
|------|------|
| Overall Accuracy | 95.2% |
| Macro Average F1 Score | 0.95 |
| Weighted Average F1 Score | 0.95 |
| Crisis Recall | 100% |
| Stage-2 Precision | 100% |

Based on these factors, **Logistic Regression** was selected as the final model due to its balance of **accuracy, stability, and generalization capability**.

---

# Model Persistence

To deploy the trained model effectively, **model serialization** was implemented.

### Benefits of Model Serialization

- Persistent storage of trained parameters
- Easy model reuse without retraining
- Version control for model updates
- Deployment-ready format
- Cross-platform compatibility

The trained machine learning model was saved using **Python serialization techniques** such as `pickle` or `joblib`.

This allows the model to be loaded directly into the web application for real-time predictions.

---

# Flask Web Application Development

To make the machine learning model accessible to users, a **Flask-based web application** was developed.

## Application Architecture

The application consists of two main components:

### Backend Implementation

- Developed using **Flask (Python Framework)**
- Handles model loading and prediction logic
- Processes patient input data
- Generates hypertension stage predictions

File: `app.py`

### Frontend Implementation

The user interface was designed using **HTML, CSS, and Bootstrap** to create a professional medical interface.

File: `index.html`

---

## Professional Medical Interface

The web application includes the following features:

- Medical-grade professional interface
- Responsive design for desktop and mobile devices
- Real-time form validation
- Color-coded risk assessment results
- Clinical recommendations for patient care
- Accessibility-focused healthcare interface

---

# Application Workflow

## 1. Home Page Interface

Features:

- Clean and professional healthcare design
- Patient assessment form
- Secure and validated input fields

---

## 2. Patient Data Input Form

The form collects the following information:

### Demographics
- Gender
- Age Group

### Medical History
- Family history
- Current medication status

### Symptoms Assessment
- Breath shortness
- Visual changes
- Nose bleeding

### Blood Pressure Readings
- Systolic values
- Diastolic values

### Lifestyle Factors
- Diet control
- Medication adherence

---

## 3. Risk Assessment Results

After submitting the form, the system provides:

- Hypertension stage classification
- Prediction confidence percentage
- Clinical recommendations
- Actionable medical guidance
- Color-coded risk levels

---

# Model Deployment

The trained machine learning model is integrated into the Flask web application, enabling **real-time hypertension risk prediction** based on patient input data.

This deployment allows healthcare professionals and users to easily interact with the prediction system through a simple web interface.

---
