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


