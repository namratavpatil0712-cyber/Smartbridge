# ============================================
#  Model Persistence – Save Model with Joblib
#  Project: Predictive Pulse – Hypertension
# ============================================

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('hypertension.csv')
X  = df.drop('Stages', axis=1)
y  = df['Stages']

# Train best model (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

# Save model using joblib
joblib.dump(model,  "logreg_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model saved as logreg_model.pkl")
print("✅ Scaler saved as scaler.pkl")