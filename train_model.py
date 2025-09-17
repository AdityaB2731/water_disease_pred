# # train_model.py
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import joblib

# # -------------------- Generate Synthetic Dataset --------------------
# np.random.seed(42)
# n = 1000
# data = pd.DataFrame({
#     'turbidity': np.random.uniform(1, 10, n),
#     'pH': np.random.uniform(6, 8, n),
#     'bacteria_presence': np.random.choice([0, 1], n, p=[0.7, 0.3]),
#     'rainfall': np.random.uniform(0, 200, n),
#     'cases_last_week': np.random.randint(0, 30, n),
#     'patient_fever': np.random.choice([0, 1], n),
#     'patient_diarrhea': np.random.choice([0, 1], n),
#     'patient_abdominal_pain': np.random.choice([0, 1], n),
#     'season': np.random.choice(['summer','monsoon','winter'], n)
# })

# # One-hot encode season
# data = pd.get_dummies(data, columns=['season'])

# # Assign disease labels
# def assign_disease(row):
#     if row['bacteria_presence'] == 1 and row['turbidity'] > 7 and row['patient_fever'] and row['patient_diarrhea']:
#         return 'cholera'
#     elif row['bacteria_presence'] == 1 and row['patient_fever'] and row['patient_abdominal_pain']:
#         return 'typhoid'
#     elif row['bacteria_presence'] == 1 and row['patient_diarrhea'] and row['rainfall'] > 100:
#         return 'giardiasis'
#     else:
#         return 'none'

# data['disease'] = data.apply(assign_disease, axis=1)

# # Prepare features & labels
# X = data.drop('disease', axis=1)
# y = data['disease']

# # Save feature order for later
# joblib.dump(X.columns.tolist(), 'training_columns.pkl')

# # Train RandomForest
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the trained model
# joblib.dump(model, 'disease_predictor.pkl')

# print("Model trained and saved as 'disease_predictor.pkl'.")
# print("Feature columns saved as 'training_columns.pkl'.")
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------- Load Dataset --------------------
data = pd.read_csv('water_disease_data.csv')

# One-hot encode 'season' column
data = pd.get_dummies(data, columns=['season'])

# Prepare features and labels
X = data.drop('disease', axis=1)
y = data['disease']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'label_encoder.pkl')
# Save feature order for later
joblib.dump(X.columns.tolist(), 'training_columns.pkl')

# Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Train RandomForest
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='weighted'),
    "recall": recall_score(y_test, y_pred, average='weighted'),
    "f1": f1_score(y_test, y_pred, average='weighted'),
    "report": classification_report(y_test, y_pred, target_names=label_encoder.classes_)
}
# Print metrics
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print("\nClassification Report:\n", report)
# Save trained model
joblib.dump(metrics, 'model_metrics.pkl')
joblib.dump(model, 'disease_predictor.pkl')
# joblib.dump(accuracy, 'model_accuracy.pkl')
print("Model trained and saved as 'disease_predictor.pkl'.")
print("Feature columns saved as 'training_columns.pkl'.")
