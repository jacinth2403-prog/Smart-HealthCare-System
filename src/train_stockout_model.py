import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("data/processed/master_dataset.csv")

features = [
    "facility_id",
    "medicine_id",
    "medicine_unit",
    "month",
    "patient_activity",
    "opening_stock",
    "reorder_level",
    "lead_time_days",
    "population",
    "male_percentage",
    "female_percentage",
    "reported_outbreak_count",
    "reported_disease_cases",
    "reported_outbreak_flag"
]

target = "stockout_flag"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

categorical_features = [
    "facility_id",
    "medicine_id",
    "medicine_unit",
    "month"
]

numeric_features = [
    "patient_activity",
    "opening_stock",
    "reorder_level",
    "lead_time_days",
    "population",
    "male_percentage",
    "female_percentage",
    "reported_outbreak_count",
    "reported_disease_cases",
    "reported_outbreak_flag"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n===== STOCKOUT PREDICTION MODEL =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "features": features
    },
    "src/stockout_model.pkl"
)

print("\nModel saved to: src/stockout_model.pkl")