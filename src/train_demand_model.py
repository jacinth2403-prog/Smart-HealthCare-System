import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

target = "demand_quantity"

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
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== DEMAND PREDICTION MODEL =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "features": features
    },
    "src/demand_model.pkl"
)

print("\nModel saved to: src/demand_model.pkl")