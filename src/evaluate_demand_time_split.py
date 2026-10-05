import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

input_file = "data/processed/dataset1_demand_ml.csv"

df = pd.read_csv(input_file)

df["medicine_id"] = df["medicine_id"].astype("category").cat.codes
df["medicine_unit"] = df["medicine_unit"].astype("category").cat.codes
df["month"] = df["month"].astype("category").cat.codes

features = [
    "medicine_id",
    "medicine_unit",
    "month",
    "patient_activity",
    "opening_stock",
    "reorder_level",
    "lead_time_days"
]

train = df[df["month"] == 0]
test = df[df["month"] == 1]

X_train = train[features]
y_train = train["demand_quantity"]

X_test = test[features]
y_test = test["demand_quantity"]

model = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("===== TIME-BASED DEMAND VALIDATION =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")