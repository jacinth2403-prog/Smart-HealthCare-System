import pandas as pd

df = pd.read_csv("data/processed/dataset1_clean.csv")

print("===== FINAL STOCKOUT CHECK =====")

print(
    "Stockout with unfulfilled demand = 0:",
    ((df["stockout_flag"] == 1) &
     (df["unfulfilled_demand"] == 0)).sum()
)

print(
    "Non-stockout with unfulfilled demand > 0:",
    ((df["stockout_flag"] == 0) &
     (df["unfulfilled_demand"] > 0)).sum()
)

print(
    "Stockout with closing stock > 0:",
    ((df["stockout_flag"] == 1) &
     (df["closing_stock"] > 0)).sum()
)

print(
    "Stockout with closing stock = 0:",
    ((df["stockout_flag"] == 1) &
     (df["closing_stock"] == 0)).sum()
)