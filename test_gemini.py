from src.gemini_service import explain_recommendation

data = {
    "medicine_name": "CETIRIZINE",
    "to_facility": "M.M.Kovilur",
    "from_facility": "Anumantharayankottai",
    "current_stock": 0,
    "reorder_level": 564,
    "shortage": 564,
    "predicted_demand": 920,
    "transfer_quantity": 12,
    "distance_km": 12.83,
    "stockout_flag": 1
}

result = explain_recommendation(data)

print(result)