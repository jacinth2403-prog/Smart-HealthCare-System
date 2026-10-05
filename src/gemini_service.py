import os
import time

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# Create Gemini client
client = genai.Client(api_key=API_KEY)


def explain_recommendation(data: dict) -> str:
    """
    Generate a human-readable explanation for a
    healthcare medicine redistribution recommendation.
    """

    prompt = f"""
You are an AI assistant for a healthcare resource management system.

Explain the following medicine redistribution recommendation
in simple, professional language for a healthcare administrator.

Recommendation details:

- Medicine: {data.get("medicine_name")}
- Receiving facility: {data.get("to_facility")}
- Donor facility: {data.get("from_facility")}
- Current stock at receiving facility: {data.get("current_stock")}
- Reorder level: {data.get("reorder_level")}
- Shortage: {data.get("shortage")}
- Predicted demand: {data.get("predicted_demand")}
- Transfer quantity: {data.get("transfer_quantity")}
- Distance between facilities: {data.get("distance_km")} km
- Stockout flag: {data.get("stockout_flag")}

Provide the explanation in four parts:

1. Why the receiving facility needs the medicine.
2. Why the donor facility was selected.
3. Why the proposed transfer quantity is reasonable.
4. A short final recommendation.

Important rules:
- Do not invent medical facts.
- Do not invent numerical values.
- Use only the information provided above.
- Clearly distinguish predictions from current inventory values.
- Keep the explanation concise and professional.
"""

    # Try the request up to 3 times
    for attempt in range(3):
        try:
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )

            return interaction.output_text

        except Exception as e:
            if attempt == 2:
                raise e

            print(
                f"Gemini request failed "
                f"(attempt {attempt + 1}/3). Retrying..."
            )

            time.sleep(5)