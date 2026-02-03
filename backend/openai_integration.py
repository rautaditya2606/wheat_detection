import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use stable OpenAI model
MODEL_TO_USE = "gpt-3.5-turbo"

# Initialize OpenAI client safely at import time so any
# library/env mismatch doesn't raise during a request.
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        # Log and continue; callers will see a clean error message instead
        print(f"Failed to initialize OpenAI client in openai_integration: {e}")

# Sample user data for testing
sample_user_data = {
    "user_id": "test_user_123",
    "location": {
        "city": "Pune",
        "region": "Maharashtra",
        "country": "India",
        "loc": "18.5204,73.8567",
        "timezone": "Asia/Kolkata",
    },
    "weather": {
        "temp_c": 28.5,
        "humidity": 65,
        "condition": "Partly cloudy",
        "wind_kph": 12.5,
        "precip_mm": 0.0,
        "last_updated": "2023-10-26 14:30",
    },
    "questionnaire_answers": {
        "soil_type": "Black Soil",
        "irrigation_method": "Drip Irrigation",
        "fertilizer_used": "NPK 19:19:19",
        "crop_stage": "Flowering",
        "sowing_date": "2023-11-15",
    },
    "crop_condition": "Yellowing leaves",
    "disease_detected": "Yellow Rust",
}


def get_openai_recommendation(user_data):
    if not OPENAI_API_KEY:
        return {
            "status": "error",
            "message": "OPENAI_API_KEY not found in environment variables",
        }

    if openai_client is None:
        # Either initialization failed or key is missing/invalid
        return {
            "status": "error",
            "message": "LLM integration is currently unavailable. Please check your OpenAI configuration.",
        }

    weather = user_data.get("weather", {})
    weather_condition = weather.get("condition", "Clear")

    answers = user_data.get("questionnaire_answers", {})
    answers_str = (
        "\n".join(
            [f"- {k.replace('_', ' ').title()}: {v}" for k, v in answers.items()]
        )
        if answers
        else "- No questionnaire data available"
    )

    prompt = f"""
You are an expert agricultural scientist and wheat production specialist. 
Provide a professional, highly structured recommendation for a farmer whose crop is affected.

FARMER'S CURRENT CONTEXT:
- Location: {user_data.get('location', {}).get('city', 'Unknown')}, {user_data.get('location', {}).get('country', 'Unknown')}
- Environmental Data: {weather_condition}, {weather.get('temp_c', 25.0)}°C, Humidity: {weather.get('humidity', 'N/A')}%
- Health Status: {user_data.get('crop_condition', 'Unknown')}
- Identified Issue: {user_data.get('disease_detected', 'None')}

ADDITIONAL FARM PARAMETERS:
{answers_str}

REQUIRED OUTPUT FORMAT:
Return the recommendation as a sequence of DIV blocks. Use the following structure exactly:

<div class="analysis-section mb-6">
  <h3 class="text-xl font-bold text-blue-800 border-b-2 border-blue-200 pb-2 mb-4">I. Scientific Analysis</h3>
  <p class="text-gray-700 leading-relaxed mb-4">[Provide a brief scientific explanation of the {user_data.get('disease_detected')} and how current weather ({weather_condition}) influences it.]</p>
</div>

<div class="actions-section mb-6 bg-red-50 p-4 rounded-lg border-l-4 border-red-500">
  <h3 class="text-xl font-bold text-red-800 mb-3">II. Immediate Rescue Actions</h3>
  <ul class="list-disc pl-5 space-y-2">
    <li><strong>[Action Name]:</strong> [Clear description of what to do now]</li>
    <li><strong>[Dosage/Application]:</strong> [Mention specific organic or chemical treatment types appropriate for this disease]</li>
  </ul>
</div>

<div class="prevention-section mb-6 bg-green-50 p-4 rounded-lg border-l-4 border-green-500">
  <h3 class="text-xl font-bold text-green-800 mb-3">III. Long-term Management Plan</h3>
  <ul class="list-disc pl-5 space-y-2">
    <li><strong>Soil Management:</strong> [Advice based on the provided soil type or general fertility]</li>
    <li><strong>Strategy:</strong> [Advice on irrigation or rotation mentioned in farm details]</li>
  </ul>
</div>

INSTRUCTIONS:
- Use professional yet accessible language.
- Ensure the HTML is valid.
- DO NOT use markdown symbols like # or * within the code.
"""

    for attempt in range(3):
        try:
            print(f"Attempting OpenAI recommendation, attempt={attempt + 1}")

            response = openai_client.chat.completions.create(
                model=MODEL_TO_USE,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural consultant specializing in wheat pathology and sustainable farming."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2048,
            )

            text = response.choices[0].message.content

            if text.strip():
                return {"status": "success", "recommendation": text.strip()}

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {
                    "status": "error",
                    "message": f"OpenAI API failure: {str(e)}",
                }

    return {
        "status": "error",
        "message": "OpenAI API failure after multiple attempts",
    }


def test_openai_integration():
    print("Starting OpenAI API Test...")
    res = get_openai_recommendation(sample_user_data)

    if res["status"] == "success":
        print("Test Successful")
        print(res["recommendation"][:300])
    else:
        print("Test Failed:", res["message"])


if __name__ == "__main__":
    test_openai_integration()
