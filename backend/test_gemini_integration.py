import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Use ONLY stable, supported models
MODELS_TO_TRY = ["gemini-3-flash-preview"]

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


def get_gemini_recommendation(user_data):
    if not GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "GEMINI_API_KEY not found in environment variables",
        }

    weather = user_data.get("weather", {})
    weather_condition = weather.get("condition", "Clear")

    answers = user_data.get("questionnaire_answers", {})
    answers_str = (
        "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in answers.items()])
        if answers
        else "- No questionnaire data available"
    )

    prompt = f"""
You are an agricultural expert providing recommendations to a wheat farmer.

Farmer's Information:
- Location: {user_data.get('location', {}).get('city', 'Unknown')},
  {user_data.get('location', {}).get('region', 'Unknown')},
  {user_data.get('location', {}).get('country', 'Unknown')}
- Current Weather: {weather_condition}, Temperature: {weather.get('temp_c', 25.0)}°C
- Crop Condition: {user_data.get('crop_condition', 'Unknown')}
- Disease Detected: {user_data.get('disease_detected', 'None')}

Additional Farm Details:
{answers_str}

Return a detailed HTML response using:
- <h3> for section titles
- <ul><li> for points
- <strong> for emphasis

Sections:
1. Detailed Analysis
2. Immediate Actions Required
3. Long-term Preventive Measures
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.8,
            "maxOutputTokens": 2048,
        },
    }

    last_error = None

    for model_name in MODELS_TO_TRY:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"

        for attempt in range(3):
            try:
                print(f"Attempting model={model_name}, attempt={attempt + 1}")

                response = requests.post(
                    f"{url}?key={GEMINI_API_KEY}",
                    headers=headers,
                    json=payload,
                    timeout=60,  # REQUIRED
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise RuntimeError("Empty candidates")

                    text = (
                        candidates[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )

                    if text.strip():
                        return {"status": "success", "recommendation": text.strip()}

                # Retry only on transient failures
                if response.status_code in (429, 503):
                    backoff = 2**attempt
                    time.sleep(backoff)
                    last_error = f"{response.status_code} transient error"
                    continue

                # Permanent failure → stop retrying this model
                last_error = f"{response.status_code}: {response.text}"
                break

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                time.sleep(2)

            except Exception as e:
                last_error = str(e)
                break

    return {
        "status": "error",
        "message": f"Gemini API failure: {last_error}",
    }


def test_gemini_integration():
    print("Starting Gemini API Test...")
    res = get_gemini_recommendation(sample_user_data)

    if res["status"] == "success":
        print("Test Successful")
        print(res["recommendation"][:300])
    else:
        print("Test Failed:", res["message"])


if __name__ == "__main__":
    test_gemini_integration()
