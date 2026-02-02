import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use stable OpenAI model
MODEL_TO_USE = "gpt-3.5-turbo"

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

    client = OpenAI(api_key=OPENAI_API_KEY)

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

    for attempt in range(3):
        try:
            print(f"Attempting OpenAI recommendation, attempt={attempt + 1}")

            response = client.chat.completions.create(
                model=MODEL_TO_USE,
                messages=[
                    {"role": "system", "content": "You are a helpful agricultural expert."},
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
