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
Return ONLY the HTML code. DO NOT include any markdown code blocks (like ```html), tags at the start/end of the output, or any conversational text.
Use the following structure exactly:

<div class="analysis-section mb-6">
  <h3 class="text-xl font-bold text-blue-800 border-b-2 border-blue-200 pb-2 mb-4">I. Scientific Analysis</h3>
  <p class="text-gray-700 leading-relaxed mb-4">[Provide a detailed scientific explanation of the {user_data.get('disease_detected')}. Explain how current weather ({weather_condition}, {weather.get('temp_c')}°C) affects the pathogen's lifecycle and spread in Pune's environment. Mention specific biological characteristics of the issue.]</p>
</div>

<div class="actions-section mb-6 bg-red-50 p-4 rounded-lg border-l-4 border-red-500">
  <h3 class="text-xl font-bold text-red-800 mb-3">II. Immediate Rescue Actions</h3>
  <ul class="list-disc pl-5 space-y-3">
    <li><strong>[Action Name]:</strong> [Provide 2-3 sentences explaining a specific, high-priority physical or cultural intervention.]</li>
    <li><strong>[Organic/Chemical Treatment]:</strong> [Recommend specific active ingredients (e.g., Propiconazole for rust, or Neem-based solutions) and explain exactly how they work against {user_data.get('disease_detected')}.]</li>
    <li><strong>[Dosage/Application]:</strong> [Provide detailed application instructions, including optimal timing (morning/evening) and any safety precautions regarding the weather.]</li>
  </ul>
</div>

<div class="prevention-section mb-6 bg-green-50 p-4 rounded-lg border-l-4 border-green-500">
  <h3 class="text-xl font-bold text-green-800 mb-3">III. Long-term Management Plan</h3>
  <ul class="list-disc pl-5 space-y-3">
    <li><strong>Soil & Nutrient Management:</strong> [Detailed advice on soil health, mentioning how specific nutrients or organic amendments can build crop resistance.]</li>
    <li><strong>Climate Adaptive Strategy:</strong> [Detailed advice on irrigation timing or crop rotation specifically designed to prevent the recurrence of {user_data.get('disease_detected')} in future seasons.]</li>
  </ul>
</div>

INSTRUCTIONS:
- Use professional yet accessible language.
- provide substantial, actionable details for each point while keeping the total length concise (around 300-400 words).
- Avoid generic advice; tailor every sentence to the disease, weather, and farm parameters provided.
- Ensure the HTML is valid and all tags are properly closed.
- DO NOT use any markdown symbols like #, *, or backticks anywhere in your response.
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

            # Clean up the output if the LLM still insists on markdown wrappers
            if text.startswith("```"):
                # Remove the first line if it contains "```html" or "```"
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove the last line if it's "```"
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

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
