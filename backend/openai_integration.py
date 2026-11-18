import os
import time
from dotenv import load_dotenv
from utils import llm_complete, LLM_MODEL

# Load environment variables
load_dotenv()

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


SYSTEM_PROMPT = "You are an expert agricultural consultant specializing in wheat pathology and sustainable farming."


def _build_prompt(user_data):
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
Provide a professional, highly structured, and concise recommendation for a farmer whose crop is affected.

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
  <p class="text-gray-700 leading-relaxed mb-4">[Provide a detailed, in-depth scientific explanation of the {user_data.get('disease_detected')}. Explain how current weather ({weather_condition}, {weather.get('temp_c')}°C) and humidity ({weather.get('humidity')}%) affects the pathogen's lifecycle, spore germination, and rate of spread in the {user_data.get('location', {}).get('city', 'local')} environment. Mention specific biological characteristics, host-pathogen interactions, and why the current conditions are particularly favorable or unfavorable for this disease.]</p>
</div>

<div class="actions-section mb-6 bg-red-50 p-4 rounded-lg border-l-4 border-red-500">
  <h3 class="text-xl font-bold text-red-800 mb-3">II. Immediate Rescue Actions</h3>
  <ul class="list-disc pl-5 space-y-3">
    <li><strong>[Action Name]:</strong> [Provide a thorough explanation (1-2 sentences) of a specific, high-priority physical or cultural intervention. Describe how to perform it and why it's effective.]</li>
    <li><strong>[Organic/Chemical Treatment]:</strong> [Recommend specific active ingredients (e.g., Propiconazole for rust, or Neem-based solutions) and explain in detail exactly how they work at a molecular or cellular level against {user_data.get('disease_detected')}.]</li>
    <li><strong>[Dosage/Application]:</strong> [Provide comprehensive application instructions, including precise dosage, water mixing ratios, optimal timing (morning/evening), frequency of application, and critical safety precautions regarding weather and personal protection.]</li>
  </ul>
</div>

<div class="prevention-section mb-6 bg-green-50 p-4 rounded-lg border-l-4 border-green-500">
  <h3 class="text-xl font-bold text-green-800 mb-3">III. Long-term Management Plan</h3>
  <ul class="list-disc pl-5 space-y-3">
    <li><strong>Soil & Nutrient Management:</strong> [Extensive advice on soil health and fertility. Mention how specific micronutrients (like Zinc or Boron) or organic amendments (like vermicompost) can specifically build physiological resistance against {user_data.get('disease_detected')}.]</li>
    <li><strong>Climate Adaptive Strategy:</strong> [Detailed, multi-year advice on irrigation systems (e.g., switching to drip), crop rotation partners, or resistant variety selection specifically designed to break the lifecycle of {user_data.get('disease_detected')} and ensure long-term wheat farm resilience.]</li>
  </ul>
</div>

INSTRUCTIONS:
- Use professional yet accessible language.
- give specific, actionable details for each point. Keep it concise: around 250-350 words in total.
- Avoid generic advice; tailor every sentence to the disease, weather, and farm parameters provided.
- Ensure the HTML is valid and all tags are properly closed.
- DO NOT use any markdown symbols like #, *, or backticks anywhere in your response.
"""
    return prompt


def stream_recommendation(user_data):
    """Yield the recommendation HTML in chunks as the model writes it (no retries once started)."""
    resp = llm_complete(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(user_data)},
        ],
        temperature=0.4,
        max_tokens=900,
        stream=True,
    )
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_openai_recommendation(user_data):
    if not LLM_MODEL:
        return {
            "status": "error",
            "message": "LLM_MODEL not set. Configure LLM_MODEL and the provider's API key.",
        }

    prompt = _build_prompt(user_data)

    for attempt in range(3):
        try:
            print(f"Attempting LLM recommendation, attempt={attempt + 1}")

            response = llm_complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=900,
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
                    "message": f"LLM API failure: {str(e)}",
                }

    return {
        "status": "error",
        "message": "LLM API failure after multiple attempts",
    }


def test_openai_integration():
    print("Starting LLM API Test...")
    res = get_openai_recommendation(sample_user_data)

    if res["status"] == "success":
        print("Test Successful")
        print(res["recommendation"][:300])
    else:
        print("Test Failed:", res["message"])


if __name__ == "__main__":
    test_openai_integration()
