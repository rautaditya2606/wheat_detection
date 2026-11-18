import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Any OpenAI-compatible endpoint: OpenRouter by default (LLM_MODEL=deepseek/deepseek-v4.1-flash, LLM_API_KEY=...).
# Plain openai SDK instead of litellm: litellm costs ~285MB RAM on import, over the 512MB free tier.
LLM_MODEL = (os.getenv("LLM_MODEL") or "").removeprefix("openrouter/")
LLM_API_BASE = os.getenv("LLM_API_BASE") or "https://openrouter.ai/api/v1"
_llm_client = None


def llm_complete(messages, **kw):
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=LLM_API_BASE, timeout=60)
    # reasoning tokens eat max_tokens (empty content) and add latency; this is plain text generation
    extra = {"reasoning": {"enabled": False}} if "openrouter" in LLM_API_BASE else None
    return _llm_client.chat.completions.create(model=LLM_MODEL, messages=messages, extra_body=extra, **kw)


def get_weather_data(location=None):
    """
    Fetch weather data using WeatherAPI
    If location is None, defaults to Pune, India
    """
    weather_api_key = os.getenv("WEATHER_API_KEY")
    if not weather_api_key:
        return None

    # Default to Pune if no location is provided
    if not location:
        location = "Pune, India"

    try:
        base_url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": weather_api_key, "q": location, "aqi": "no"}

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            location_data = data.get("location", {})

            return {
                "temperature": current.get("temp_c"),
                "humidity": current.get("humidity"),
                "conditions": current.get("condition", {}).get("text", "N/A"),
                "wind_speed": current.get("wind_kph"),
                "wind_dir": current.get("wind_dir", "N/A"),
                "precipitation": current.get("precip_mm", 0),
                "location": f"{location_data.get('name', 'Unknown')}, {location_data.get('country', 'Unknown')}",
                "feels_like": current.get("feelslike_c"),
                "cloud_cover": current.get("cloud"),
                "last_updated": current.get("last_updated", "N/A"),
            }
    except Exception as e:
        print(f"Error getting weather data: {e}")

    return None


def get_llm_recommendation(disease, questionnaire_data=None, weather_data=None):
    """
    Get treatment recommendations from the configured LLM
    """
    if not LLM_MODEL:
        return (
            "LLM integration is currently unavailable. "
            "Please set LLM_MODEL and the provider's API key."
        )

    try:
        # Prepare the prompt
        prompt = f"""You are an agricultural expert providing recommendations for wheat disease management.
        
Disease detected: {disease}

Additional information:"""

        if questionnaire_data:
            prompt += "\n\nFarm and Crop Details:"
            for key, value in questionnaire_data.items():
                if value:  # Only include non-empty responses
                    prompt += f"\n- {key.replace('_', ' ').title()}: {value}"

        if weather_data:
            prompt += "\n\nCurrent Weather Conditions:"
            for key, value in weather_data.items():
                prompt += f"\n- {key.replace('_', ' ').title()}: {value}"

        prompt += """

Please provide a detailed, step-by-step recommendation for treating and managing this disease. 
Include both immediate actions and long-term prevention strategies. 

IMPORTANT: Return your response in a clean, user-friendly HTML format.
Use <h3> for section titles, <ul> and <li> for points, and <strong> for emphasis.
Avoid using markdown symbols like # or * within the tags.

Example structure:
<div class="recommendation-card">
  <h3 class="text-xl font-bold text-indigo-800 mb-4 divider-bottom">Immediate Actions</h3>
  <ul class="list-disc pl-5 space-y-2 mb-6 text-gray-700">
    <li><strong>Action 1:</strong> Description...</li>
    <li><strong>Action 2:</strong> Description...</li>
  </ul>

  <h3 class="text-xl font-bold text-indigo-800 mb-4 divider-bottom">Long-term Prevention</h3>
  <ul class="list-disc pl-5 space-y-2 text-gray-700">
    <li>Prevention strategy...</li>
  </ul>
</div>
"""

        # Generate content via litellm
        response = llm_complete(
            messages=[
                {"role": "system", "content": "You are a helpful agricultural expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2048,
        )

        # Extract the generated text
        if response and response.choices:
            return response.choices[0].message.content
        else:
            return "Failed to generate recommendations. Please try again later."

    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "An error occurred while generating recommendations. Please try again later."
