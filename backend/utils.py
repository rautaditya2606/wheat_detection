import os
import json
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    # Using gemini-3-flash-preview as per documentation
    model = genai.GenerativeModel("gemini-3-flash-preview")


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
    Get treatment recommendations from Gemini LLM
    """
    if not gemini_api_key:
        return "LLM integration not configured. Please set GEMINI_API_KEY in .env file."

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

        # Generate content using Gemini
        response = model.generate_content(prompt)

        # Extract the generated text
        if response and hasattr(response, "text"):
            return response.text
        else:
            return "Failed to generate recommendations. Please try again later."

    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "An error occurred while generating recommendations. Please try again later."
