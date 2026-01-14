import os
import json
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# List of models to try in order of preference
MODELS_TO_TRY = ["gemini-3-flash-preview", "gemini-1.5-flash", "gemini-1.5-pro"]

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
    """
    Get recommendations from Gemini API based on user data.
    Implements model fallback and retries for robustness.
    """
    if not GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "GEMINI_API_KEY not found in environment variables",
        }

    try:
        # Prepare weather context
        weather = user_data.get("weather", {})
        weather_condition = weather.get("condition") or weather.get("conditions", "Clear")

        # Extract farm details
        answers = user_data.get("questionnaire_answers", {})
        answers_str = (
            "\n        ".join(
                [f"- {k.replace('_', ' ').title()}: {v}" for k, v in answers.items()]
            )
            if answers
            else "- No questionnaire data available"
        )

        # Build prompt
        prompt = f"""
        You are an agricultural expert providing recommendations to a wheat farmer.
        
        Farmer's Information:
        - Location: {user_data.get('location', {}).get('city', 'Unknown')}, {user_data.get('location', {}).get('region', 'Unknown')}, {user_data.get('location', {}).get('country', 'Unknown')}
        - Current Weather: {weather_condition}, Temperature: {weather.get('temp_c', 25.0)}°C
        - Crop Condition: {user_data.get('crop_condition', 'Unknown')}
        - Disease Detected: {user_data.get('disease_detected', 'None')}
        
        Additional Farm Details:
        {answers_str}
        
        Please provide a comprehensive, detailed, and elaborative response in HTML format.
        IMPORTANT: Return the response in a clean, user-friendly HTML structure.
        Use <h3> for section titles, <ul> and <li> for points, and <strong> for emphasis.
        
        Sections to include:
        1. <h3>Detailed Analysis</h3> (How conditions like {weather_condition} and {weather.get('temp_c', 25.0)}°C affect this disease)
        2. <h3>Immediate Actions Required</h3> (Urgent tasks, dosage, timing)
        3. <h3>Long-term Preventive Measures</h3> (Future seasons)
        
        Style the HTML with class names like 'section-title', 'recommendation-list', 'action-item'.
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
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            
            # Reduce retries and use shorter timeout for faster feedback
            for attempt in range(2):
                try:
                    print(f"Debug - Attempting {model_name} (Attempt {attempt + 1})...")
                    response = requests.post(
                        f"{url}?key={GEMINI_API_KEY}", 
                        headers=headers, 
                        json=payload, 
                        timeout=15
                    )

                    if response.status_code == 200:
                        result = response.json()
                        candidates = result.get("candidates", [])
                        if not candidates:
                            raise KeyError("No candidates in response")
                        
                        part = candidates[0].get("content", {}).get("parts", [{}])[0]
                        text = part.get("text", "")
                        
                        if text.strip():
                            return {"status": "success", "recommendation": text.strip()}
                    
                    elif response.status_code in [503, 429]:
                        # Backoff
                        sleep_time = (2 ** attempt) + 1
                        print(f"Debug - {model_name} busy. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        last_error = f"Status {response.status_code}"
                    else:
                        last_error = f"Status {response.status_code}: {response.text}"
                        break # Try next model for 4xx errors
                        
                except Exception as e:
                    print(f"Debug - Request error with {model_name}: {str(e)}")
                    last_error = str(e)
                    time.sleep(1)

        return {
            "status": "error",
            "message": f"AI service overloaded. Please try again soon. ({last_error})",
        }

    except Exception as e:
        return {"status": "error", "message": f"System error: {str(e)}"}


def test_gemini_integration():
    """Test function for command line validation"""
    print("Starting Gemini API Test...")
    res = get_gemini_recommendation(sample_user_data)
    if res["status"] == "success":
        print("Test Successful!")
        print(res["recommendation"][:200] + "...")
    else:
        print(f"Test Failed: {res['message']}")


if __name__ == "__main__":
    test_gemini_integration()
