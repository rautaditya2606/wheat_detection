import os
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Sample user data (replace with actual data in a real scenario)
sample_user_data = {
    "user_id": "user123",
    "location": {
        "ip": "8.8.8.8",  # Example IP
        "city": "Mumbai",
        "region": "Maharashtra",
        "country": "IN",
        "loc": "19.0760,72.8777",  # Mumbai coordinates
        "timezone": "Asia/Kolkata"
    },
    "weather": {
        "temp_c": 28.0,
        "humidity": 70,
        "condition": "Partly cloudy",
        "wind_kph": 10.0,
        "precip_mm": 0.0,
        "last_updated": "2023-10-20 11:30"
    },
    "questionnaire_answers": {
        "farm_size": "2-5 hectares",
        "soil_type": "Black soil",
        "irrigation_type": "Drip irrigation",
        "previous_crop": "Soybean",
        "fertilizer_used": "Urea, DAP",
        "pest_issues": "Aphids, Rust"
    },
    "crop_condition": "Healthy",  # From image analysis
    "disease_detected": "None"     # From image analysis
}
def get_gemini_recommendation(user_data):
    """
    Send user data to Gemini 2.5 Flash model using direct REST API call.
    """
    try:
        # Debug: Print the received user_data
        print("Debug - User Data Received:", json.dumps(user_data, indent=2))
        
        # Ensure weather data has required fields
        weather = user_data.get('weather', {})
        weather_condition = weather.get('condition') or weather.get('conditions', 'Clear')
        
        # Format the prompt with user data
        prompt = f"""
        You are an agricultural expert providing recommendations to a wheat farmer.
        
        Farmer's Information:
        - Location: {user_data.get('location', {}).get('city', 'Unknown')}, {user_data.get('location', {}).get('region', 'Unknown')}, {user_data.get('location', {}).get('country', 'Unknown')}
        - Current Weather: {weather_condition}, Temperature: {weather.get('temp_c', weather.get('temperature', 25.0))}°C
        - Crop Condition: {user_data.get('crop_condition', 'Unknown')}
        - Disease Detected: {user_data.get('disease_detected', 'None')}
        
        Please provide:
        1. Brief analysis of the current farming conditions (max 3 sentences)
        2. Top 3 recommendations for crop management
        3. Any immediate actions needed (if any)
        4. Preventive measures for common issues in this region (max 3)
        
        Keep the response concise and actionable.
        """
        
        # API endpoint
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        # Request headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Request payload
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 1024
            }
        }
        
        # Debug: Print the request payload
        print("Debug - Sending request to Gemini API...")
        
        # Make the API request
        response = requests.post(
            f"{url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            try:
                # Try the new Gemini response format first
                candidates = result.get("candidates", [])
                if not candidates:
                    raise KeyError("No candidates found in response")

                content = candidates[0].get("content", {})
                text = ""

                # Handle both old and new formats
                if "parts" in content and isinstance(content["parts"], list):
                    text = content["parts"][0].get("text", "")
                elif "text" in content:
                    text = content["text"]
                else:
                    # Newer format: response is under 'candidates[0]["content"]["parts"][0]["text"]'
                    parts = content.get("parts", [])
                    if parts and "text" in parts[0]:
                        text = parts[0]["text"]
                    else:
                        text = str(content)

                if not text.strip():
                    raise KeyError("Empty text content from Gemini response")

                return {
                    "status": "success",
                    "recommendation": text.strip()
                }

            except Exception as e:
                error_msg = f"Error parsing Gemini API response: {str(e)}"
                print(error_msg)
                print("Response content:", json.dumps(result, indent=2))
                return {
                    "status": "error",
                    "message": error_msg,
                    "response": result
                }
        else:
            error_msg = f"Gemini API request failed with status {response.status_code}"
            print(error_msg)
            print("Response content:", response.text)
            return {
                'status': 'error',
                'message': error_msg,
                'status_code': response.status_code,
                'response': response.text
            }

    except Exception as e:
        import traceback
        error_msg = f"Error in get_gemini_recommendation: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {
            'status': 'error',
            'message': error_msg,
            'traceback': traceback.format_exc()
        }
def test_gemini_integration():
    """
    Test the Gemini API integration with sample data.
    """
    print("Testing Gemini API integration...\n")
    print(json.dumps(sample_user_data, indent=2))
    
    print("\nGetting recommendations from Gemini...")
    result = get_gemini_recommendation(sample_user_data)
    
    print("\nResponse from Gemini:")
    if result.get("status") == "success":
        print("\nRecommendations:")
        print("-" * 50)
        print(result["recommendation"])
        print("-" * 50)
    else:
        print(f"Error: {result['message']}")

if __name__ == "__main__":
    test_gemini_integration()
