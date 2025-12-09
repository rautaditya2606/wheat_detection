import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Sample user data for testing
sample_user_data = {
    "user_id": "test_user_123",
    "location": {
        "city": "Pune",
        "region": "Maharashtra",
        "country": "India",
        "loc": "18.5204,73.8567",
        "timezone": "Asia/Kolkata"
    },
    "weather": {
        "temp_c": 28.5,
        "humidity": 65,
        "condition": "Partly cloudy",
        "wind_kph": 12.5,
        "precip_mm": 0.0,
        "last_updated": "2023-10-26 14:30"
    },
    "questionnaire_answers": {
        "soil_type": "Black Soil",
        "irrigation_method": "Drip Irrigation",
        "fertilizer_used": "NPK 19:19:19",
        "crop_stage": "Flowering",
        "sowing_date": "2023-11-15"
    },
    "crop_condition": "Yellowing leaves",
    "disease_detected": "Yellow Rust"
}

def get_gemini_recommendation(user_data):
    """
    Get recommendations from Gemini API based on user data.
    """
    if not GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "GEMINI_API_KEY not found in environment variables"
        }

    try:
        # Debug: Print the received user_data
        print("Debug - User Data Received:", json.dumps(user_data, indent=2))
        
        # Ensure weather data has required fields
        weather = user_data.get('weather', {})
        weather_condition = weather.get('condition') or weather.get('conditions', 'Clear')
        
        # Extract questionnaire answers
        answers = user_data.get('questionnaire_answers', {})
        answers_str = "\n        ".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in answers.items()]) if answers else "- No questionnaire data available"

        # Format the prompt with user data
        prompt = f"""
        You are an agricultural expert providing recommendations to a wheat farmer.
        
        Farmer's Information:
        - Location: {user_data.get('location', {}).get('city', 'Unknown')}, {user_data.get('location', {}).get('region', 'Unknown')}, {user_data.get('location', {}).get('country', 'Unknown')}
        - Current Weather: {weather_condition}, Temperature: {weather.get('temp_c', weather.get('temperature', 25.0))}°C
        - Crop Condition: {user_data.get('crop_condition', 'Unknown')}
        - Disease Detected: {user_data.get('disease_detected', 'None')}
        
        Additional Farm Details:
        {answers_str}
        
        Please provide a comprehensive, detailed, and elaborative response in HTML format (do not use markdown code blocks, just raw HTML).
        The response should be in-depth, explaining the 'why' and 'how' for each recommendation.
        
        Use the following structure:
        
        <h3>Detailed Analysis</h3>
        <p>...provide a deep analysis of the conditions, explaining how the weather, location, and specific farm details interact with the detected disease or crop condition...</p>
        
        <h3>Comprehensive Recommendations</h3>
        <ul>
            <li><strong>Recommendation 1:</strong> ...provide detailed steps, dosage (if applicable), and timing...</li>
            <li><strong>Recommendation 2:</strong> ...explain the expected outcome and why this is recommended...</li>
            <li><strong>Recommendation 3:</strong> ...include alternative options if available...</li>
        </ul>
        
        <h3>Immediate Actions Required</h3>
        <ul>
            <li>...urgent steps to take immediately to mitigate damage...</li>
        </ul>
        
        <h3>Long-term Preventive Measures</h3>
        <ul>
            <li>...strategies to prevent recurrence in future seasons...</li>
        </ul>
        
        Make the content very informative, educational, and actionable for the farmer. Avoid brevity; prioritize clarity and depth.
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
                "maxOutputTokens": 4096
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
