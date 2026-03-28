import requests
import time

BASE_URL = "https://rautaditya2606-clipmodel-wheat.hf.space"
HEALTH_URL = f"{BASE_URL}/health"
VALIDATE_URL = f"{BASE_URL}/validate/"

# A real public image of wheat for testing URL validation
TEST_IMAGE_URL = "https://res.cloudinary.com/dbzgvzeeh/image/upload/v1740428905/wheat_disease/vpxis1jlv93gup2cliv3.png"

def test_hf_space():
    print(f"--- Testing HF Space Health: {HEALTH_URL} ---")
    try:
        health_resp = requests.get(HEALTH_URL, timeout=30)
        print(f"Health Status: {health_resp.status_code}")
        print(f"Health JSON: {health_resp.json()}")
    except Exception as e:
        print(f"Health Check Failed: {e}")

    print(f"\n--- Testing HF Space Validation (via URL): {VALIDATE_URL} ---")
    try:
        # User mentioned: "test /validate with a Cloudinary image URL"
        # We received 405 Method Not Allowed on POST (JSON) and GET (URL param). 
        # This usually means /validate might not have a trailing slash or uses different property names.
        
        # Test 1: POST (JSON) - ensuring no trailing slash mismatch
        TEST_URL_NO_SLASH = VALIDATE_URL.rstrip('/')
        payload = {"image_url": TEST_IMAGE_URL}
        print(f"Testing POST (JSON) to {TEST_URL_NO_SLASH}...")
        resp = requests.post(TEST_URL_NO_SLASH, json=payload, timeout=30)
        print(f"Status: {resp.status_code}, Resp: {resp.text}")

        # Test 2: POST (Multipart/form-data) - ensuring it's not looking for a file upload
        print(f"\nTesting POST (Multipart) to {VALIDATE_URL}...")
        resp = requests.post(VALIDATE_URL, data={"image_url": TEST_IMAGE_URL}, timeout=30)
        print(f"Status: {resp.status_code}, Resp: {resp.text}")

    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_hf_space()
