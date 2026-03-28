import requests
import os

CLIP_URL = "https://clipmodel-wheat.onrender.com/verify-crop/"

def test_clip():
    print(f"Testing CLIP connection at {CLIP_URL}...")
    
    # Try to find a sample image
    sample_img = "backend/static/samples/aphid_30.png"
    if not os.path.exists(sample_img):
        print(f"Sample not found: {sample_img}")
        # Create a dummy file if sample doesn't exist
        with open("test_dummy.jpg", "wb") as f:
            f.write(b"dummy image data")
        sample_img = "test_dummy.jpg"

    try:
        with open(sample_img, 'rb') as f:
            files = {'file': (os.path.basename(sample_img), f, 'image/png' if sample_img.endswith('.png') else 'image/jpeg')}
            response = requests.post(CLIP_URL, files=files, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
    except Exception as e:
        print(f"Connection Error: {e}")
        print("\nMake sure your CLIP microservice is running on port 8000!")

if __name__ == "__main__":
    test_clip()
