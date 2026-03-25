import torch
import clip
from PIL import Image
import os

class ModelGatekeeper:
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        self.device = device
            
        print(f"Loading CLIP model {model_name} on {self.device}...")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        except Exception as e:
            print(f"Failed to load CLIP: {e}. Falling back to CPU.")
            self.device = "cpu"
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Define the categories for validation
        self.labels = [
            "a photo of wheat", 
            "a photo of a crop", 
            "a photo of a plant",
            "a photo of a field",
            "an animal", 
            "a person", 
            "a car", 
            "a building",
            "random noise",
            "a diagram"
        ]
        self.text_tokens = clip.tokenize(self.labels).to(self.device)

    def is_valid_input(self, image_path, threshold=0.4):
        """
        Returns (is_valid, top_label, confidence)
        """
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits_per_image, _ = self.model(image, self.text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Find the top prediction
            top_idx = probs.argmax()
            top_label = self.labels[top_idx]
            confidence = probs[top_idx]
            
            # Logic: Valid if top prediction is wheat, crop, field or plant
            valid_keywords = ["wheat", "crop", "field", "plant"]
            is_valid = any(kw in top_label.lower() for kw in valid_keywords)
            
            # Even if it matches keywords, ensure confidence is reasonable
            if confidence < threshold:
                is_valid = False
                
            return is_valid, top_label, confidence
            
        except Exception as e:
            print(f"Error in Gatekeeper: {e}")
            return False, "error", 0.0

if __name__ == "__main__":
    # Test script
    gatekeeper = ModelGatekeeper()
    # Replace with a local sample if available
    test_path = "static/uploads/test.jpg" 
    if os.path.exists(test_path):
        valid, label, conf = gatekeeper.is_valid_input(test_path)
        print(f"Result: {'VALID' if valid else 'INVALID'} | Label: {label} | Conf: {conf:.2f}")
    else:
        print(f"Please place a test image at {test_path} to verify.")
