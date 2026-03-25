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
        
        # Group labels for combined scoring
        self.wheat_labels = [
            "a photo of wheat crop",
            "a photo of wheat plant disease",
            "a photo of wheat leaves",
            "a photo of wheat field",
        ]
        self.non_wheat_labels = [
            "a photo of an animal", 
            "a photo of a person", 
            "a photo of a car", 
            "a photo of a building",
            "a diagram",
            "a photo of something else"
        ]
        
        self.all_labels = self.wheat_labels + self.non_wheat_labels
        self.text_tokens = clip.tokenize(self.all_labels).to(self.device)

    def is_valid_input(self, image_path, threshold=0.6):
        """
        Returns (is_valid, wheat_score, top_non_wheat_label)
        """
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits_per_image, _ = self.model(image, self.text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Sum all wheat label probabilities together for binary domain validation
            wheat_score = float(probs[:len(self.wheat_labels)].sum())
            
            # Find the top non-wheat label for debugging/feedback
            non_wheat_probs = probs[len(self.wheat_labels):]
            top_non_wheat_idx = non_wheat_probs.argmax()
            top_non_wheat_label = self.non_wheat_labels[top_non_wheat_idx]
            
            is_valid = wheat_score > threshold
                
            return is_valid, wheat_score, top_non_wheat_label
            
        except Exception as e:
            print(f"Error in Gatekeeper: {e}")
            return False, 0.0, "error"

if __name__ == "__main__":
    # Test script
    gatekeeper = ModelGatekeeper()
    # Replace with a local sample if available
    test_path = "static/uploads/test.jpg" 
    if os.path.exists(test_path):
        valid, score, top_label = gatekeeper.is_valid_input(test_path)
        print(f"Result: {'VALID' if valid else 'INVALID'} | Wheat Score: {score:.2f} | Top Non-Wheat: {top_label}")
    else:
        print(f"Please place a test image at {test_path} to verify.")
