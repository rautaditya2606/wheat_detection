import onnxruntime as ort
import numpy as np
import os
import sys

def test_inference(model_path):
    print(f"Testing inference on: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return False
    
    try:
        # Load the model
        session = ort.InferenceSession(model_path)
        print("Model loaded successfully.")
        
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Input Name: {input_name}")
        print(f"Input Shape: {input_shape}")
        
        # Handle dynamic axes for testing (batch size 1)
        test_shape = [1, 3, 224, 224]
        
        # Create a dummy input (matching ResNet50 input)
        dummy_input = np.random.randn(*test_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        print("Inference successful!")
        print(f"Output shape: {outputs[0].shape}")
        return True
    except Exception as e:
        print(f"Inference failed with error: {e}")
        return False

if __name__ == "__main__":
    model_to_test = "backend/wheat_resnet50_quantized.onnx"
    success = test_inference(model_to_test)
    if success:
        print("\nPASSED: The quantized model is functioning correctly.")
        sys.exit(0)
    else:
        print("\nFAILED: The quantized model is not working.")
        sys.exit(1)
