import os
import csv
import time
import pytest
import numpy as np
import onnxruntime as ort
from PIL import Image

# Absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
model_path = os.path.join(backend_dir, "onnx_models", "convnext_tiny_clean_int8.onnx")
val_csv_path = "/home/adityaraut/Documents/research_paper/non-leaky/splits/val.csv"

def preprocess_image(image_path):
    """Preprocess image to match the production pipeline."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    
    img_data = np.array(image).astype("float32")
    img_data /= 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1))  # (C, H, W)
    img_data = np.expand_dims(img_data, axis=0)    # (1, C, H, W)
    return img_data

@pytest.fixture
def session():
    """Load ONNX runtime session."""
    assert os.path.exists(model_path), f"ONNX model not found at {model_path}"
    return ort.InferenceSession(model_path)

def test_model_accuracy_threshold(session):
    """Verify that the model meets the minimum accuracy threshold on a subset of validation data."""
    assert os.path.exists(val_csv_path), f"Validation CSV not found at {val_csv_path}"
    
    # Read first 30 samples from the validation split
    samples = []
    with open(val_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 30:
                break
            samples.append((row['path'], int(row['label'])))
            
    assert len(samples) > 0, "No samples found in validation split"
    
    correct = 0
    input_name = session.get_inputs()[0].name
    
    for path, label in samples:
        assert os.path.exists(path), f"Validation image not found at {path}"
        img_data = preprocess_image(path)
        outputs = session.run(None, {input_name: img_data})
        pred = np.argmax(outputs[0], axis=1)[0]
        if pred == label:
            correct += 1
            
    accuracy = correct / len(samples)
    print(f"Regression Accuracy: {accuracy:.4f}")
    
    # Ensure accuracy is at least 80% (0.80) on validation subset
    assert accuracy >= 0.80, f"Model accuracy dropped to {accuracy:.4f} (expected >= 0.80)"

def test_model_cpu_latency(session):
    """Verify that average CPU inference latency is below 100ms."""
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy_input})
        
    # Benchmark
    start_time = time.time()
    runs = 50
    for _ in range(runs):
        session.run(None, {input_name: dummy_input})
    duration = time.time() - start_time
    
    avg_latency_ms = (duration / runs) * 1000
    print(f"Average CPU Latency: {avg_latency_ms:.2f}ms")
    
    # Assert latency is under 100ms
    assert avg_latency_ms < 100.0, f"Inference latency is too high: {avg_latency_ms:.2f}ms (expected < 100ms)"

def test_model_robustness(session):
    """Test model handles noisy and empty inputs without crashing."""
    input_name = session.get_inputs()[0].name
    
    # 1. Random noise input
    noise = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {input_name: noise})
    assert outputs[0].shape == (1, 15)
    
    # 2. Blank/Zero input
    zeros = np.zeros((1, 3, 224, 224), dtype=np.float32)
    outputs = session.run(None, {input_name: zeros})
    assert outputs[0].shape == (1, 15)
