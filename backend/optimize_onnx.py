import onnx
from onnxsim import simplify
import os

def optimize_onnx(model_path, output_path):
    print(f"Loading ONNX model from {model_path}...")
    model = onnx.load(model_path)
    
    # Simplify the model
    print("Simplifying the model...")
    model_simp, check = simplify(model)
    
    if not check:
        print("Simplified model validation failed!")
        return
    
    # Save the simplified model
    onnx.save(model_simp, output_path)
    
    initial_size = os.path.getsize(model_path) / (1024 * 1024)
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Optimization complete!")
    print(f"Initial size: {initial_size:.2f} MB")
    print(f"Final size: {final_size:.2f} MB")
    print(f"Reduction: {((initial_size - final_size) / initial_size) * 100:.2f}%")

if __name__ == "__main__":
    optimize_onnx("backend/wheat_resnet50.onnx", "backend/wheat_resnet50_optimized.onnx")
