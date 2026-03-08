import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_onnx(model_path, output_path):
    print(f"Quantizing ONNX model from {model_path}...")
    
    # Quantize the model
    quantize_dynamic(
        model_path,
        output_path,
        weight_type=QuantType.QUInt8
    )
    
    initial_size = os.path.getsize(model_path) / (1024 * 1024)
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Quantization complete!")
    print(f"Initial size: {initial_size:.2f} MB")
    print(f"Final size: {final_size:.2f} MB")
    print(f"Reduction: {((initial_size - final_size) / initial_size) * 100:.2f}%")

if __name__ == "__main__":
    quantize_onnx("backend/wheat_resnet50.onnx", "backend/wheat_resnet50_quantized.onnx")
