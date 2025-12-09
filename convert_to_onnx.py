import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import os

# Configuration
NUM_CLASSES = 15
MODEL_PATH = 'wheat_resnet50.pt'
ONNX_PATH = 'wheat_resnet50.onnx'

def convert_to_onnx():
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    
    # Initialize model architecture
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Load weights
    device = torch.device('cpu')
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Fix 'module.' prefix if present (handling DataParallel saves)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Create dummy input for tracing (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting to ONNX: {ONNX_PATH}...")
    
    # Export the model
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input (or a tuple for multiple inputs)
        ONNX_PATH,                  # where to save the model (can be a file or file-like object)
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=11,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                      'output': {0: 'batch_size'}}
    )
    
    print("Conversion complete!")
    print(f"ONNX model saved size: {os.path.getsize(ONNX_PATH) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_to_onnx()
