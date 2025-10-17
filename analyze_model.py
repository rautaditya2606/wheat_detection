import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import sys

def analyze_model(model_path):
    try:
        # Load the model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        print("\n" + "="*50)
        print("MODEL ANALYSIS REPORT")
        print("="*50 + "\n")
        
        # Check if it's a state dict or full model
        if isinstance(model, dict):
            print("Model type: State dictionary")
            print(f"Number of parameter tensors: {len(model)}")
            
            # Calculate total parameters
            total_params = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
            print(f"\nTotal parameters: {total_params:,}")
            
            # Print layer names and shapes
            print("\nLayer shapes:")
            for name, param in model.items():
                if isinstance(param, torch.Tensor):
                    print(f"{name}: {tuple(param.shape)}")
                    
        elif isinstance(model, nn.Module):
            print("Model type: PyTorch Module")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")
            print("\nModel architecture:")
            print(model)
            
            # Try to get input shape (if possible)
            try:
                input_size = (3, 224, 224)  # Default for ResNet
                summary(model, input_size, device='cpu')
            except:
                print("\nCould not generate model summary. The model might require specific input shapes.")
                
        # Check for any custom attributes or metadata
        if hasattr(model, 'classes'):
            print(f"\nNumber of output classes: {len(model.classes) if model.classes else 'Unknown'}")
            
        if hasattr(model, 'input_size'):
            print(f"Input size: {model.input_size}")
            
        # Check for optimizer state (if saved with the model)
        if 'optimizer' in str(model):
            print("\nOptimizer state found in the model file.")
            
        # Check for training metadata
        if 'epoch' in str(model):
            print("\nTraining metadata found in the model file.")
            
    except Exception as e:
        print(f"\nError analyzing model: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have the required PyTorch version installed")
        print("2. The model might be saved in a custom format")
        print("3. The model might require specific dependencies")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_model.py <path_to_model.pt>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    print(f"Analyzing model: {model_path}")
    analyze_model(model_path)
