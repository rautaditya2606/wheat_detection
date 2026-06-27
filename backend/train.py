import os
import re
import sys
import copy
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import timm
import mlflow
import mlflow.pytorch
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType

# Add backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Canonical class list
CLASS_NAMES = [
    'aphid', 'black_rust', 'blast', 'brown_rust', 'common_root_rot',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'mildew', 'mite',
    'septoria', 'smut', 'stem_fly', 'tan_spot', 'yellow_rust'
]
NUM_CLASSES = len(CLASS_NAMES)

# Patched LayerNorm for ONNX/TensorRT compatibility (from research notebook)
class LayerNormPrimitive(nn.Module):
    def __init__(self, weight, bias, eps):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(weight.clone()))
        self.register_parameter('bias', nn.Parameter(bias.clone()))
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == self.weight.shape[0]:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight * x + self.bias

def replace_layernorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            new_ln = LayerNormPrimitive(module.weight, module.bias, module.eps)
            setattr(model, name, new_ln)
        else:
            replace_layernorm(module)
    return model

def freeze_backbone(model):
    """Freeze all layers except the classification head."""
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True
    elif hasattr(model, 'classifier'):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, 'fc'):
        for p in model.fc.parameters():
            p.requires_grad = True

def unfreeze_all(model):
    """Unfreeze all layers."""
    for p in model.parameters():
        p.requires_grad = True

# Dataset classes
class CSVDataset(Dataset):
    """Reads (path, label) pairs from a CSV file."""
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        self.samples = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['path'], int(row['label'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class RetrainDataset(Dataset):
    """Loads exported user feedback images organized in an ImageFolder format."""
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        if os.path.exists(root_dir):
            for class_name in os.listdir(root_dir):
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    norm_name = class_name.lower().replace(" ", "_")
                    if norm_name in CLASS_NAMES:
                        label_idx = CLASS_NAMES.index(norm_name)
                        for fname in os.listdir(class_path):
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                self.samples.append((os.path.join(class_path, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    parser = argparse.ArgumentParser(description="Retrain ConvNeXt-Tiny model with combined production feedback.")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs.")
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Epochs to train head-only.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (8 is recommended to avoid OOM).")
    parser.add_argument("--checkpoint-path", type=str, 
                        default="/home/adityaraut/Documents/research_paper/non-leaky/convnext_tiny_clean/convnext_tiny_clean.pth",
                        help="Path to initial clean weights.")
    parser.add_argument("--splits-dir", type=str, 
                        default="/home/adityaraut/Documents/research_paper/non-leaky/splits",
                        help="Path to splits directory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up MLflow
    mlflow.set_experiment("Wheat_Disease_Retraining")

    # Data Transforms
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # 1. Load Datasets
    train_csv_path = os.path.join(args.splits_dir, "train.csv")
    val_csv_path = os.path.join(args.splits_dir, "val.csv")

    base_train_ds = CSVDataset(train_csv_path, transform=train_transform)
    val_ds = CSVDataset(val_csv_path, transform=val_transform)

    feedback_dir = os.path.join(current_dir, "data", "retrain_dataset")
    feedback_ds = RetrainDataset(feedback_dir, transform=train_transform)

    print(f"Baseline train dataset size: {len(base_train_ds)}")
    print(f"Production feedback dataset size: {len(feedback_ds)}")

    # Combine datasets
    if len(feedback_ds) > 0:
        train_ds = ConcatDataset([base_train_ds, feedback_ds])
        print(f"Combined train dataset size: {len(train_ds)}")
    else:
        train_ds = base_train_ds
        print("No production feedback dataset found; training with baseline data only.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Build Model
    print("Loading ConvNeXt-Tiny model...")
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=NUM_CLASSES)
    
    if os.path.exists(args.checkpoint_path):
        print(f"Loading weights from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Checkpoint not found. Initializing with random weights.")
        
    model.to(device)

    # Freeze backbone initially
    freeze_backbone(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_weights = None

    with mlflow.start_run() as run:
        mlflow.log_params({
            "total_epochs": args.epochs,
            "freeze_epochs": args.freeze_epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "feedback_samples_added": len(feedback_ds)
        })

        for epoch in range(1, args.epochs + 1):
            # Unfreeze backbone after freeze_epochs
            if epoch == args.freeze_epochs + 1:
                print(f"Unfreezing backbone at epoch {epoch}...")
                unfreeze_all(model)
                optimizer = optim.AdamW(model.parameters(), lr=args.lr)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_epochs)

            # Training phase
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_ds)
            scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_ds)
            val_acc = correct / total

            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

        print(f"Best validation accuracy achieved: {best_val_acc:.4f}")
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Load best weights
        model.load_state_dict(best_weights)

        # 3. Save PyTorch checkpoint
        output_dir = os.path.join(current_dir, "pt_models")
        os.makedirs(output_dir, exist_ok=True)
        pt_model_path = os.path.join(output_dir, "convnext_tiny_retrained.pth")
        torch.save({
            "state_dict": best_weights,
            "best_val_acc": best_val_acc,
            "class_names": CLASS_NAMES
        }, pt_model_path)
        print(f"Saved PyTorch checkpoint to: {pt_model_path}")
        mlflow.log_artifact(pt_model_path)

        # 4. Patch LayerNorm and export to ONNX
        print("Patching LayerNorm for ONNX compatibility...")
        model.eval()
        model.cpu()
        model = replace_layernorm(model)

        onnx_dir = os.path.join(current_dir, "onnx_models")
        os.makedirs(onnx_dir, exist_ok=True)
        raw_onnx_path = os.path.join(onnx_dir, "convnext_tiny_raw.onnx")
        
        dummy_input = torch.randn(1, 3, 224, 224)
        print("Exporting raw ONNX model...")
        torch.onnx.export(
            model, dummy_input, raw_onnx_path,
            opset_version=16,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )

        # 5. Simplify ONNX model
        simp_onnx_path = os.path.join(onnx_dir, "convnext_tiny_simplified.onnx")
        print("Simplifying ONNX model...")
        onnx_model = mlflow.pytorch.load_model # Wait, onnx simplify from file
        import onnx
        loaded_model = onnx.load(raw_onnx_path)
        model_simp, check = simplify(loaded_model)
        if check:
            onnx.save(model_simp, simp_onnx_path)
            print(f"Simplified ONNX model saved to: {simp_onnx_path}")
        else:
            print("Simplification check failed! Using raw ONNX for quantization.")
            simp_onnx_path = raw_onnx_path

        # 6. Quantize ONNX model to INT8
        quantized_onnx_path = os.path.join(onnx_dir, "convnext_tiny_clean_int8.onnx")
        print("Quantizing ONNX model to INT8...")
        quantize_dynamic(
            simp_onnx_path,
            quantized_onnx_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized INT8 ONNX model saved to: {quantized_onnx_path}")
        
        mlflow.log_artifact(quantized_onnx_path)
        print("Model retraining and export successfully logged to MLflow!")

if __name__ == "__main__":
    main()
