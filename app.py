from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import os
from collections import OrderedDict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes
NUM_CLASSES = 15

# Disease labels
CLASS_NAMES = {
    0: 'Aphid', 1: 'Black Rust', 2: 'Blast', 3: 'Brown Rust',
    4: 'Common Root Rot', 5: 'Fusarium Head Blight', 6: 'Healthy',
    7: 'Leaf Blight', 8: 'Mildew', 9: 'Mite', 10: 'Septoria',
    11: 'Smut', 12: 'Stem fly', 13: 'Tan spot', 14: 'Yellow Rust'
}

# Load model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load state_dict and fix 'module.' prefix if present
state_dict = torch.load('wheat_resnet50.pt', map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Open image and preprocess
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, 1).item()
        predicted_label = CLASS_NAMES[predicted_class]

    return render_template('result.html', label=predicted_label, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
