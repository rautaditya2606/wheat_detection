034dcf5fc6cd4a3eaf1175716251910
# Wheat Disease Classifier

A web application that uses deep learning to identify wheat diseases from images of wheat crops and leaves.

## Features

- **Wheat-Specific Analysis**: Upload images of wheat crops or leaves only for accurate diagnosis.
- Get instant predictions about potential diseases
- View confidence scores for predictions
- Mobile-responsive design
- Drag and drop file upload

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rautaditya2606/wheat_detection.git
   cd wheat-disease-classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you have the model file `wheat_resnet50.pt` in the project root directory.

2. Start the Flask development server:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
wheat-disease-classifier/
├── app.py                 # Main Flask application
├── requirements.txt        # Python dependencies
├── wheat_resnet50.pt      # Pre-trained model
├── static/                # Static files (CSS, JS, uploads)
│   ├── style.css
│   └── script.js
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   └── result.html
└── README.md
```

## Model Information

The application uses a pre-trained ResNet50 model that has been fine-tuned for wheat disease classification. The model can identify 15 different wheat conditions including healthy plants and various diseases.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the researchers and developers who contributed to the wheat disease dataset and model training.
- Built with Flask, PyTorch, and Tailwind CSS.
