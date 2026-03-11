# Wheat Health & Disease Diagnostics System

An advanced AI-driven platform for wheat crop disease classification, real-time environmental analysis, and personalized treatment recommendations.

**[Live Demo: Wheat Health & Disease Diagnostics](https://wheat-detection-1.onrender.com/)**

## Key Features

- **High-Accuracy AI Diagnostics**: Uses an optimized **Quantized ResNet50 ONNX model** to identify 15 different wheat conditions and diseases with high precision.
- **Human-in-the-Loop Feedback**: Integrated verification system that allows users to confirm or correct model predictions, building a high-quality dataset for continuous model improvement.
- **Advanced Model Optimization**: The detection engine is optimized using 8-bit integer (INT8) quantization, reducing model size by 75% while maintaining near-perfect prediction consistency.
- **Expert AI Recommendations**: Integrated with **OpenAI (GPT-3.5-Turbo)** to provide step-by-step treatment plans, immediate actions, and long-term prevention strategies.
- **Context-Aware Analysis**: Automatically fetches real-time **weather conditions** (Temperature, Humidity, Precipitation) and **geolocation** to tailor recommendations to your specific environment.
- **Comprehensive PDF Reporting**: Generate and download professional PDF reports containing analysis results, uploaded images, and expert-level treatment plans.
- **Secure Farmer Accounts**: Personalized dashboard and secure authentication to track crop history and questionnaire data.
- **Responsive UI**: Built with a modern, mobile-first design using Tailwind CSS for field-access.

## Tech Stack

- **Backend**: Python (Flask, Uvicorn, ASGI)
- **AI/ML**: ONNX Runtime (Quantized), PyTorch (Training), NumPy, Pillow, ONNX-Simplifier
- **LLM**: OpenAI (GPT-3.5-Turbo)
- **Frontend**: HTML5, Tailwind CSS, JavaScript (ES6+)
- **Reporting**: ReportLab (PDF Generation)
- **External APIs**: OpenAI API, WeatherAPI, GeoIP2

## Getting Started

### Prerequisites
- Python 3.10 or higher
- [OpenAI API Key](https://platform.openai.com/)
- [WeatherAPI Key](https://www.weatherapi.com/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rautaditya2606/wheat_detection.git
   cd wheat_detection
   ```

2. **Set up the backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the `backend/` directory:
   ```env
   SECRET_KEY=your_secret_key
   OPENAI_API_KEY=your_openai_api_key
   WEATHER_API_KEY=your_weather_api_key
   ```

### Running the App

Start the server from the `backend/` directory:
```bash
python app.py
```
The application will be accessible at `http://localhost:10000`.

## Project Structure

```text
/
├── backend/
│   ├── app.py              # Main ASGI/Flask Application
│   ├── models.py           # User & Database models
│   ├── utils.py            # OpenAI & Weather utilities
│   ├── location.py         # Geolocation logic
│   ├── wheat_resnet50_quantized.onnx # Optimized AI Detection Model
│   ├── model_feedback.csv  # Human-in-the-loop feedback data
│   ├── templates/          # Jinja2 HTML Templates
│   ├── static/             # CSS (Tailwind), JS, and Uploads
│   └── tests/              # Performance & Integration tests
├── docs/                   # feature documentation
└── README.md
```

## Model Information and Optimization

The core classification engine uses a **Convolutional Neural Network (ResNet50)** fine-tuned on specialized wheat disease datasets. 

### Performance Enhancements:
- **ONNX Quantization**: The model has been optimized using **INT8 Quantization**, reducing the file size from **90MB to 22.6MB (75% reduction)**. This ensures faster loading and reduced memory usage without compromising accuracy.
- **Improved Geolocation**: The system now supports automatic GPS tracking with human-readable city and location name display, providing a more intuitive user experience.
- **Human-in-the-Loop**: A feedback mechanism on the results page allows users to validate model outputs. Corrective feedback is logged to `model_feedback.csv`, providing a structured dataset to analyze edge cases and improve future model iterations.

## Usage Guide

1. **Sign Up/Log In**: Create an account to track your history and get personalized recommendations.
2. **Set Location**: Use the "Automatic GPS" or "Manual" mode in the questionnaire to provide environmental context.
3. **Analyze Image**: Upload a photo of the affected wheat.
4. **Review Results**: See the AI prediction and environmental analysis.
5. **Provide Feedback**: Help improve the model by confirming or correcting the result.
6. **Get Treatment Plan**: Click "Get Expert Recommendations" for a detailed, AI-generated management strategy.
7. **Export Report**: Download the full analysis and plan as a PDF.