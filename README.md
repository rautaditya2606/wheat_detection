# Wheat Health & Disease Diagnostics System

An advanced AI-driven platform for wheat crop disease classification, real-time environmental analysis, and personalized treatment recommendations.

## 🌟 Key Features

- **High-Accuracy AI Diagnostics**: Uses an optimized **ResNet50 ONNX model** to identify 15 different wheat conditions and diseases with high precision.
- **Expert AI Recommendations**: Integrated with **OpenAI (GPT-3.5-Turbo)** to provide step-by-step treatment plans, immediate actions, and long-term prevention strategies.
- **Context-Aware Analysis**: Automatically fetches real-time **weather conditions** (Temperature, Humidity, Precipitation) and **geolocation** to tailor recommendations to your specific environment.
- **Comprehensive PDF Reporting**: Generate and download professional PDF reports containing analysis results, uploaded images, and expert-level treatment plans.
- **Secure Farmer Accounts**: Personalized dashboard and secure authentication to track crop history and questionnaire data.
- **Responsive UI**: Built with a modern, mobile-first design using Tailwind CSS for field-access.

## 🛠️ Tech Stack

- **Backend**: Python (Flask, Uvicorn, ASGI)
- **AI/ML**: ONNX Runtime, PyTorch (Training), NumPy, Pillow
- **LLM**: OpenAI (GPT-3.5-Turbo)
- **Frontend**: HTML5, Tailwind CSS, JavaScript (ES6+)
- **Reporting**: ReportLab (PDF Generation)
- **External APIs**: OpenAI API, WeatherAPI, GeoIP2

## 🚀 Getting Started

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

## 📁 Project Structure

```text
/
├── backend/
│   ├── app.py              # Main ASGI/Flask Application
│   ├── models.py           # User & Database models
│   ├── utils.py            # OpenAI & Weather utilities
│   ├── location.py         # Geolocation logic
│   ├── wheat_resnet50.onnx # AI Detection Model
│   ├── templates/          # Jinja2 HTML Templates
│   ├── static/             # CSS (Tailwind), JS, and Uploads
│   └── tests/              # Performance & Integration tests
├── docs/                   # feature documentation
└── README.md
```

## 🧠 Model Information

The core classification engine uses a **Convolutional Neural Network (ResNet50)** fine-tuned on specialized wheat disease datasets. For production efficiency, the model is exported to **ONNX format**, ensuring sub-second inference times even on CPU environments.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
*Developed for efficient and sustainable wheat farming diagnostics.*
