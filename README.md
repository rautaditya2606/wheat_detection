# Context-Aware Wheat Disease Diagnostics System

**An AI-powered crop health platform that goes beyond simple disease classification** — integrating computer vision, real-time environmental data, and LLM-based treatment synthesis to deliver personalized, field-specific recommendations for farmers.

**[🌾 Live Demo](https://wheat-detection-1.onrender.com/)** | **Built for SIH 2025 + Production Enhancements**

---

## The Problem

Traditional plant disease classifiers output a label and confidence score. But **farmers don't need labels — they need actionable treatment plans**. A fungal infection requires different management in humid vs. dry climates, in drip-irrigated vs. rain-fed fields, and varies based on soil type and crop rotation history. **Classification alone doesn't solve the problem.**

---

## The Solution: Multi-Modal Diagnostic Pipeline

This system fuses **three data sources** to generate contextualized recommendations:

### 1. **Computer Vision** → Disease Detection
- **ResNet50 (Quantized ONNX)**: 15-class wheat disease classification
- **INT8 Quantization**: 75% model size reduction (90MB → 22.6MB) for faster inference
- **Human-in-the-Loop Validation**: Users confirm/correct predictions → feedback logged to `model_feedback.csv` for continuous improvement

### 2. **Environmental Context** → Field Conditions
- **10-Question Assessment**: Captures irrigation type, soil characteristics, previous crop, fertilizer usage
- **Automatic GPS Tracking**: User location → OpenWeatherAPI → real-time temperature, humidity, precipitation
- **Geolocation Integration**: Human-readable city/region display

### 3. **LLM Synthesis** → Personalized Treatment Plans
- **GPT-3.5-Turbo Orchestration**: Combines disease prediction + field questionnaire + weather data
- **Structured Prompt Engineering**: Generates three-tier recommendations:
  - **Scientific Analysis**: Disease behavior in current environmental conditions
  - **Immediate Rescue Actions**: Inspection protocols, treatment options (organic + chemical)
  - **Long-Term Management**: Soil health, crop rotation, integrated pest management
- **PDF Report Generation**: Downloadable analysis with uploaded image, diagnosis, and full treatment plan

---

## Sample Output

**Input:**
- Disease: Aphid infestation
- Weather: Clear, 27.7°C, 12% humidity
- Field: Drip irrigation, loamy soil, previous crop: chickpea

**Generated Recommendation (excerpt):**
> **Scientific Analysis**  
> Aphids thrive in warm and dry environments. The current clear weather with low humidity (12%) favors infestations...
> 
> **Immediate Rescue Actions**  
> • Inspect population density on affected plants  
> • Apply neem oil or insecticidal soap (organic option)  
> • Consider pyrethroid-based insecticide at 2ml/L (chemical option)
> 
> **Long-Term Management**  
> • Introduce beneficial insects (ladybugs, lacewings) for natural control  
> • Implement crop rotation to break pest cycles  
> • Monitor soil health and incorporate organic matter

[📄 View Full Sample Report](docs/sample_report.pdf)

---

## Why This Approach Matters

| **Traditional Classifiers** | **This System** |
|------------------------------|-----------------|
| Output: "Aphid - 94% confidence" | Output: "Aphid detected in dry conditions (12% humidity). Apply neem oil within 24 hours. Here's why..." |
| Single data source (image only) | Multi-modal fusion (image + field data + weather) |
| Static recommendations | Context-aware treatment plans |
| No feedback loop | Human-in-the-loop validation + continuous learning |

---

## Tech Stack

**Backend & AI**
- Python (Flask, Uvicorn ASGI)
- ONNX Runtime (quantized inference), PyTorch (training), ONNX-Simplifier
- OpenAI API (GPT-3.5-Turbo for treatment synthesis)

**Data Integration**
- OpenWeatherAPI (real-time climate data)
- GeoIP2 (location services)
- ReportLab (PDF generation)

**Frontend**
- HTML5, Tailwind CSS (mobile-first design)
- JavaScript (ES6+, GPS integration)

**Deployment**
- Render (live production environment)
- SQLite (user accounts, feedback logging)

---

## Getting Started

### Prerequisites
- Python 3.10+
- [OpenAI API Key](https://platform.openai.com/)
- [WeatherAPI Key](https://www.weatherapi.com/)

### Installation
```bash
# Clone repository
git clone https://github.com/rautaditya2606/wheat_detection.git
cd wheat_detection/backend

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cat > .env << EOF
SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_api_key
WEATHER_API_KEY=your_weather_api_key
EOF

# Run application
python app.py
# Access at http://localhost:10000
```

---

## Project Architecture
```
wheat_detection/
├── backend/
│   ├── app.py                          # ASGI/Flask application + API routes
│   ├── models.py                       # SQLAlchemy user/database models
│   ├── utils.py                        # OpenAI prompt engineering + weather API
│   ├── location.py                     # GPS → geolocation → weather integration
│   ├── wheat_resnet50_quantized.onnx   # INT8 quantized ONNX model (22.6MB)
│   ├── model_feedback.csv              # Human-in-the-loop correction logs
│   ├── templates/                      # Jinja2 HTML (results, dashboard, etc.)
│   └── static/                         # Tailwind CSS, JS, user uploads
└── docs/                               # Sample outputs, architecture diagrams
```

---

## Usage Workflow

1. **Sign Up/Log In** → Secure account with personalized dashboard
2. **Complete Field Assessment** → 10-question form OR auto-GPS location
3. **Upload Wheat Image** → Drag-and-drop or camera capture
4. **Receive Diagnosis** → Disease classification + confidence score
5. **Validate Prediction** → Confirm/correct result (builds training data)
6. **Generate Treatment Plan** → Click "Get Expert Recommendations"
7. **Export Report** → Download PDF with image, analysis, and action plan

---

## Model Optimization Details

**Architecture Choice:**
- ResNet50 selected for proven performance on agricultural datasets
- Future consideration: EfficientNet-B0 for mobile deployment (2-3x faster inference)

**Quantization Pipeline:**
- **Pre-quantization**: 90MB FP32 model, ~150ms inference (CPU)
- **Post-quantization**: 22.6MB INT8 model, ~80ms inference (CPU)
- **Accuracy retention**: <1% degradation on validation set
- **Tools**: ONNX Runtime quantization, onnx-simplifier for graph optimization

**Human-in-the-Loop Data Collection:**
- Feedback mechanism on results page (thumbs up/down + optional correction)
- Logged format: `timestamp, predicted_class, user_correction, confidence, image_hash`
- Use case: Identify edge cases for model retraining, track real-world accuracy

---

## Key Learnings & Design Decisions

**Why multi-modal over pure vision:**
- Disease symptoms often ambiguous in images (early-stage rust vs. nutrient deficiency)
- Environmental factors change treatment efficacy (fungicides fail in rain, neem oil ineffective in cold)
- Farmers need *what to do*, not just *what it is*

**Why LLM integration:**
- Structured knowledge bases become outdated quickly
- GPT-3.5-Turbo adapts to novel conditions (e.g., "unseasonably dry March in Punjab")
- Prompt engineering allows injection of local agricultural practices

**Why quantization:**
- Target users: farmers with mid-range smartphones (limited bandwidth, storage)
- 75% size reduction = faster loading in rural areas with poor connectivity
- Inference speed matters for real-time field diagnostics

---

## Future Enhancements

- [ ] **Mobile App**: React Native wrapper for offline inference
- [ ] **EfficientNet Migration**: 2-3x faster inference for mobile deployment
- [ ] **MLOps Pipeline**: MLflow experiment tracking, automated retraining on feedback data
- [ ] **Multi-Language Support**: Hindi, Punjabi, Marathi voice/text interfaces
- [ ] **Yield Impact Tracking**: Longitudinal study of treatment effectiveness

---

## Contributing

This project was built for **SIH 2025** and extended with production features. Contributions welcome for:
- Additional crop support (rice, maize, cotton)
- Model accuracy improvements (submit to `model_feedback.csv` analysis)
- Regional treatment knowledge (local agricultural practices)

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

- **SIH 2025**: Initial problem statement on wheat disease classification
- **OpenAI**: GPT-3.5-Turbo API for treatment synthesis
- **WeatherAPI**: Real-time climate data integration
- **Agricultural Domain Experts**: Treatment validation and field testing feedback

---

**Built by [Aditya Raut](https://github.com/rautaditya2606)** | 2nd Year CSE (AI & Analytics) @ MIT ADT University  
*This project secured a GenAI Engineering internship demonstrating end-to-end ML system design capabilities.*
