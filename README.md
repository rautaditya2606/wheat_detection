# Wheat Disease Intelligence Platform

> Production-style web platform for wheat disease detection with contextual AI recommendations.

**[Live Demo](https://wheat-detection-es09.onrender.com/)**

---

## Why This Project Matters

Most crop-disease demos stop at a class label. This system goes further:

1. **Stable Bulk Diagnostics (Up to 10 Images)**: Upload up to 10 images in one request. The backend processes each image sequentially (Cloudinary -> CLIP validation -> ONNX inference -> overlay) for predictable behavior on low-resource deployments.
2. **Heuristic Region Overlays**: Automatically generates visual disease highlighting using OpenCV-based color and texture analysis (e.g., reddish-brown rust pustules) to ground the model's prediction in visual evidence.
3. **On-Demand AI Recommendation**: Bulk analysis focuses on fast, stable classification. Users open an image result and click **Get Expert Recommendations** on `/result` to generate personalized guidance.
4. **Quantized Edge Inference**: Uses an INT8 quantized ResNet50 model (89% accuracy) optimized for 75% smaller footprint and faster CPU cold-starts.
5. **Human-in-the-Loop Feedback**: Captures user-corrected labels and stores them in Aiven ClickHouse, creating a verifiable ground-truth dataset for future active learning cycles.

This demonstrates end-to-end engineering across ML inference, backend architecture, cloud storage, managed database integration, and product UX — not just a notebook experiment.

---

## Architecture
```mermaid
graph TB
    %% Enhanced Style Definitions
    classDef userNode fill:#0ea5e9,stroke:#0284c7,stroke-width:1px,color:#fff,rx:10,ry:10;
    classDef backendNode fill:#f59e0b,stroke:#d97706,stroke-width:1px,color:#fff,rx:5,ry:5;
    classDef storageNode fill:#8b5cf6,stroke:#7c3aed,stroke-width:1px,color:#fff,rx:5,ry:5;
    classDef intelligenceNode fill:#10b981,stroke:#059669,stroke-width:1px,color:#fff,rx:5,ry:5;
    classDef uiNode fill:#ef4444,stroke:#dc2626,stroke-width:1px,color:#fff,rx:10,ry:10;

    %% Nodes
    User([fa:fa-user User Browser]):::userNode
    
    subgraph "Compute & Orchestration"
        Backend{Flask Route Layer}:::backendNode
        ResNet(ResNet50 Engine):::backendNode
        OpenCV(OpenCV Highlight):::backendNode
    end

    subgraph "Validation & Assets"
        Cloudinary[(Cloudinary)]:::storageNode
        CLIP{HF CLIP}:::storageNode
    end

    subgraph "Persistence & Intelligence"
        LLM(GPT-3.5):::intelligenceNode
        ClickHouse[(Aiven ClickHouse)]:::intelligenceNode
    end

    Response[[Results Grid + Detail View]]:::uiNode

    %% Styled Connections
    User ==>|Upload + Metadata| Backend
    Backend --> Cloudinary
    Backend --> CLIP
    Backend --> ResNet
    Backend --> OpenCV
    Backend --> ClickHouse
    User ==>|Open Result + Click Recommendation| LLM
    Backend ==>|JSON Response| Response

    %% Global Link Styling
    linkStyle default stroke:#64748b,stroke-width:1px,transition:0.3s;
```

### Flow Logic
1. **Single or Bulk Upload**: User uploads 1 image (`/predict`) or up to 10 images (`/predict-bulk`).
2. **Sequential Processing**: Each bulk image is processed in-order for stability in deployment environments.
3. **Validation First**: Cloudinary URL is created, then CLIP verifies wheat content; non-wheat images are rejected.
4. **Visual Diagnosis**: OpenCV heuristics generate highlighted overlays for infected regions.
5. **Recommendation on Demand**: AI recommendation is requested from the result page when the user clicks the recommendation button.

---

## Core Features

- **Stable Bulk Upload (Up to 10 Images)** — A simple and deployment-friendly bulk endpoint (`/predict-bulk`) that processes images sequentially and returns a unified JSON response.
- **Heuristic Disease Highlighting** — Visual overlays for 15+ diseases. Uses OpenCV color-masking and edge detection to show the user exactly where the model's prediction aligns with visual symptoms.
- **Mobile-Perfect Responsiveness** — Optimized Tailwind UI with adaptive grids (2-column mobile, 4-column desktop) and touch-optimized navigation for field use.
- **Multi-Stage AI Validation (CLIP Gatekeeper)** — Uses a specialized CLIP microservice to validate image content before full processing. Non-wheat images are automatically rejected and purged from storage.
- **High-Accuracy Inference** — Quantized ResNet50 ONNX model, 89% top-1 accuracy across 15 wheat disease classes.
- **INT8 Quantization** — Model size reduced from 90MB to 22.6MB (75% reduction), faster cold-starts on CPU deployment.
- **Context-Aware Recommendations** — Generated on demand from the result page (`/result`) using weather + user questionnaire + detected disease context.
- **Admin Monitoring Dashboard** — Secure `/admin` interface for auditing predictions and monitoring system health.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Flask, Flask-SQLAlchemy, Gunicorn |
| ML | PyTorch (training), ONNX, ONNX Runtime |
| Storage | Cloudinary (images), Aiven ClickHouse (feedback records) |
| Intelligence | OpenAI GPT-3.5-Turbo, WeatherAPI, GeoIP2 |
| Reporting | ReportLab (PDF generation) |
| Deployment | Docker, Render |

---

## ML Pipeline

### Model: ResNet50

ResNet50 is a strong fit for this problem:
- Residual connections give stable training on limited data
- Good accuracy/latency tradeoff for 224×224 crop images
- Mature ONNX export and Runtime ecosystem
- Easily portable to CPU-only production environments

**Achieves 89% top-1 accuracy across 15 wheat disease classes.**

### Optimization Pipeline

```
Train in PyTorch (ResNet50 classifier)
  -> Export to ONNX          [convert_to_onnx.py]
  -> Simplify ONNX graph     [optimize_onnx.py]
  -> INT8 dynamic quantization [quantize_onnx.py]
  -> Serve with ONNX Runtime
```

INT8 quantization via `onnxruntime.quantization.quantize_dynamic` (weight_type=QUInt8):
- 75% model size reduction (90MB → 22.6MB)
- Lower CPU memory pressure
- Better inference throughput on CPU-heavy deployments

## Cloudinary + ClickHouse: How Images and Labels Are Linked

Every uploaded image is stored in Cloudinary. The returned `secure_url` is saved directly into the ClickHouse feedback record — this URL is the link between the image asset and its label.

**Feedback schema:**

| Field | Type | Description |
|---|---|---|
| id | String(36) | UUID-formatted primary key |
| image_url | String | Cloudinary secure URL |
| predicted_class | String | Model output label |
| confidence | Float (nullable) | Prediction confidence percent |
| correct_class | String (nullable) | User-corrected label if flagged |
| is_correct | Boolean | User confirmation |
| created_at | DateTime | Timestamp |

This creates a reliable audit trail: prediction request → cloud image → persisted labeled record.

To reconstruct a training dataset from collected feedback:

```python
SELECT image_url, correct_class FROM feedback
# Download each image, save to dataset/{correct_class}/filename.jpg
# PyTorch ImageFolder reads folder names as class labels directly
```

---

## Project Structure

```
/
├── backend/
│   ├── app.py                          # Main Flask application
│   ├── models.py                       # Feedback schema (ClickHouse via SQLAlchemy)
│   ├── utils.py                        # Utility helpers (weather + misc)
│   ├── openai_integration.py           # OpenAI recommendation orchestration
│   ├── location.py                     # Geolocation logic
│   ├── convert_to_onnx.py              # PyTorch → ONNX export
│   ├── optimize_onnx.py                # ONNX graph simplification
│   ├── quantize_onnx.py                # INT8 dynamic quantization
│   ├── wheat_resnet50_quantized.onnx   # Production model
│   ├── templates/                      # Jinja2 HTML templates
│   └── static/                         # Tailwind CSS, JS, uploads
├── docs/                               # Feature documentation
├── Dockerfile
└── README.md
```

---

## Run Locally

**Prerequisites:** Python 3.10+, OpenAI API key, WeatherAPI key, Cloudinary account, Aiven ClickHouse instance

```bash
git clone https://github.com/rautaditya2606/wheat_detection.git
cd wheat_detection/backend
pip install -r requirements.txt
```

Create `backend/.env`:

```
OPENAI_API_KEY=your_openai_key
WEATHER_API_KEY=your_weather_key
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_key
CLOUDINARY_API_SECRET=your_cloudinary_secret
DATABASE_URL=your_aiven_clickhouse_url
SECRET_KEY=your_secret_key
```

```bash
python app.py
# http://localhost:10000
```

---

## Roadmap

- **Active learning pipeline** — scraped and user-corrected images stored in Cloudinary + ClickHouse, exported as an `ImageFolder`-compatible dataset for periodic fine-tuning
- **CI/CD triggered retraining** — GitHub Actions workflow that triggers fine-tuning automatically when verified sample count crosses a class threshold, exports updated ONNX model
- **Incremental fine-tuning** — new data mixed with original dataset samples to prevent catastrophic forgetting
- **Model observability** — latency tracking, confidence distribution monitoring, and class-level prediction drift detection
- **Dataset versioning** — track which feedback samples were used in each retraining run via `used_in_training` flag

---

Built by: [Aditya Raut](https://github.com/rautaditya2606) 
