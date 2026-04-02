import os
import sys

# Add the directory containing this file to the Python path
# This ensures that internal imports work correctly when running from the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    make_response,
    session,
    send_from_directory,
)
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user,
)
import time
import uvicorn
import re
import requests
import json
import uuid
import threading
import queue
import numpy as np
import onnxruntime as ort
import cloudinary
import cloudinary.uploader
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image
from collections import OrderedDict, deque
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from werkzeug.utils import secure_filename
from functools import wraps
from models import user_db, User, db, Feedback
from user_data import user_data, QUESTIONNAIRE
from utils import get_weather_data, get_llm_recommendation
from location import location_bp, get_ip_geolocation, reverse_geocode
from overlay_utils import highlight_infection

load_dotenv()

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9]+$")

# CLIP Microservice Configuration
CLIP_VERIFY_URL = os.getenv("CLIP_VERIFY_URL", "http://127.0.0.1:8000/verify-crop/")

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)


def get_current_user_weather():
    """Helper to get weather data based on current user's saved location."""
    location_query = None
    if current_user.is_authenticated:
        if (
            current_user.location_type == "automatic"
            and current_user.latitude
            and current_user.longitude
        ):
            location_query = f"{current_user.latitude},{current_user.longitude}"
        elif current_user.manual_location:
            location_query = current_user.manual_location

    return get_weather_data(location=location_query)


app = Flask(__name__)
CORS(app, supports_credentials=True)

# Global variables for bulk processing
processing_queue = queue.Queue()
# dictionary to store results: {batch_id: {image_id: {result_data}}}
batch_results = {}
# Thread-safe lock for model inference (sequential processing constraint)
model_lock = threading.Lock()


def bg_worker():
    """Background worker that processes images from the queue one by one."""
    while True:
        try:
            task = processing_queue.get()
            if task is None:
                break

            batch_id = task["batch_id"]
            image_id = task["image_id"]
            filepath = task["filepath"]
            cloudinary_url = task.get("cloudinary_url")
            user_id = task.get("user_id")

            # Use app context for DB operations if needed
            with app.app_context():
                try:
                    # 0. CLIP validation (mirror single-upload pipeline)
                    try:
                        if cloudinary_url:
                            val_url = CLIP_VERIFY_URL.rstrip("/")
                            val_response = requests.post(
                                val_url, json={"image_url": cloudinary_url}, timeout=45
                            )

                            if val_response.status_code == 200:
                                val_data = val_response.json()
                                is_valid = val_data.get("is_valid", True)
                                wheat_score = val_data.get("wheat_score", 0.0)

                                if not is_valid:
                                    app.logger.warning(
                                        f"[BULK] CLIP rejected {image_id} (score={wheat_score})"
                                    )
                                    if batch_id not in batch_results:
                                        batch_results[batch_id] = {}
                                    batch_results[batch_id][image_id] = {
                                        "image_id": image_id,
                                        "status": "failed",
                                        "error": f"Not a wheat image (confidence: {wheat_score * 100:.2f}%)",
                                    }
                                    processing_queue.task_done()
                                    continue
                            else:
                                app.logger.error(
                                    f"[BULK] CLIP API error {val_response.status_code}"
                                )
                        else:
                            app.logger.warning(
                                "[BULK] Missing Cloudinary URL, skipping CLIP"
                            )
                    except Exception as ce:
                        app.logger.error(f"[BULK] CLIP validation error: {str(ce)}")
                        # Fail-open (same as single flow)

                    # 1. Classify image (Sequential processing with model_lock)
                    image = Image.open(filepath).convert("RGB")
                    processed_data = preprocess_image(image)

                    with model_lock:
                        ort_inputs = {ort_session.get_inputs()[0].name: processed_data}
                        ort_outs = ort_session.run(None, ort_inputs)

                    logits = ort_outs[0]
                    # Softmax workaround for flattened arrays
                    flat_logits = logits.flatten()
                    exp_val = np.exp(flat_logits - np.max(flat_logits))
                    probs = exp_val / np.sum(exp_val)

                    class_idx = int(np.argmax(probs))
                    label = CLASS_NAMES.get(class_idx, "Unknown")
                    confidence = float(np.max(probs))

                    # 3. Create Highlighted Overlay for infected images
                    highlighted_url = None
                    if label != "Healthy":
                        try:
                            highlighted_filename = f"highlighted_{os.path.basename(filepath)}"
                            highlighted_path = os.path.join(app.config["UPLOAD_FOLDER"], highlighted_filename)
                            if highlight_infection(filepath, label, highlighted_path):
                                highlighted_url = f"/uploads/{highlighted_filename}"
                                app.logger.info(f"[BULK] Highlighted image saved to {highlighted_path}")
                        except Exception as he:
                            app.logger.error(f"[BULK] Overlay error: {str(he)}")

                    # 4. Store result
                    if batch_id not in batch_results:
                        batch_results[batch_id] = {}

                    result_entry = {
                        "image_id": image_id,
                        "status": "completed",
                        "label": label,
                        "confidence": f"{confidence * 100:.2f}%",
                        "cloudinary_url": cloudinary_url,
                        "highlighted_url": highlighted_url,
                        "timestamp": time.time(),
                    }
                    batch_results[batch_id][image_id] = result_entry

                    # 5. Save to DB with user_id mapping
                    # Note: Feedback model currently doesn't have a user_id column
                    # so we'll skip passing it until the schema is updated.
                    new_feedback = Feedback(
                        image_url=cloudinary_url, 
                        predicted_class=label, 
                        confidence=float(confidence * 100),
                        is_correct=True
                    )
                    db.session.add(new_feedback)
                    db.session.commit()
                    result_entry["feedback_id"] = new_feedback.id

                except Exception as e:
                    app.logger.error(f"Error in bg_worker for {image_id}: {str(e)}")
                    if batch_id not in batch_results:
                        batch_results[batch_id] = {}
                    batch_results[batch_id][image_id] = {
                        "error": str(e),
                        "status": "failed",
                        "image_id": image_id,
                    }

            processing_queue.task_done()
        except Exception as e:
            app.logger.error(f"Worker thread error: {str(e)}")


# Start the background worker thread
threading.Thread(target=bg_worker, daemon=True).start()

# Initialize ThreadPoolExecutor for background/parallel tasks
executor = ThreadPoolExecutor(max_workers=4)

# Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 180,
}
db.init_app(app)

# Create tables in app context
with app.app_context():
    db.create_all()

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key-change-in-production")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["WTF_CSRF_TIME_LIMIT"] = 3600  # 1 hour CSRF token expiration

# Register blueprints
app.register_blueprint(location_bp, url_prefix="")

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return user_db.get_user_by_id(user_id)


# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Ensure upload and data directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("data", exist_ok=True)


# Admin authentication helper
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.authorization
        if not auth or not (
            auth.username == os.getenv("ADMIN_USERNAME")
            and auth.password == os.getenv("ADMIN_PASSWORD")
        ):
            return make_response(
                "Could not verify your access level for that URL.\n"
                "You have to login with proper credentials",
                401,
                {"WWW-Authenticate": 'Basic realm="Login Required"'},
            )
        return f(*args, **kwargs)

    return decorated_function


# Disease labels
CLASS_NAMES = {
    0: "Aphid",
    1: "Black Rust",
    2: "Blast",
    3: "Brown Rust",
    4: "Common Root Rot",
    5: "Fusarium Head Blight",
    6: "Healthy",
    7: "Leaf Blight",
    8: "Mildew",
    9: "Mite",
    10: "Septoria",
    11: "Smut",
    12: "Stem fly",
    13: "Tan spot",
    14: "Yellow Rust",
}

# Load ONNX model
try:
    # Use absolute path based on current file location
    onnx_model_path = os.path.join(
        current_dir, "onnx_models", "wheat_resnet50_quantized.onnx"
    )
    ort_session = ort.InferenceSession(onnx_model_path)
    print(f"Successfully loaded ONNX model from: {onnx_model_path}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    ort_session = None


def preprocess_image(image):
    # Resize to 224x224
    image = image.resize((224, 224))

    # Convert to numpy array and normalize
    # PIL image is RGB, 0-255
    img_data = np.array(image).astype("float32")

    # Normalize to 0-1
    img_data /= 255.0

    # Normalize with mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_data = (img_data - mean) / std

    # Transpose to (Channels, Height, Width) -> (3, 224, 224)
    img_data = img_data.transpose(2, 0, 1)

    # Add batch dimension -> (1, 3, 224, 224)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data


# Authentication Routes


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = user_db.get_user_by_username(username)

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password")

        if not username:
            flash("Username is required", "danger")
        elif not USERNAME_PATTERN.fullmatch(username):
            flash("Username can only contain letters and numbers", "danger")
        elif user_db.get_user_by_username(username):
            flash("Username already exists", "danger")
        else:
            user_db.add_user(username, password, email)
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# Admin Routes


@app.route("/admin")
@admin_required
def admin_panel():
    # Show all images so admin can monitor everything and delete bad data
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).all()
    return render_template("admin.html", feedbacks=feedbacks, current_user=current_user)


@app.route("/admin/delete/<feedback_id>", methods=["POST"])
@admin_required
def admin_delete(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)

    # 1. Attempt to delete from Cloudinary if it's a URL
    if feedback.image_url and feedback.image_url.startswith("http"):
        try:
            # Extract public_id from cloudinary URL
            # Format: .../folder/public_id.jpg
            public_id = feedback.image_url.split("/")[-1].split(".")[0]
            # If it's in a folder, we might need 'wheat_disease/' prefix
            cloudinary.uploader.destroy(f"wheat_disease/{public_id}")
            app.logger.info(f"Deleted Cloudinary resource: {public_id}")
        except Exception as e:
            app.logger.error(f"Failed to delete from Cloudinary: {e}")

    # 2. Delete from ClickHouse
    db.session.delete(feedback)
    db.session.commit()

    flash("Entry and image deleted successfully.", "warning")
    return redirect(url_for("admin_panel"))


# Main Routes


@app.route("/")
def index():
    return render_template("index.html", current_user=current_user)


@app.route("/result")
def result():
    feedback_id = request.args.get("feedback_id")

    # Guard against invalid ids coming from frontend (e.g., 'undefined')
    if not feedback_id or feedback_id in ["undefined", "null"]:
        feedback_id = None

    # 1) If feedback_id present → ALWAYS use DB (bulk flow)
    confidence_param = request.args.get("confidence")
    label_param = request.args.get("label")

    if feedback_id or (label_param and confidence_param):
        try:
            feedback = Feedback.query.get(feedback_id) if feedback_id else None
        except Exception:
            feedback = None

        # PRIORITY: URL params (most reliable for bulk)
        label = label_param if label_param else (feedback.predicted_class if feedback else "Unknown")
        confidence = confidence_param if confidence_param else (f"{feedback.confidence*100:.2f}%" if feedback and feedback.confidence else "N/A")

        # Image Logic for Bulk
        cloudinary_url = request.args.get("cloudinary_url") or (feedback.image_url if feedback else "")
        highlighted_path = request.args.get("highlighted_url") or ""
        image_path = cloudinary_url # Fallback for template
        
        weather_data = get_current_user_weather() if current_user.is_authenticated else {}
        cloudinary_error = None
    else:
        # 2) Session fallback (single upload flow)
        result_data = session.get("analysis_result")

        if result_data:
            label = result_data.get("label", "Unknown")
            confidence = result_data.get("confidence", "N/A")
            image_path = result_data.get("image_path", "")
            highlighted_path = result_data.get("highlighted_path", "")
            cloudinary_url = result_data.get("cloudinary_url", "")
            cloudinary_error = result_data.get("cloudinary_error")
            weather_data = result_data.get("weather_data", {})
            feedback_id = result_data.get("feedback_id", feedback_id)
        else:
            label = "Unknown"
            confidence = "N/A"
            image_path = ""
            highlighted_path = ""
            cloudinary_url = ""
            cloudinary_error = None
            weather_data = {}

    # Check if feedback has already been submitted for this record
    feedback_submitted = False
    if feedback_id and session.get(f"feedback_submitted_{feedback_id}"):
        feedback_submitted = True

    return render_template(
        "result.html",
        label=label,
        confidence=confidence,
        image_path=image_path,
        highlighted_path=highlighted_path,
        cloudinary_url=cloudinary_url,
        cloudinary_error=cloudinary_error,
        feedback_id=feedback_id,
        weather_data=weather_data,
        feedback_submitted=feedback_submitted,
        current_user=current_user,
    )


@app.route("/questionnaire")
@login_required
def questionnaire():
    # Get weather data for the questionnaire based on saved location
    weather_data = get_current_user_weather()

    return render_template(
        "questionnaire.html",
        weather_data=weather_data,
        questionnaire=QUESTIONNAIRE,
        current_user=current_user,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        app.logger.info("Received request to /predict endpoint")
        filepath = None
        cloudinary_url = None
        public_id = None

        # Handle sample image selection (JSON request)
        if request.is_json:
            data = request.get_json()
            sample_path = data.get("sample_path")
            if sample_path:
                # Security check: ensure path is within static/samples
                if not sample_path.startswith("/static/samples/"):
                    return jsonify({"error": "Invalid sample path"}), 400

                # Convert URL path to local path
                # Use os.path.join with app.root_path and strip leading slash
                relative_path = sample_path.lstrip("/")
                filepath = os.path.join(app.root_path, relative_path)

                if not os.path.exists(filepath):
                    app.logger.error(f"Sample file not found at: {filepath}")
                    return jsonify({"error": "Sample file not found"}), 404

                app.logger.info(f"Using sample image: {filepath}")

                # IMPORTANT: For sample images, we also need to upload to Cloudinary
                # for the CLIP validation (which is URL-based)
                try:
                    upload_result = cloudinary.uploader.upload(
                        filepath, folder="wheat_disease"
                    )
                    cloudinary_url = upload_result.get("secure_url")
                    public_id = upload_result.get("public_id")
                except Exception as ce:
                    app.logger.error(f"Cloudinary upload failed for sample: {str(ce)}")
            else:
                return jsonify({"error": "No sample path provided"}), 400

        # Handle traditional file upload
        else:
            if "file" not in request.files:
                app.logger.error("No file part in the request")
                return jsonify({"error": "No file part"}), 400

            file = request.files["file"]
            if file.filename == "":
                app.logger.error("No file selected")
                return jsonify({"error": "No selected file"}), 400

            # Check file extension
            allowed_extensions = {"png", "jpg", "jpeg"}
            if (
                "." not in file.filename
                or file.filename.rsplit(".", 1)[1].lower() not in allowed_extensions
            ):
                app.logger.error(f"Invalid file type: {file.filename}")
                return (
                    jsonify(
                        {
                            "error": "Invalid file type. Please upload a PNG, JPG, or JPEG image."
                        }
                    ),
                    400,
                )

            # Ensure upload directory exists
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            app.logger.info(f"File saved to {filepath}")

            # CLIP Validation (Hugging Face Space)
            app.logger.info(
                f"Sending image URL for CLIP validation to {CLIP_VERIFY_URL}"
            )
            try:
                # 1. Start Cloudinary Upload FIRST (Needed because validation is now URL-based)
                # We upload even if it's junk, but we'll purge it in a few lines if validation fails.
                upload_result = cloudinary.uploader.upload(
                    filepath, folder="wheat_disease"
                )
                cloudinary_url = upload_result.get("secure_url")
                public_id = upload_result.get("public_id")
            except Exception as ce:
                app.logger.error(f"Cloudinary upload failed for upload: {str(ce)}")

            try:
                if cloudinary_url:
                    # 2. Call Hugging Face Space CLIP Validation
                    # No trailing slash for HF Spaces to avoid 405/500 redirect loops
                    val_url = CLIP_VERIFY_URL.rstrip("/")
                    val_response = requests.post(
                        val_url, json={"image_url": cloudinary_url}, timeout=45
                    )

                    if val_response.status_code == 200:
                        val_data = val_response.json()
                        # Updated to match your final HF Space code: {"is_valid": True/False, "wheat_score": ...}
                        is_valid = val_data.get("is_valid", True)
                        wheat_score = val_data.get("wheat_score", 0.0)

                        if not is_valid:
                            app.logger.warning(
                                f"CLIP validation failed (Score: {wheat_score}). Purging image."
                            )
                            # Purge from Cloudinary immediately
                            if public_id:
                                cloudinary.uploader.destroy(public_id)
                            if "/static/samples/" not in filepath and os.path.exists(
                                filepath
                            ):
                                os.remove(filepath)
                            return jsonify(
                                {
                                    "error": f"Image validation failed: This doesn't look like a wheat crop (Wheat Confidence: {wheat_score * 100:.2f}%)",
                                    "success": False,
                                }
                            ), 400
                    else:
                        app.logger.error(
                            f"HF Space returned error {val_response.status_code}"
                        )
                else:
                    app.logger.warning(
                        "Skipping CLIP validation as Cloudinary URL is missing"
                    )

            except Exception as e:
                app.logger.error(f"Error during CLIP validation: {str(e)}")
                # Continue if validation is down to avoid blocking user

        # Open and verify image (ResNet50 processing)
        try:
            if not filepath:
                return jsonify({"error": "File path not established"}), 400

            image = Image.open(filepath).convert("RGB")
            # For saved files we might want to verify, but for samples we know they are valid
            if "/static/samples/" not in filepath:
                image.verify()
                # After verify() we MUST reopen the file because verify() consumes the stream
                image = Image.open(filepath).convert("RGB")
        except Exception as e:
            app.logger.error(f"Invalid image file: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Invalid image file"}), 400

        # Preprocess and predict
        try:
            # Start ResNet50 Inference
            input_data = preprocess_image(image)
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            ort_outs = ort_session.run(None, ort_inputs)
            outputs = ort_outs[0]

            # ... (rest of processing using the cloudinary_url we already created)
            # Apply Softmax to get probabilities (confidence scores)
            if len(outputs.shape) == 1:
                exp_outputs = np.exp(outputs - np.max(outputs))
                probabilities = exp_outputs / np.sum(exp_outputs)
            else:
                exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
                probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

            predicted_class = np.argmax(probabilities)
            predicted_label = CLASS_NAMES.get(int(predicted_class), "Unknown")
            confidence_score = float(np.max(probabilities)) * 100

            # Wait for Cloudinary upload result (already done above, but we reuse the vars)
            cloudinary_error = None
            if not cloudinary_url:
                cloudinary_error = "Upload failed during validation step"

            # Save Initial Feedback/Log Entry to Aiven (ClickHouse)
            new_feedback = Feedback(
                image_url=cloudinary_url
                if cloudinary_url
                else os.path.basename(filepath),
                predicted_class=predicted_label,
                confidence=float(confidence_score),
                is_correct=True,  # Default until user feedback
            )
            db.session.add(new_feedback)
            db.session.commit()

            # Get weather data for current user if available
            weather_data = get_current_user_weather()

            # Apply infection highlighting
            highlighted_url = None
            if predicted_label != "Healthy":
                highlighted_filename = f"highlighted_{os.path.basename(filepath)}"
                highlighted_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], highlighted_filename
                )
                if highlight_infection(filepath, predicted_label, highlighted_path):
                    highlighted_url = f"/uploads/{highlighted_filename}"
                    app.logger.info(f"Highlighted image saved to {highlighted_path}")

            # Prepare response data
            image_url = f"/uploads/{os.path.basename(filepath)}"
            if "/static/samples/" in filepath:
                image_url = f"/static/samples/{os.path.basename(filepath)}"

            response_data = {
                "success": True,
                "label": predicted_label,
                "confidence": f"{confidence_score:.2f}%",
                "image_url": image_url,
                "highlighted_url": highlighted_url,
                "cloudinary_url": cloudinary_url,
                "cloudinary_error": cloudinary_error,
                "feedback_id": new_feedback.id,
                "weather_data": weather_data,
                "show_questionnaire": current_user.is_authenticated,
                "redirect_url": url_for("result"),
            }

            # Store result in session
            session["analysis_result"] = {
                "label": predicted_label,
                "confidence": f"{confidence_score:.2f}%",
                "image_path": image_url,
                "highlighted_path": highlighted_url,
                "cloudinary_url": cloudinary_url,
                "cloudinary_error": cloudinary_error,
                "feedback_id": new_feedback.id,
                "weather_data": weather_data,
            }

            # Prepare separate response for AJAX if needed
            response_data = {
                "success": True,
                "label": predicted_label,
                "confidence": f"{confidence_score:.2f}%",
                "image_url": image_url,
                "highlighted_url": highlighted_url,
                "cloudinary_url": cloudinary_url,
                "cloudinary_error": cloudinary_error,
                "feedback_id": new_feedback.id,
                "weather_data": weather_data,
                "show_questionnaire": current_user.is_authenticated,
                "redirect_url": url_for("result"),
            }

            app.logger.info(
                f"Prediction successful: {predicted_label} with {confidence_score:.2f}% confidence"
            )
            return jsonify(response_data)

        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return jsonify({"error": "Error processing image"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in predict route: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route("/bulk-predict", methods=["POST"])
def bulk_predict():
    """Endpoint for parallel image uploads. Returns a batch_id and image_id immediately."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        batch_id = request.form.get("batch_id")
        image_id = request.form.get("image_id")

        if not batch_id or not image_id:
            return jsonify({"error": "Missing batch_id or image_id"}), 400

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save file
        filename = secure_filename(f"{batch_id}_{image_id}_{file.filename}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Capture user_id from context
        user_id = current_user.id if current_user.is_authenticated else None

        def process_upload_and_queue(fp, bid, iid, uid):
            try:
                upload_result = cloudinary.uploader.upload(fp, folder="wheat_disease")
                c_url = upload_result.get("secure_url")

                # Add to sequential processing queue
                processing_queue.put(
                    {
                        "batch_id": bid,
                        "image_id": iid,
                        "filepath": fp,
                        "cloudinary_url": c_url,
                        "user_id": uid,
                    }
                )
            except Exception as e:
                app.logger.error(f"Async upload failed for {iid}: {str(e)}")
                if bid not in batch_results:
                    batch_results[bid] = {}
                batch_results[bid][iid] = {
                    "status": "failed",
                    "error": "Upload failed",
                    "image_id": iid,
                }

        executor.submit(process_upload_and_queue, filepath, batch_id, image_id, user_id)

        return jsonify({"status": "queued", "batch_id": batch_id, "image_id": image_id})

    except Exception as e:
        app.logger.error(f"Bulk predict error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/results-stream/<batch_id>")
def results_stream(batch_id):
    """SSE endpoint to stream results back to the user as they finish."""

    def event_stream():
        sent_images = set()
        start_time = time.time()
        timeout = 300  # 5 minutes

        while time.time() - start_time < timeout:
            if batch_id in batch_results:
                current_results = batch_results[batch_id]
                for img_id, data in list(current_results.items()):
                    if img_id not in sent_images:
                        yield f"data: {json.dumps(data)}\n\n"
                        sent_images.add(img_id)

                # Check if we should terminate (caller can signal end or we can check queue)
                # For simplicity, we just keep polling until timeout or client disconnects

            time.sleep(0.5)
            # Send keep-alive
            yield ": keep-alive\n\n"

    return make_response(
        event_stream(),
        {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
        },
    )


@app.route("/batch-recommendation/<batch_id>")
def batch_recommendation(batch_id):
    """Generates a cohesive AI recommendation for the entire batch of images."""
    if batch_id not in batch_results:
        return "No batch data found.", 404

    # Only consider successfully processed wheat images
    results = [
        r for r in batch_results[batch_id].values() if r.get("status") == "completed"
    ]
    if not results:
        return "No valid wheat images to analyze.", 400

    # Summarize findings
    disease_counts = {}
    for r in results:
        label = r.get("label")
        if not label or label == "Unknown":
            continue
        disease_counts[label] = disease_counts.get(label, 0) + 1

    total = sum(disease_counts.values())
    if total == 0:
        return "No confident disease predictions available.", 400
    summary_str = ", ".join(
        [f"{count} {label}" for label, count in disease_counts.items()]
    )

    # Simple logic for now, could be upgraded to use LLM in openai_integration.py
    main_disease = max(disease_counts, key=disease_counts.get)

    # Get weather data for contextual intelligence
    weather = get_current_user_weather()
    weather_desc = (
        weather.get("condition", {}).get("text", "Normal")
        if isinstance(weather, dict)
        else "Normal"
    )

    prompt_context = {
        "weather": {"condition": weather_desc},
        "crop_condition": f"Mixed findings across {total} images: {summary_str}",
        "disease_detected": main_disease,
        "is_batch": True,
    }

    # Try to get detailed LLM recommendation if API key exists
    try:
        from openai_integration import get_openai_recommendation

        html_rec = get_openai_recommendation(prompt_context)
        if isinstance(html_rec, str):
            return html_rec
    except:
        pass

    # Fallback to simple HTML recommendation
    if main_disease == "Healthy":
        return f"<p class='text-green-700 font-semibold'>Field status looks great! {disease_counts.get('Healthy', 0)} of {total} images were healthy. Maintain current practices.</p>"
    else:
        return f"<p class='text-red-700 font-semibold'>Warning: Significant presence of {main_disease} detected ({disease_counts.get(main_disease, 0)}/{total} images). We recommend immediate inspection and targeted treatment for infected areas.</p>"


@app.route("/update-location", methods=["POST"])
@login_required
def update_location():
    try:
        data = request.json
        loc_type = data.get("type", "manual")

        location_query = None
        lat = None
        lon = None
        manual_location = None

        if loc_type == "automatic":
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is not None and lon is not None:
                location_query = f"{lat},{lon}"
            else:
                return jsonify({"success": False, "error": "Coordinates missing"}), 400
        else:
            manual_location = data.get("location")
            if not manual_location:
                return jsonify({"success": False, "error": "Location is required"}), 400
            location_query = manual_location

        # Get weather data for the location
        weather_data = get_weather_data(location=location_query)

        # Update user's location and weather data
        current_user.location_type = loc_type
        current_user.manual_location = manual_location
        current_user.latitude = lat
        current_user.longitude = lon
        current_user.weather_data = weather_data
        user_db.save_users()

        return jsonify(
            {
                "success": True,
                "message": "Location updated successfully",
                "weather_data": weather_data,
                "city": weather_data.get("location") if weather_data else None,
            }
        )
    except Exception as e:
        app.logger.error(f"Error updating location: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Failed to update location. Please try again.",
                }
            ),
            500,
        )


@app.route("/submit-questionnaire", methods=["POST"])
@login_required
def submit_questionnaire():
    try:
        # Get form data
        form_data = {}
        for item in QUESTIONNAIRE:
            field_id = item["id"]
            if item["type"] == "checkbox":
                form_data[field_id] = request.form.getlist(field_id)
            else:
                form_data[field_id] = request.form.get(field_id, "").strip()

        # Store responses in user profile
        current_user.questionnaire_responses = form_data
        user_db.save_users()

        # Return success response with confirmation message
        return jsonify(
            {
                "success": True,
                "message": "Thank you for sharing your crop details! We will use this information to provide better recommendations when you analyze your wheat plants.",
            }
        )

    except Exception as e:
        app.logger.error(f"Error processing questionnaire: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "An error occurred while saving your responses. Please try again.",
                }
            ),
            500,
        )


# Error Handlers


@app.errorhandler(404)
def not_found_error(error):
    return (
        jsonify(
            {"status": "error", "message": "The requested resource was not found."}
        ),
        404,
    )


@app.errorhandler(400)
def bad_request_error(error):
    return (
        jsonify(
            {
                "status": "error",
                "message": "Bad request. Please check your input and try again.",
            }
        ),
        400,
    )


@app.errorhandler(500)
def internal_error(error):
    return (
        jsonify(
            {
                "status": "error",
                "message": "An internal server error occurred. Please try again later.",
            }
        ),
        500,
    )


# User Answers Endpoints


@app.route("/get-user-answers")
@login_required
def get_user_answers():
    try:
        if not current_user.questionnaire_responses:
            return (
                jsonify({"success": False, "error": "No answers found for this user."}),
                404,
            )

        return jsonify(
            {"success": True, "answers": current_user.questionnaire_responses}
        )
    except Exception as e:
        app.logger.error(f"Error getting user answers: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "An error occurred while retrieving your answers.",
                }
            ),
            500,
        )


@app.route("/update-answers", methods=["POST"])
@login_required
def update_answers():
    try:
        # Get form data
        updated_responses = {}
        for key, value in request.form.items():
            # Handle multi-select checkboxes
            if key.endswith("[]"):
                base_key = key[:-2]  # Remove '[]' from the key
                if base_key not in updated_responses:
                    updated_responses[base_key] = []
                updated_responses[base_key].append(value)
            else:
                updated_responses[key] = value

        # Update user's responses
        current_user.questionnaire_responses.update(updated_responses)
        user_db.save_users()

        return jsonify({"success": True, "message": "Answers updated successfully."})
    except Exception as e:
        app.logger.error(f"Error updating answers: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "An error occurred while updating your answers.",
                }
            ),
            500,
        )


@app.route("/get-recommendations", methods=["POST"])
@login_required
def get_recommendations():
    try:
        data = request.get_json()

        # Get the most up-to-date weather for the user
        weather_data = get_current_user_weather() or data.get("weather_data", {})
        if isinstance(weather_data, str):
            try:
                weather_data = json.loads(weather_data)
            except json.JSONDecodeError:
                weather_data = {}

        # Prepare location data
        # Prioritize saved user location info
        city = "Unknown"
        region = "Unknown"
        country = "IN"
        loc = "0,0"

        if current_user.weather_data and "location" in current_user.weather_data:
            # WeatherAPI returns "City, Country" in location string
            loc_parts = current_user.weather_data["location"].split(",")
            city = loc_parts[0].strip()
            if len(loc_parts) > 1:
                country = loc_parts[-1].strip()

        if current_user.latitude and current_user.longitude:
            loc = f"{current_user.latitude},{current_user.longitude}"
        elif current_user.manual_location:
            loc = current_user.manual_location

        # Prepare user data with proper structure for OpenAI API
        user_data = {
            "user_id": current_user.id,
            "location": {
                "city": city,
                "region": region,
                "country": country,
                "loc": loc,
                "timezone": "Asia/Kolkata",
            },
            "weather": {
                "temp_c": weather_data.get("temperature", 25.0),
                "humidity": weather_data.get("humidity", 60),
                "condition": weather_data.get("conditions", "Clear"),
                "wind_kph": weather_data.get("wind_speed", 10.0),
                "precip_mm": weather_data.get("precipitation", 0.0),
                "last_updated": weather_data.get("last_updated", "N/A"),
            },
            "questionnaire_answers": getattr(
                current_user, "questionnaire_responses", {}
            )
            or {},
            "crop_condition": (
                "Healthy" if data.get("disease") == "Healthy" else "Affected"
            ),
            "disease_detected": data.get("disease", "None"),
        }

        # Store user_data in session for export later
        session["last_user_data"] = user_data

        # Get recommendations from OpenAI
        try:
            from openai_integration import get_openai_recommendation

            result = get_openai_recommendation(user_data)

            if result["status"] == "success":
                # Store recommendation in session for export
                session["last_recommendation"] = result["recommendation"]
                session["last_image_path"] = data.get("image_path")

                return jsonify(
                    {"status": "success", "recommendation": result["recommendation"]}
                )
            else:
                app.logger.error(
                    f"OpenAI API error: {result.get('message', 'Unknown error')}"
                )
                status_code = 503 if "503" in result.get("message", "") else 400
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Recommendation service currently overloaded. Please try again in a few moments.",
                        }
                    ),
                    status_code,
                )

        except ImportError as ie:
            app.logger.error(f"Failed to import OpenAI module: {str(ie)}")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Recommendation service is currently unavailable.",
                    }
                ),
                500,
            )

    except Exception as e:
        app.logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        return (
            jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}),
            500,
        )


@app.route("/export-report")
@login_required
def export_report():
    user_data = session.get("last_user_data")
    recommendation = session.get("last_recommendation")
    image_path = session.get("last_image_path")

    if not user_data or not recommendation:
        flash("No report data available to export.", "warning")
        return redirect(url_for("index"))

    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        "MainTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.indigo,
        spaceAfter=20,
        alignment=1,
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=18,
        textColor=colors.darkblue,
        spaceBefore=15,
        spaceAfter=10,
    )

    story = []

    # Title
    story.append(Paragraph("Wheat Disease Analysis Report", title_style))
    story.append(Spacer(1, 12))

    # Summary Info Table
    summary_data = [
        ["User", current_user.username],
        ["Disease Detected", user_data.get("disease_detected", "Unknown")],
        [
            "Location",
            f"{user_data.get('location', {}).get('city')}, {user_data.get('location', {}).get('country')}",
        ],
        [
            "Weather",
            f"{user_data.get('weather', {}).get('condition')}, {user_data.get('weather', {}).get('temp_c')}°C",
        ],
    ]

    t = Table(summary_data, colWidths=[150, 300])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 20))

    # Add Image if available
    if image_path:
        try:
            # Handle both local and URL paths
            if image_path.startswith("static/"):
                full_image_path = os.path.join(app.root_path, image_path)
            else:
                full_image_path = os.path.join(
                    app.root_path, "static", "uploads", os.path.basename(image_path)
                )

            if os.path.exists(full_image_path):
                img = RLImage(full_image_path, width=400, height=300)
                story.append(img)
                story.append(Spacer(1, 20))
        except Exception as e:
            print(f"Error adding image to PDF: {e}")

    # Recommendation Content
    story.append(Paragraph("Expert Recommendations", section_style))
    story.append(Spacer(1, 10))

    # Improved HTML cleaning for PDF
    # Convert div sections to bold headers and spacing for PDF
    processed_rec = recommendation
    # Replace headers
    processed_rec = re.sub(
        r"<h3[^>]*>(.*?)</h3>",
        r'<br/><font color="darkblue" size="14"><b>\1</b></font><br/>',
        processed_rec,
    )
    # Replace list items
    processed_rec = processed_rec.replace("<li>", "&bull; ").replace("</li>", "<br/>")
    processed_rec = processed_rec.replace("<ul>", "").replace("</ul>", "")
    # Replace bold
    processed_rec = processed_rec.replace("<strong>", "<b>").replace(
        "</strong>", "</b>"
    )
    # Remove div tags but keep content
    processed_rec = re.sub(r"<div[^>]*>", "", processed_rec)
    processed_rec = processed_rec.replace("</div>", "<br/>")

    # Strip any remaining tags that ReportLab Paragraph doesn't support
    # (ReportLab only supports a small subset like <b>, <i>, <u>, <font>, <br/>, <a>)
    clean_rec = re.sub(
        r"<(?!b|/b|i|/i|u|/u|font|/font|br|/br|a|/a)[^>]+>", "", processed_rec
    )

    # Normalize line breaks
    clean_rec = clean_rec.replace("\n", " ").strip()

    try:
        story.append(Paragraph(clean_rec, styles["BodyText"]))
    except Exception as e:
        app.logger.error(f"Error rendering PDF paragraph: {e}")
        # Final fallback: strip EVERYTHING
        final_fallback = re.sub("<[^<]+?>", "", recommendation)
        story.append(Paragraph(final_fallback, styles["BodyText"]))

    # Build PDF
    doc.build(story)
    buffer.seek(0)

    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = (
        f"attachment; filename=Wheat_Report_{user_data.get('disease_detected')}.pdf"
    )

    return response


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Submit user feedback for prediction accuracy and save to PostgreSQL."""
    try:
        data = request.json
        feedback_id = data.get("feedback_id")
        is_correct = data.get("is_correct")
        correct_class = data.get("correct_class")

        # Update specific feedback record in PostgreSQL
        feedback = Feedback.query.get(feedback_id)
        if feedback:
            feedback.is_correct = is_correct
            if not is_correct and correct_class:
                feedback.correct_class = correct_class
            db.session.commit()

            # Set a session flag so the UI remembers feedback was given on reload
            session[f"feedback_submitted_{feedback_id}"] = True

            return jsonify(
                {"success": True, "message": "PostgreSQL feedback updated successfully"}
            )

        return jsonify({"success": False, "error": "Feedback record not found"}), 404
    except Exception as e:
        app.logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Create ASGI application
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    # Create admin user if not exists
    if not user_db.get_user_by_username("admin"):
        admin = user_db.add_user("admin", "admin123", "admin@example.com")

    # For development, you can still run with Flask's development server
    if os.environ.get("ENV") == "development":
        print("Starting Dev Server: http://127.0.0.1:10000")
        app.run(debug=True, host="0.0.0.0", port=10000, use_reloader=False)
    else:
        print("Starting Production Server: http://127.0.0.1:10000")
        import uvicorn

        uvicorn.run(asgi_app, host="0.0.0.0", port=10000)
