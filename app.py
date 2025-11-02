from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response, session, send_from_directory
from asgiref.wsgi import WsgiToAsgi
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
# CSRF protection removed for development
import torch
from torchvision import models, transforms
from PIL import Image
import os
from collections import OrderedDict
from dotenv import load_dotenv
import json
from models import user_db, User
from user_data import user_data, QUESTIONNAIRE
from utils import get_weather_data, get_llm_recommendation
from location import location_bp, get_ip_geolocation, reverse_geocode

app = Flask(__name__)
load_dotenv()

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hour CSRF token expiration

# Register blueprints
app.register_blueprint(location_bp, url_prefix='')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return user_db.get_user_by_id(user_id)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure upload and data directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

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

# Load model with updated weights parameter
model = models.resnet50(weights=None)  # Using weights=None instead of pretrained=False
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

# Authentication Routes

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = user_db.get_user_by_username(username)
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if user_db.get_user_by_username(username):
            flash('Username already exists', 'danger')
        else:
            user_db.add_user(username, password, email)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Main Routes

@app.route('/')
def index():
    return render_template('index.html', current_user=current_user)

@app.route('/result')
def result():
    label = request.args.get('label', 'Unknown')
    image_path = request.args.get('image_path', '')
    weather_data = request.args.get('weather_data', '{}')
    
    try:
        weather_data = json.loads(weather_data)
    except (json.JSONDecodeError, TypeError):
        weather_data = {}
    
    return render_template('result.html',
                         label=label,
                         image_path=image_path,
                         weather_data=weather_data,
                         current_user=current_user)

@app.route('/questionnaire')
@login_required
def questionnaire():
    # Get weather data for the questionnaire
    weather_data = get_weather_data()
    
    return render_template('questionnaire.html',
                         weather_data=weather_data,
                         questionnaire=QUESTIONNAIRE,
                         current_user=current_user)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Received request to /predict endpoint")
        
        if 'file' not in request.files:
            app.logger.error("No file part in the request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error("No file selected")
            return jsonify({'error': 'No selected file'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            app.logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400
        
        try:
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.info(f"File saved to {filepath}")

            # Open and verify image
            try:
                image = Image.open(filepath).convert("RGB")
                # Verify it's a valid image by trying to load it
                image.verify()
                image = Image.open(filepath).convert("RGB")
            except Exception as e:
                app.logger.error(f"Invalid image file: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'Invalid image file'}), 400

            # Preprocess and predict
            try:
                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    predicted_class = torch.argmax(outputs, 1).item()
                    predicted_label = CLASS_NAMES.get(predicted_class, 'Unknown')
                
                # Get weather data
                weather_data = get_weather_data()
                
                # Prepare response data
                response_data = {
                    'success': True,
                    'label': predicted_label,
                    'image_url': f'/uploads/{os.path.basename(filepath)}',
                    'weather_data': weather_data,
                    'show_questionnaire': current_user.is_authenticated,
                    'redirect_url': url_for('result', 
                                         label=predicted_label, 
                                         image_path=f'/uploads/{os.path.basename(filepath)}',
                                         weather_data=json.dumps(weather_data),
                                         _external=True)
                }
                
                app.logger.info(f"Prediction successful: {predicted_label}")
                return jsonify(response_data)
                
            except Exception as e:
                app.logger.error(f"Error during prediction: {str(e)}", exc_info=True)
                return jsonify({'error': 'Error processing image'}), 500
            
        except Exception as e:
            app.logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return jsonify({'error': 'Error processing file'}), 500
            
    except Exception as e:
        app.logger.error(f"Unexpected error in predict route: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/update-location', methods=['POST'])
@login_required
def update_location():
    try:
        location = request.json.get('location')
        if not location:
            return jsonify({'success': False, 'error': 'Location is required'}), 400
        
        # Get weather data for the location
        weather_data = get_weather_data(location=location)
        
        # Update user's location and weather data
        current_user.manual_location = location
        current_user.weather_data = weather_data
        user_db.save_users()
        
        return jsonify({
            'success': True,
            'message': 'Location updated successfully',
            'weather_data': weather_data
        })
    except Exception as e:
        app.logger.error(f"Error updating location: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to update location. Please try again.'
        }), 500

@app.route('/submit-questionnaire', methods=['POST'])
@login_required
def submit_questionnaire():
    try:
        # Get form data
        form_data = {}
        for item in QUESTIONNAIRE:
            field_id = item['id']
            if item['type'] == 'checkbox':
                form_data[field_id] = request.form.getlist(field_id)
            else:
                form_data[field_id] = request.form.get(field_id, '').strip()
        
        # Store responses in user profile
        current_user.questionnaire_responses = form_data
        user_db.save_users()
        
        # Return success response with confirmation message
        return jsonify({
            'success': True,
            'message': 'Thank you for sharing your crop details! We will use this information to provide better recommendations when you analyze your wheat plants.'
        })
    
    except Exception as e:
        app.logger.error(f"Error processing questionnaire: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while saving your responses. Please try again.'
        }), 500

# Error Handlers

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'The requested resource was not found.'
    }), 404

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Bad request. Please check your input and try again.'
    }), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'An internal server error occurred. Please try again later.'
    }), 500

# User Answers Endpoints

@app.route('/get-user-answers')
@login_required
def get_user_answers():
    try:
        if not current_user.questionnaire_responses:
            return jsonify({
                'success': False,
                'error': 'No answers found for this user.'
            }), 404
            
        return jsonify({
            'success': True,
            'answers': current_user.questionnaire_responses
        })
    except Exception as e:
        app.logger.error(f"Error getting user answers: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while retrieving your answers.'
        }), 500

@app.route('/update-answers', methods=['POST'])
@login_required
def update_answers():
    try:
        # Get form data
        updated_responses = {}
        for key, value in request.form.items():
            # Handle multi-select checkboxes
            if key.endswith('[]'):
                base_key = key[:-2]  # Remove '[]' from the key
                if base_key not in updated_responses:
                    updated_responses[base_key] = []
                updated_responses[base_key].append(value)
            else:
                updated_responses[key] = value
        
        # Update user's responses
        current_user.questionnaire_responses.update(updated_responses)
        user_db.save_users()
        
        return jsonify({
            'success': True,
            'message': 'Answers updated successfully.'
        })
    except Exception as e:
        app.logger.error(f"Error updating answers: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while updating your answers.'
        }), 500


@app.route('/get-recommendations', methods=['POST'])
@login_required
def get_recommendations():
    try:
        data = request.get_json()
        
        # Get weather data with proper structure
        weather_data = data.get('weather_data', {})
        if isinstance(weather_data, str):
            try:
                weather_data = json.loads(weather_data)
            except json.JSONDecodeError:
                weather_data = {}
        
        # Prepare user data with proper structure for Gemini API
        user_data = {
            'user_id': current_user.id,
            'location': {
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country', 'IN'),
                'loc': data.get('loc', '0,0'),
                'timezone': data.get('timezone', 'Asia/Kolkata')
            },
            'weather': {
                'temp_c': weather_data.get('temperature', weather_data.get('temp_c', 25.0)),
                'humidity': weather_data.get('humidity', 60),
                'condition': weather_data.get('conditions', weather_data.get('condition', 'Clear')),  # Handle both 'condition' and 'conditions'
                'wind_kph': weather_data.get('wind_speed', weather_data.get('wind_kph', 10.0)),
                'precip_mm': weather_data.get('precipitation', weather_data.get('precip_mm', 0.0)),
                'last_updated': weather_data.get('last_updated', '2023-10-20 12:00')
            },
            'questionnaire_answers': getattr(current_user, 'questionnaire_responses', {}) or {},
            'crop_condition': 'Healthy' if data.get('disease') == 'Healthy' else 'Affected',
            'disease_detected': data.get('disease', 'None')
        }
        
        # Get recommendations from Gemini
        try:
            from test_gemini_integration import get_gemini_recommendation
            result = get_gemini_recommendation(user_data)
            
            if result['status'] == 'success':
                return jsonify({
                    'status': 'success',
                    'recommendation': result['recommendation']
                })
            else:
                app.logger.error(f"Gemini API error: {result.get('message', 'Unknown error')}")
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to generate recommendations. Please try again.'
                }), 400
                
        except ImportError as ie:
            app.logger.error(f"Failed to import Gemini module: {str(ie)}")
            return jsonify({
                'status': 'error',
                'message': 'Recommendation service is currently unavailable.'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Create ASGI application
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # Create admin user if not exists
    if not user_db.get_user_by_username('admin'):
        admin = user_db.add_user('admin', 'admin123', 'admin@example.com')
    
    # For development, you can still run with Flask's development server
    if os.environ.get('ENV') == 'development':
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        import uvicorn
        uvicorn.run(asgi_app, host='0.0.0.0', port=5000)