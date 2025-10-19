"""Location service for handling geolocation and reverse geocoding."""

import requests
from flask import Blueprint, request, jsonify, current_app as flask_current_app
from flask_login import current_user, login_required
from datetime import datetime

# Import user_db here to avoid circular imports
from models import user_db

# Initialize Blueprint
location_bp = Blueprint('location', __name__)


def get_ip_geolocation(ip_address=None):
    """Get geolocation data using IP address using free ip-api.com service."""
    try:
        # Use ip-api.com for IP-based geolocation (free tier)
        fields = 'status,message,country,regionName,city,lat,lon,query'
        base_url = "http://ip-api.com/json/"
        ip_param = f"{ip_address or ''}"
        url = f"{base_url}{ip_param}?fields={fields}"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success':
            city = data.get('city', '')
            region = data.get('regionName', '')
            country = data.get('country', '')
            address = ", ".join(filter(None, [city, region, country]))
            return {
                'lat': data['lat'],
                'lon': data['lon'],
                'accuracy': 50000,  # Default accuracy for IP-based location (50km)
                'address': address,
                'source': 'ip',
                'ip': data.get('query', ip_address)
            }
    except requests.exceptions.RequestException as e:
        flask_current_app.logger.error("IP geolocation error: %s", str(e))
    
    return None


def reverse_geocode(lat, lon):
    """Convert coordinates to human-readable address using OpenStreetMap Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'accept-language': 'en',
            'zoom': 10  # Get more detailed address
        }
        
        headers = {
            'User-Agent': 'WheatDiseaseDetection/1.0 (your@email.com)'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Extract address components
        address = data.get('address', {})
        address_parts = [
            address.get('village'),
            address.get('town'),
            address.get('city'),
            address.get('county'),
            address.get('state'),
            address.get('country')
        ]
        
        # Join non-empty parts with comma
        formatted_address = ', '.join(filter(None, address_parts))
        
        return {
            'formatted_address': formatted_address,
            'details': address
        }
    except requests.exceptions.RequestException as e:
        flask_current_app.logger.error("Reverse geocoding error: %s", str(e))
        return None


def update_user_location(user, location_data, source):
    """Update user's location in the database."""
    if not user or not user.is_authenticated:
        flask_current_app.logger.error(
            "Cannot update location: User not authenticated"
        )
        return None
        
    updates = {
        'last_location_lat': location_data.get('lat'),
        'last_location_lon': location_data.get('lon'),
        'location_accuracy_km': location_data.get('accuracy'),
        'location_source': source,
        'location_updated_at': datetime.utcnow().isoformat()
    }
    
    # Only update address if we have it
    if 'address' in location_data:
        updates['last_location_address'] = location_data['address']
    
    return user_db.update_user(user.id, **updates)


@location_bp.route('/api/location/update', methods=['POST'])
@login_required
def update_location():
    """Update user's location from browser geolocation."""
    data = request.get_json()
    
    # Validate required fields
    if not data or 'lat' not in data or 'lon' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Latitude and longitude are required'
        }), 400
    
    try:
        lat = float(data['lat'])
        lon = float(data['lon'])
        accuracy = float(data.get('accuracy', 0)) / 1000  # Convert meters to km
        
        # Get address from reverse geocoding
        address_data = reverse_geocode(lat, lon)
        address = address_data['formatted_address'] if address_data else None
        
        # Prepare location data
        location_data = {
            'lat': lat,
            'lon': lon,
            'accuracy': accuracy,
            'address': address,
            'source': 'browser'
        }
        
        # Update user's location
        user = update_user_location(
            current_user._get_current_object(), 
            location_data, 
            'browser'
        )
        
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update location: User not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'location': {
                'lat': user.last_location_lat,
                'lon': user.last_location_lon,
                'address': user.last_location_address,
                'accuracy_km': user.location_accuracy_km,
                'source': user.location_source,
                'updated_at': user.location_updated_at
            }
        })
        
    except (ValueError, TypeError) as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid location data',
            'error': str(e)
        }), 400
    except Exception as e:
        flask_current_app.logger.error("Location update error: %s", str(e), exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Failed to update location',
            'error': str(e)
        }), 500


@location_bp.route('/api/location/consent', methods=['POST'])
@login_required
def update_location_consent():
    """Update user's location sharing consent."""
    data = request.get_json()
    
    if 'consent' not in data or not isinstance(data['consent'], bool):
        return jsonify({
            'status': 'error',
            'message': 'Consent status is required and must be a boolean'
        }), 400
    
    try:
        # Update consent status
        user = user_db.update_user(
            current_user.id,
            location_consent=data['consent']
        )
        
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'consent': user.location_consent
        })
        
    except Exception as e:
        flask_current_app.logger.error("Consent update error: %s", str(e), exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Failed to update consent status',
            'error': str(e)
        }), 500
