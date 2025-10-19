"""Tests for the location service."""

import pytest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from location import get_ip_geolocation, reverse_geocode, update_user_location


def test_get_ip_geolocation_success():
    """Test successful IP geolocation lookup."""
    mock_response = {
        'status': 'success',
        'lat': 37.7749,
        'lon': -122.4194,
        'city': 'San Francisco',
        'regionName': 'California',
        'country': 'United States',
        'query': '8.8.8.8'
    }
    
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status.return_value = None
        
        result = get_ip_geolocation('8.8.8.8')
        
        assert result['lat'] == 37.7749
        assert result['lon'] == -122.4194
        assert 'San Francisco' in result['address']
        assert result['source'] == 'ip'
        assert result['ip'] == '8.8.8.8'


def test_get_ip_geolocation_failure():
    """Test failed IP geolocation lookup."""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("API Error")
        result = get_ip_geolocation('8.8.8.8')
        assert result is None


def test_reverse_geocode_success():
    """Test successful reverse geocoding."""
    mock_response = {
        'address': {
            'city': 'San Francisco',
            'state': 'California',
            'country': 'United States'
        }
    }
    
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status.return_value = None
        
        result = reverse_geocode(37.7749, -122.4194)
        
        assert 'San Francisco' in result['formatted_address']
        assert 'California' in result['formatted_address']
        assert result['details'] == mock_response['address']


def test_update_user_location():
    """Test updating user location in the database."""
    from models import User, user_db
    
    # Create a test user
    test_user = User(
        id='1',
        username='testuser',
        password_hash='testhash',
        location_consent=True
    )
    
    # Mock the user database
    with patch('models.user_db') as mock_db:
        mock_db.update_user.return_value = test_user
        
        location_data = {
            'lat': 37.7749,
            'lon': -122.4194,
            'accuracy': 1.0,
            'address': 'San Francisco, CA'
        }
        
        result = update_user_location(test_user, location_data, 'browser')
        
        assert result == test_user
        mock_db.update_user.assert_called_once()
        
        # Check that the update included the expected fields
        update_args = mock_db.update_user.call_args[1]
        assert update_args['last_location_lat'] == 37.7749
        assert update_args['last_location_lon'] == -122.4194
        assert update_args['location_accuracy_km'] == 1.0
        assert update_args['last_location_address'] == 'San Francisco, CA'
        assert update_args['location_source'] == 'browser'
        assert 'location_updated_at' in update_args


def test_update_location_endpoint(client, auth):
    """Test the /api/location/update endpoint."""
    # Login first
    auth.login()
    
    # Mock the reverse_geocode and update_user_location functions
    with patch('location.reverse_geocode') as mock_reverse_geocode, \
         patch('location.update_user_location') as mock_update_location:
        
        # Setup mocks
        mock_reverse_geocode.return_value = {
            'formatted_address': 'San Francisco, CA',
            'details': {}
        }
        
        mock_user = MagicMock()
        mock_user.last_location_lat = 37.7749
        mock_user.last_location_lon = -122.4194
        mock_user.last_location_address = 'San Francisco, CA'
        mock_user.location_accuracy_km = 0.05
        mock_user.location_source = 'browser'
        mock_user.location_updated_at = datetime.utcnow().isoformat()
        
        mock_update_location.return_value = mock_user
        
        # Make the request
        response = client.post('/api/location/update', json={
            'lat': 37.7749,
            'lon': -122.4194,
            'accuracy': 50
        })
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['location']['lat'] == 37.7749
        assert data['location']['lon'] == -122.4194
        assert 'San Francisco' in data['location']['address']


def test_update_location_consent_endpoint(client, auth):
    """Test the /api/location/consent endpoint."""
    # Login first
    auth.login()
    
    # Mock the user database
    with patch('models.user_db') as mock_db:
        mock_user = MagicMock()
        mock_user.location_consent = True
        mock_db.update_user.return_value = mock_user
        
        # Make the request
        response = client.post('/api/location/consent', json={
            'consent': True
        })
        
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['consent'] is True
        mock_db.update_user.assert_called_once()


if __name__ == '__main__':
    pytest.main()
