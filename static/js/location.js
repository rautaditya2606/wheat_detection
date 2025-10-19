/**
 * Location handling for Wheat Disease Detection
 * Handles browser geolocation and fallback to IP-based geolocation
 */

class LocationManager {
    constructor() {
        this.options = {
            enableHighAccuracy: true,
            timeout: 10000,  // 10 seconds
            maximumAge: 5 * 60 * 1000  // 5 minutes
        };
        
        // Bind methods
        this.requestLocation = this.requestLocation.bind(this);
        this._handleGeolocationSuccess = this._handleGeolocationSuccess.bind(this);
        this._handleGeolocationError = this._handleGeolocationError.bind(this);
        this._updateLocationConsent = this._updateLocationConsent.bind(this);
        
        // Initialize
        this._setupEventListeners();
    }
    
    /**
     * Set up event listeners for location-related UI elements
     */
    _setupEventListeners() {
        // Location permission toggle
        const locationConsentToggle = document.getElementById('location-consent-toggle');
        if (locationConsentToggle) {
            locationConsentToggle.addEventListener('change', (e) => {
                this._updateLocationConsent(e.target.checked);
                if (e.target.checked) {
                    this.requestLocation();
                }
            });
        }
        
        // Manual location refresh button
        const refreshLocationBtn = document.getElementById('refresh-location');
        if (refreshLocationBtn) {
            refreshLocationBtn.addEventListener('click', () => this.requestLocation());
        }
    }
    
    /**
     * Request the user's current location
     * @returns {Promise} Resolves with location data or rejects with error
     */
    requestLocation() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                console.warn('Geolocation is not supported by this browser');
                this._fallbackToIPLocation().then(resolve).catch(reject);
                return;
            }
            
            // Show loading state
            this._updateLocationUI('loading');
            
            navigator.geolocation.getCurrentPosition(
                (position) => this._handleGeolocationSuccess(position, resolve, reject),
                (error) => this._handleGeolocationError(error, resolve, reject),
                this.options
            );
        });
    }
    
    /**
     * Handle successful geolocation
     * @private
     */
    _handleGeolocationSuccess(position, resolve, reject) {
        const { latitude, longitude, accuracy } = position.coords;
        const locationData = {
            lat: latitude,
            lon: longitude,
            accuracy: accuracy / 1000  // Convert to km
        };
        
        // Send to server
        this._sendLocationToServer(locationData, 'browser')
            .then(data => {
                this._updateLocationUI('success', data);
                resolve(data);
            })
            .catch(error => {
                console.error('Error sending location to server:', error);
                this._updateLocationUI('error', { error: 'Failed to save location' });
                reject(error);
            });
    }
    
    /**
     * Handle geolocation errors
     * @private
     */
    _handleGeolocationError(error, resolve, reject) {
        console.warn('Geolocation error:', error);
        
        // Fall back to IP-based location if permission denied or position unavailable
        if (error.code === error.PERMISSION_DENIED || error.code === error.POSITION_UNAVAILABLE) {
            this._fallbackToIPLocation().then(resolve).catch(reject);
        } else {
            this._updateLocationUI('error', { error: 'Unable to retrieve your location' });
            reject(error);
        }
    }
    
    /**
     * Fall back to IP-based geolocation
     * @private
     */
    async _fallbackToIPLocation() {
        this._updateLocationUI('loading', { message: 'Using approximate location based on IP' });
        
        try {
            const response = await fetch('/api/location/ip', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to get IP-based location');
            }
            
            const data = await response.json();
            this._updateLocationUI('success', { ...data, approximate: true });
            return data;
        } catch (error) {
            console.error('IP-based location failed:', error);
            this._updateLocationUI('error', { error: 'Failed to determine your location' });
            throw error;
        }
    }
    
    /**
     * Send location data to the server
     * @private
     */
    async _sendLocationToServer(locationData, source) {
        const response = await fetch('/api/location/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': this._getCSRFToken()
            },
            body: JSON.stringify({
                ...locationData,
                source: source
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to update location');
        }
        
        return response.json();
    }
    
    /**
     * Update location sharing consent
     * @private
     */
    async _updateLocationConsent(consent) {
        try {
            const response = await fetch('/api/location/consent', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': this._getCSRFToken()
                },
                body: JSON.stringify({ consent })
            });
            
            if (!response.ok) {
                throw new Error('Failed to update consent');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error updating location consent:', error);
            throw error;
        }
    }
    
    /**
     * Update the UI based on location status
     * @private
     */
    _updateLocationUI(status, data = {}) {
        const locationStatus = document.getElementById('location-status');
        const locationInfo = document.getElementById('location-info');
        const locationError = document.getElementById('location-error');
        
        if (!locationStatus) return;
        
        // Reset all states
        locationStatus.className = 'location-status';
        locationStatus.innerHTML = '';
        
        if (locationInfo) locationInfo.textContent = '';
        if (locationError) locationError.textContent = '';
        
        switch (status) {
            case 'loading':
                locationStatus.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Detecting your location...';
                locationStatus.classList.add('loading');
                break;
                
            case 'success':
                locationStatus.innerHTML = '<i class="fa fa-check-circle"></i> Location detected';
                locationStatus.classList.add('success');
                
                if (locationInfo) {
                    const { address, accuracy_km, approximate } = data;
                    let infoText = address || 'Location updated';
                    
                    if (approximate) {
                        infoText += ' (approximate)';
                    } else if (accuracy_km) {
                        infoText += ` (accuracy: ${Math.round(accuracy_km * 10) / 10} km)`;
                    }
                    
                    locationInfo.textContent = infoText;
                }
                break;
                
            case 'error':
                locationStatus.innerHTML = '<i class="fa fa-exclamation-triangle"></i> Location error';
                locationStatus.classList.add('error');
                
                if (locationError) {
                    locationError.textContent = data.error || 'Unable to determine your location';
                }
                break;
                
            default:
                break;
        }
    }
    
    /**
     * Get CSRF token from cookies
     * @private
     */
    _getCSRFToken() {
        const cookieValue = document.cookie
            .split('; ')
            .find(row => row.startsWith('csrf_token='))
            ?.split('=')[1];
            
        return cookieValue || '';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if we have location-related elements on the page
    if (document.querySelector('.location-status') || document.getElementById('location-consent-toggle')) {
        window.locationManager = new LocationManager();
        
        // Request location if consent was previously given
        const consentToggle = document.getElementById('location-consent-toggle');
        if (consentToggle && consentToggle.checked) {
            window.locationManager.requestLocation();
        }
    }
});
