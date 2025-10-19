# Location and Disease Detection Integration

This document explains the location-based features in the Wheat Disease Classifier application, including how location data is collected, stored, and used for disease prediction and recommendations.

## Table of Contents

1. [Overview](#overview)
2. [Location Data Collection](#location-data-collection)
3. [Privacy and Consent](#privacy-and-consent)
4. [Integration with Disease Prediction](#integration-with-disease-prediction)
5. [Setup and Configuration](#setup-and-configuration)
6. [API Endpoints](#api-endpoints)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Overview

The application uses location data to provide more accurate disease predictions and recommendations. Location data can be collected through:

1. **Browser Geolocation API** (high accuracy, requires user permission)
2. **IP-based Geolocation** (fallback method, less accurate)

## Location Data Collection

### Browser Geolocation

- Uses the HTML5 Geolocation API (`navigator.geolocation`)
- Requires explicit user permission
- Provides high accuracy (typically within a few meters)
- Includes accuracy estimation in meters
- Works only on secure contexts (HTTPS or localhost)

### IP-based Geolocation

- Used as a fallback when:
  - User denies location permission
  - Browser doesn't support geolocation
  - Geolocation times out
- Less accurate (city/region level)
- Uses [ipinfo.io](https://ipinfo.io/) service (requires API key)

## Privacy and Consent

- Location data is collected **only** with explicit user consent
- Users can enable/disable location sharing at any time
- Location data is stored securely in the user's profile
- Users can request deletion of their location data
- IP addresses are not stored long-term
- All location data is transmitted over HTTPS

## Integration with Disease Prediction

When a user submits a crop image for analysis:

1. The system checks for the user's location (if consent was given)
2. If available, location data is included in the analysis request
3. The ML model considers location-specific factors:
   - Common diseases in the region
   - Local weather conditions
   - Growing season and crop stage
4. Recommendations are tailored based on the user's location

## Setup and Configuration

### Required Environment Variables

```bash
# IPInfo.io API Key (for IP-based geolocation)
IPINFO_TOKEN=your-ipinfo-token

# Path to GeoLite2 City database (optional, for local GeoIP lookups)
GEOIP2_CITY_DB=GeoLite2-City.mmdb
```

### Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the GeoLite2 City database (optional):
   - Sign up at [MaxMind](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data)
   - Download the GeoLite2 City database
   - Place the `.mmdb` file in the project directory
   - Update the `GEOIP2_CITY_DB` environment variable with the path to the file

## API Endpoints

### Update Location

- **URL**: `/api/location/update`
- **Method**: `POST`
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "lat": 37.7749,
    "lon": -122.4194,
    "accuracy": 50.5
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "location": {
      "lat": 37.7749,
      "lon": -122.4194,
      "address": "San Francisco, CA, USA",
      "accuracy_km": 0.05,
      "source": "browser",
      "updated_at": "2023-11-01T12:00:00Z"
    }
  }
  ```

### Update Location Consent

- **URL**: `/api/location/consent`
- **Method**: `POST`
- **Authentication**: Required
- **Request Body**:
  ```json
  {
    "consent": true
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "consent": true
  }
  ```

## Security Considerations

1. **HTTPS**: Always use HTTPS in production to protect location data in transit
2. **Data Minimization**: Only collect the minimum location data needed
3. **User Control**: Allow users to view and delete their location data
4. **Rate Limiting**: Implement rate limiting on location update endpoints
5. **Logging**: Be careful not to log sensitive location data
6. **CORS**: Configure CORS headers appropriately

## Troubleshooting

### Location permission denied

- Ensure the site is served over HTTPS (or localhost for development)
- Check browser settings to ensure location access is allowed
- The user may have denied the permission request

### Inaccurate location

- For browser geolocation, ensure the device has GPS/WiFi enabled
- IP-based geolocation is less accurate (city/region level)
- Check the accuracy value in the response

### API errors

- Verify the IPInfo API key is set and valid
- Check the server logs for detailed error messages
- Ensure the GeoLite2 database is up to date (if used)

## Deployment

### Production Checklist

- [ ] Enable HTTPS
- [ ] Set appropriate CORS headers
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Test with location services disabled
- [ ] Update privacy policy to include location data handling

### Monitoring

Monitor the following metrics:

- Geolocation success/failure rates
- Average accuracy of location data
- API response times
- Error rates for location-related endpoints

## Future Enhancements

1. Support for offline location caching
2. More granular location-based recommendations
3. Integration with weather APIs for real-time conditions
4. Support for multiple locations per user
5. Anonymized location analytics for disease tracking
