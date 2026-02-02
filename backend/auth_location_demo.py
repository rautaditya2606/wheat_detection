from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# In-memory storage for demonstration purposes
# In a real app, this would be a database
user_sessions = {}
user_locations = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auth & Location Access Demo</title>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #f0f2f5; }
        .card { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 300px; text-align: center; }
        input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #status { margin-top: 15px; font-size: 0.9rem; color: #666; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Demo App</h1>
        
        {% if not username %}
            <div id="auth-section">
                <h3>Login or Sign Up</h3>
                <input type="text" id="username-input" placeholder="Enter Username">
                <button onclick="handleLogin()">Login / Sign Up</button>
            </div>
        {% else %}
            <div id="welcome-section">
                <h3>Welcome, {{ username }}!</h3>
                <p id="status">Checking for location access...</p>
                <div id="location-info"></div>
                <button onclick="handleLogout()" style="background: #dc3545; margin-top: 20px;">Logout</button>
            </div>
        {% endif %}
    </div>

    <script>
        async function handleLogin() {
            const username = document.getElementById('username-input').value;
            if (!username) return alert('Please enter a username');
            
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username })
            });

            if (response.ok) {
                window.location.reload();
            }
        }

        async function handleLogout() {
            await fetch('/logout', { method: 'POST' });
            window.location.reload();
        }

        {% if username %}
        function requestLocation() {
            const status = document.getElementById('status');
            const info = document.getElementById('location-info');

            if (!navigator.geolocation) {
                status.innerText = "Geolocation is not supported by your browser.";
                return;
            }

            status.innerText = "Requesting location permission...";

            navigator.geolocation.getCurrentPosition(
                async (position) => {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    
                    status.innerText = "Location permission granted. Saving coordinates...";
                    
                    const response = await fetch('/save_location', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lat, lon })
                    });
                    
                    const result = await response.json();
                    if (response.ok) {
                        status.innerText = "Location saved successfully!";
                        info.innerHTML = `<p>Latitude: ${lat}<br>Longitude: ${lon}</p>`;
                    } else {
                        status.innerText = "Failed to save location: " + result.message;
                    }
                },
                (error) => {
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            status.innerText = "User denied the request for Geolocation.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            status.innerText = "Location information is unavailable.";
                            break;
                        case error.TIMEOUT:
                            status.innerText = "The request to get user location timed out.";
                            break;
                        default:
                            status.innerText = "An unknown error occurred.";
                            break;
                    }
                }
            );
        }

        // Automatically trigger location request after login/loading
        window.onload = requestLocation;
        {% endif %}
    </script>
</body>
</html>
"""

# Simple session management (Demo only)
current_session_user = None

@app.route("/")
def index():
    global current_session_user
    return render_template_string(HTML_TEMPLATE, username=current_session_user)

@app.route("/login", methods=["POST"])
def login():
    global current_session_user
    data = request.json
    username = data.get("username")
    if username:
        current_session_user = username
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid username"}), 400

@app.route("/logout", methods=["POST"])
def logout():
    global current_session_user
    current_session_user = None
    return jsonify({"status": "success"})

@app.route("/save_location", methods=["POST"])
def save_location():
    global current_session_user
    if not current_session_user:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")
    
    # Store the location
    user_locations[current_session_user] = {
        "latitude": lat,
        "longitude": lon,
        "timestamp": request.date if hasattr(request, 'date') else None
    }
    
    print(f"Saved location for {current_session_user}: Lat {lat}, Lon {lon}")
    return jsonify({"status": "success", "data": user_locations[current_session_user]})

if __name__ == "__main__":
    print("Demo server starting at http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
