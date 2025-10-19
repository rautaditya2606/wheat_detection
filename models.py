from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
from datetime import datetime

class User(UserMixin):    
    def __init__(self, id, username, password_hash, email=None, 
                 manual_location=None, weather_data=None,
                 questionnaire_responses=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.manual_location = manual_location
        self.weather_data = weather_data or {}
        self.questionnaire_responses = questionnaire_responses or {}
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
        
    def update_questionnaire_responses(self, responses):
        """Update user's questionnaire responses."""
        self.questionnaire_responses = responses
        user_db.save_users()  # Save changes to database

class UserDB:
    def __init__(self, filename='users.json'):
        self.filename = filename
        self.users = {}
        self.load_users()
        
    def update_user(self, user_id, **kwargs):
        """Update user attributes."""
        user = self.users.get(str(user_id))
        if not user:
            return None
            
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
                
        self.save_users()
        return user
    
    def load_users(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.users = {}
                    for uid, user_data in data.items():
                        # Ensure all required fields are present
                        user_data.setdefault('email', None)
                        user_data.setdefault('manual_location', None)
                        user_data.setdefault('weather_data', {})
                        user_data.setdefault('questionnaire_responses', {})
                        
                        # Clean up old fields that no longer exist in the User class
                        for field in ['location_consent', 'last_location_lat', 'last_location_lon', 
                                    'last_location_address', 'location_updated_at', 
                                    'location_accuracy_km', 'location_source']:
                            user_data.pop(field, None)
                            
                        self.users[uid] = User(
                            id=user_data['id'],
                            username=user_data['username'],
                            password_hash=user_data['password_hash'],
                            email=user_data['email'],
                            manual_location=user_data.get('manual_location'),
                            weather_data=user_data.get('weather_data', {}),
                            questionnaire_responses=user_data.get('questionnaire_responses', {})
                        )
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading users: {e}")
                self.users = {}
    
    def save_users(self):
        with open(self.filename, 'w') as f:
            users_data = {}
            for uid, user in self.users.items():
                users_data[uid] = {
                    'id': user.id,
                    'username': user.username,
                    'password_hash': user.password_hash,
                    'email': user.email,
                    'manual_location': user.manual_location,
                    'weather_data': user.weather_data,
                    'questionnaire_responses': user.questionnaire_responses
                }
            json.dump(users_data, f, indent=2)
    
    def add_user(self, username, password, email=None, **kwargs):
        user_id = str(len(self.users) + 1)
        user = User(
            id=user_id,
            username=username,
            password_hash=generate_password_hash(password),
            email=email,
            manual_location=kwargs.get('manual_location'),
            weather_data=kwargs.get('weather_data', {}),
            questionnaire_responses=kwargs.get('questionnaire_responses', {})
        )
        self.users[user_id] = user
        self.save_users()
        return user
    
    def get_user_by_username(self, username):
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_id(self, user_id):
        return self.users.get(str(user_id))

# Initialize user database
user_db = UserDB()
