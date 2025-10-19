import json
import os
from datetime import datetime

class UserData:
    def __init__(self, filename='user_responses.json'):
        self.filename = filename
        self.responses = {}
        self.load_responses()
    
    def load_responses(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.responses = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.responses = {}
    
    def save_responses(self):
        with open(self.filename, 'w') as f:
            json.dump(self.responses, f, indent=2)
    
    def save_response(self, user_id, response_data):
        if user_id not in self.responses:
            self.responses[user_id] = []
        
        # Add timestamp to the response
        response_data['timestamp'] = datetime.now().isoformat()
        self.responses[user_id].append(response_data)
        self.save_responses()
    
    def get_user_responses(self, user_id):
        return self.responses.get(str(user_id), [])

# Initialize user data handler
user_data = UserData()

# Questionnaire template
QUESTIONNAIRE = [
    {
        'id': 'soil_type',
        'question': 'What type of soil is your wheat crop growing in?',
        'type': 'select',
        'options': [
            'Sandy', 'Clay', 'Loamy', 'Silty', 'Peaty', 'Chalky', 'Loamy Sand', 'Sandy Loam', 'Silt Loam', 'Silty Clay', 'Clay Loam', 'Sandy Clay', 'Silty Clay Loam', 'Sandy Clay Loam', 'Other'
        ],
        'required': False
    },
    {
        'id': 'fertilizer_used',
        'question': 'What type of fertilizer or manure have you used?',
        'type': 'select',
        'options': [
            'None', 'Urea', 'DAP', 'NPK', 'Compost', 'Farmyard Manure', 'Vermicompost', 'Other'
        ],
        'required': False
    },
    {
        'id': 'pesticide_used',
        'question': 'Have you used any pesticides or fungicides?',
        'type': 'select',
        'options': [
            'None', 'Insecticides', 'Fungicides', 'Herbicides', 'Combination', 'Other'
        ],
        'required': False
    },
    {
        'id': 'crop_rotation',
        'question': 'Do you practice crop rotation?',
        'type': 'radio',
        'options': ['Yes', 'No', 'Sometimes'],
        'required': False
    },
    {
        'id': 'irrigation_method',
        'question': 'What irrigation method do you use?',
        'type': 'select',
        'options': [
            'Flood', 'Drip', 'Sprinkler', 'Rain-fed', 'Other'
        ],
        'required': False
    },
    {
        'id': 'previous_crop',
        'question': 'What was the previous crop grown in this field?',
        'type': 'text',
        'required': False
    },
    {
        'id': 'planting_date',
        'question': 'When was the wheat planted? (YYYY-MM-DD)',
        'type': 'date',
        'required': False
    },
    {
        'id': 'additional_notes',
        'question': 'Any additional notes about your crop or the disease symptoms?',
        'type': 'textarea',
        'required': False
    }
]
