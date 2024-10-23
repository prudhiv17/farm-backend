from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the model and label encoder
with open('C:/Users/Admin/Documents/Deploy_Git/models/disease/decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('C:/Users/Admin/Documents/Deploy_Git/models/disease/label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Ensure label_encoder is a LabelEncoder
if not isinstance(label_encoder, LabelEncoder):
    raise ValueError("label_encoder is not an instance of LabelEncoder")

# Mapping from encoded label to disease name
disease_map = {
    0: "Brownplanthopper",
    1: "Caseworm",
    2: "Gallmidge",
    3: "Greenleafhoper",
    4: "LeafFolder",
    5: "Miridbug",
    6: "Whitebackedplanthopper",
    7: "Yellowstemborer - LT",
    8: "Yellowstemborer - PT"
}

# Disease prevention information
disease_prevention = {
    "Brownplanthopper": {
        "crops_affected": "Rice",
        "prevention": [
            "Use resistant rice varieties.",
            "Apply insecticides when necessary.",
            "Maintain field sanitation.",
            "Control alternative hosts."
        ]
    },
    "Caseworm": {
        "crops_affected": "Rice, Corn",
        "prevention": [
            "Regularly monitor fields.",
            "Use biological control agents like Trichogramma.",
            "Apply appropriate insecticides if thresholds are exceeded."
        ]
    },
    "Gallmidge": {
        "crops_affected": "Rice",
        "prevention": [
            "Implement water management strategies to reduce breeding sites.",
            "Use resistant varieties.",
            "Maintain regular field inspections."
        ]
    },
    "Miridbug": {
        "crops_affected": "Cotton",
        "prevention": [
            "Use resistant cotton varieties.",
            "Apply insecticides during early infestations.",
            "Avoid excessive nitrogen fertilizer."
        ]
    },
    "Greenleafhoper": {
        "crops_affected": "Rice",
        "prevention": [
            "Monitor crops regularly for early detection.",
            "Use insecticidal sprays if infestation levels are high.",
            "Remove weeds that may host the pests."
        ]
    },
    "Whitebackedplanthopper": {
        "crops_affected": "Rice",
        "prevention": [
            "Use resistant rice varieties.",
            "Apply insecticides if necessary.",
            "Practice crop rotation and remove alternative hosts."
        ]
    },
    "LeafFolder": {
        "crops_affected": "Rice",
        "prevention": [
            "Use resistant rice varieties.",
            "Apply insecticides if pest levels are high.",
            "Maintain proper field management practices."
        ]
    },
    "Yellowstemborer - LT": {
        "crops_affected": "Rice",
        "prevention": [
            "Use resistant varieties.",
            "Apply insecticides at the appropriate growth stages.",
            "Practice good field sanitation."
        ]
    },
    "Yellowstemborer - PT": {
        "crops_affected": "Rice",
        "prevention": [
            "Use resistant varieties.",
            "Apply insecticides as soon as symptoms are noticed.",
            "Practice field sanitation and proper water management."
        ]
    }
}

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Extract data from request
        data = request.json
        features = [
            data['temperature'],
            data['minTemperature'],
            data['humidity'],
            data['humidity2'],
            data['rainfall'],
            data['windSpeed'],
            data['sunshineHours'],
            data['evaporation']
        ]
        features = np.array([features])

        # Make a prediction
        if hasattr(model, 'predict'):
            prediction_encoded = model.predict(features)
        else:
            raise ValueError("Model object does not have 'predict' method")

        # Decode the prediction
        prediction_number = prediction_encoded[0]
        prediction_name = disease_map.get(prediction_number, "Unknown disease")

        # Get disease prevention information
        prevention_info = disease_prevention.get(prediction_name, {
            "crops_affected": "Unknown",
            "prevention": ["Information not available."]
        })

        return jsonify({
            'prediction': prediction_name,
            'crops_affected': prevention_info['crops_affected'],
            'prevention': prevention_info['prevention']
        })
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
