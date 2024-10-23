from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)


model_path = r'models/crop_recommendation/random_forest_model.pkl'
scaler_path = r'models/crop_recommendation/scaler.pkl'
label_encoder_path = r'models/crop_recommendation/label_encoder.pkl'


with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.get("/")
async def ping():
    return "Hello, I am alive"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    features = np.array([data], dtype=float)
    scaled_data = scaler.transform(features)
    prediction_encoded = model.predict(scaled_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

