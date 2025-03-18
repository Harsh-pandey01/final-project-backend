from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load the trained model
model = joblib.load('crop_model.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allows frontend to connect to backend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from frontend

        # Extract input values
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        # Convert input into the right format
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Make prediction
        prediction = model.predict(input_data)  # ✅ This will now work!

        # Return predicted crop
        return jsonify({'crop': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
