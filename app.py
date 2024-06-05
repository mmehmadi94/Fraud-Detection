from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    # Extract features and reshape for the model
    features = np.array([data['year'], data['month'], data['day'],
                         data['hour'], data['minute'], data['age'],
                         data['amt'], data['gender'], data['category'],
                         data['merch_lat'], data['merch_long']
                        ]).reshape(1, -1)
    # Scale the features
    features_scaled = scaler.transform(features)
    # Predict using the model
    prediction = model.predict(features_scaled)
    # Return the result as a JSON response
    return jsonify({'is_fraud': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, port=8088)  # Specify the port number here
