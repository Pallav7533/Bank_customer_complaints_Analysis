from flask import Flask, request, jsonify
import joblib
import os

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model with error handling
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise Exception(f"Model not found at path: {model_path}")

# Health check endpoint to ensure the app is running
@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='ok'), 200

# Endpoint to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the incoming POST request
        data = request.get_json()

        # Extract the 'narrative' field from the JSON data
        narrative = data.get('narrative', '')

        # Check if the 'narrative' field is missing or empty
        if not narrative:
            return jsonify(error="The 'narrative' field is required for prediction."), 400

        # Make a prediction using the loaded model
        prediction = model.predict([narrative])

        # Class mapping from index to product type
        class_mapping = {
            0: 'credit_card',
            1: 'credit_reporting',
            2: 'debt_collection',
            3: 'mortgages_and_loans',
            4: 'retail_banking'
        }

        # Determine the predicted class
        predicted_class = class_mapping.get(prediction[0], "Unknown")

        # Return the predicted class as a JSON response
        return jsonify(predicted_class=predicted_class), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify(error=f"An error occurred during prediction: {str(e)}"), 500

# Run the Flask app if executed as a standalone program
if __name__ == '__main__':
    # Set debug to True for development purposes; change to False for production
    app.run(debug=True, host='0.0.0.0', port=5000)  # Modify port if needed
