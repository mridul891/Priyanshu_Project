import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load models
try:
    all_models = joblib.load('models/models.pkl')
    if not isinstance(all_models, list):  # Check if models are a list of model objects
        raise ValueError("The loaded models are not in a list format.")
except Exception as e:
    print(f"Error loading models: {e}")
    all_models = []

@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template("index.html")

@app.route('/aboutUs', methods=['GET'])
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/api', methods=['POST'])
def predict():
    # Get form data
    name = request.form['name']
    email = request.form['email']
    age = request.form['age']
    fgender = request.form['gender']
    cp = request.form['cp']
    trestbps = request.form.get('trestbps', 95)  # Default if not provided
    chol = request.form.get('chol', 150)  # Default if not provided
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form.get('thalach', 72)  # Default if not provided
    exang = request.form['exang']
    oldpeak = request.form.get('oldpeak', 2)  # Default if not provided
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    # Prepare display data for the result page
    input_data = {
        "age": age,
        "Gender": fgender,
        "Chest Pain Types": cp,
        "Resting Blood Pressure(in mm/Hg)": trestbps,
        "Cholesterol Level": chol,
        "is Fasting Blood Sugar > 120mg/Dl?": fbs,
        "Resting Electro Cardio Graphic Result": restecg,
        "Maximum Heart Rate Achieved": thalach,
        "Does Exercise Induced Angina?": exang,
        "Old Peak (ST Depression Induced by Exercise Relative to Rest)": oldpeak,
        "Slope of ST Segment": slope,
        "Number of Major Vessels (0-3) Colored by Fluoroscopy": ca,
        "Thal Type": thal
    }

    # Encode categorical values
    gender = 1 if fgender == "Male" else 0

    thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
    thal = thal_map.get(thal, 0)

    restecg_map = {"Normal": 0, "STT Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    restecg = restecg_map.get(restecg, 0)

    exang = 1 if exang == "Yes" else 0

    # Convert all to appropriate types for model prediction
    features = [
        int(age), gender, int(cp), int(trestbps), int(chol), int(fbs),
        int(restecg), int(thalach), int(exang), float(oldpeak),
        int(slope), int(ca), int(thal)
    ]

    prediction_summary = {}
    avg = 0

    # Make predictions using all models
    for model in all_models:
        if hasattr(model, 'predict'):  # Ensure the model has a predict method
            res = model.predict([features])
            prediction_summary[type(model).__name__] = "High Chance of Heart Disease" if res[0] == 1 else "Low Chance of Heart Disease"
            avg += res[0]
        else:
            prediction_summary[type(model).__name__] = "Invalid model"

    # Calculate accuracy (percentage of models predicting 'High Chance')
    accuracy = round(avg / len(all_models), 2) if len(all_models) > 0 else 0
    personal_info = [name, email]

    responses = [input_data, prediction_summary, personal_info, accuracy]

    return render_template("result.html", result=responses)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
