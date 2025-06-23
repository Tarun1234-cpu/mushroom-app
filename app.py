from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and label encoders
model = joblib.load('mushroom_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')

features = [
    'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
    'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
    'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
    'stalk_surface_below_ring', 'stalk_color_above_ring',
    'stalk_color_below_ring', 'veil_type', 'veil_color',
    'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat'
]

HTML = """
<!doctype html>
<title>Mushroom Classification</title>
<h2>Enter Mushroom Features</h2>
<form method="POST">
  {% for feature in features %}
    <label>{{ feature.replace('_', ' ').title() }}:</label><br>
    <input type="text" name="{{ feature }}" required><br><br>
  {% endfor %}
  <input type="submit" value="Predict">
</form>

{% if prediction %}
  <h3>Prediction: <b>{{ prediction }}</b></h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def classify():
    prediction = None
    if request.method == "POST":
        input_data = {}
        for feature in features:
            input_data[feature] = request.form.get(feature).strip()

        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # Encode input features
        for col, le in label_encoders.items():
            if col != 'class' and col in df.columns:
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # If input value not recognized, send error message
                    return render_template_string(HTML, features=features, prediction=f"Invalid input for '{col}': {df[col].values[0]}")

        # Predict
        pred_encoded = model.predict(df)[0]

        # Decode prediction
        pred_label = label_encoders['class'].inverse_transform([pred_encoded])[0]
        prediction = "Edible" if pred_label == 'e' else "Poisonous"

    return render_template_string(HTML, features=features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
