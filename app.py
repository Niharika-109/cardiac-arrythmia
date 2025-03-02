from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("cardiac_arrhythmia_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    features = [float(request.form["feature1"]), 
                float(request.form["feature2"]), 
                float(request.form["feature3"]), 
                float(request.form["feature4"])]
    
    # Convert to numpy array and reshape for model
    features_array = np.array(features).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(features_array)
    
    # Convert to binary output (0 or 1)
    prediction = 1 if prediction[0][0] > 0.5 else 0
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
