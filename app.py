import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("./ufo-model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    int_features = [int(x) for x in request.form.values()]
    
    # Create DataFrame with feature names (to avoid warning)
    feature_names = ['Seconds', 'Latitude', 'Longitude']
    input_df = pd.DataFrame([int_features], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Get the predicted country
    output = prediction[0]
    countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
    return render_template(
        "index.html", 
        prediction_text="Likely country: {}".format(countries[output])
    )

if __name__ == "__main__":
    app.run(debug=True)