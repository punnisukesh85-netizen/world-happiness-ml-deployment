from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------- LOAD TRAINED OBJECTS ----------------
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
rfe = joblib.load("rfe.joblib")
original_features = joblib.load("original_features.pkl")
# ------------------------------------------------------

@app.route("/")
def home():
    # Send original feature names to HTML
    return render_template("index.html", features=original_features)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Collect user input (ALL original features)
        input_data = {}
        for feature in original_features:
            input_data[feature] = float(request.form[feature])

        # 2️⃣ Convert to DataFrame (VERY IMPORTANT)
        input_df = pd.DataFrame([input_data])

        # 3️⃣ Apply SAME preprocessing as training
        input_scaled = scaler.transform(input_df)
        input_rfe = rfe.transform(input_scaled)

        # 4️⃣ Predict
        prediction = model.predict(input_rfe)[0]

        return render_template(
            "index.html",
            features=original_features,
            prediction_text=f"Predicted Value: {prediction:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            features=original_features,
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
