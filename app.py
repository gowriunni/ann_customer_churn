
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained ANN and preprocessing pipeline
model = tf.keras.models.load_model("churn_ann.keras")
preprocessor = joblib.load("preprocessor.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Read customer details from the form
            customer = {
                "gender": request.form["gender"],
                "SeniorCitizen": int(request.form["SeniorCitizen"]),
                "Partner": request.form["Partner"],
                "Dependents": request.form["Dependents"],
                "tenure": int(request.form["tenure"]),
                "PhoneService": request.form["PhoneService"],
                "MultipleLines": request.form["MultipleLines"],
                "InternetService": request.form["InternetService"],
                "OnlineSecurity": request.form["OnlineSecurity"],
                "OnlineBackup": request.form["OnlineBackup"],
                "DeviceProtection": request.form["DeviceProtection"],
                "TechSupport": request.form["TechSupport"],
                "StreamingTV": request.form["StreamingTV"],
                "StreamingMovies": request.form["StreamingMovies"],
                "Contract": request.form["Contract"],
                "PaperlessBilling": request.form["PaperlessBilling"],
                "PaymentMethod": request.form["PaymentMethod"],
                "MonthlyCharges": float(request.form["MonthlyCharges"]),
                "TotalCharges": float(request.form["TotalCharges"])
            }

            # Use the same column order as the original dataset
            columns = [
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "tenure", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling",
                "PaymentMethod", "MonthlyCharges", "TotalCharges"
            ]

            input_df = pd.DataFrame([customer], columns=columns)

            # Apply the already-fitted preprocessing pipeline
            processed = preprocessor.transform(input_df)

            if hasattr(processed, "toarray"):
                processed = processed.toarray()

            processed = np.asarray(processed, dtype=np.float32)

            # Predict probability of churn
            probability = float(
                model.predict(processed, verbose=0)[0][0] * 100
            )

            prediction = (
                "Likely to Churn"
                if probability >= 50
                else "Less Likely to Churn"
            )

        except (ValueError, KeyError) as exc:
            error = (
                "Please enter valid customer details. "
                "Total Charges must be numeric."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained ANN and preprocessing pipeline
model = tf.keras.models.load_model("churn_ann.keras")
preprocessor = joblib.load("preprocessor.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Read customer details from the form
            customer = {
                "gender": request.form["gender"],
                "SeniorCitizen": int(request.form["SeniorCitizen"]),
                "Partner": request.form["Partner"],
                "Dependents": request.form["Dependents"],
                "tenure": int(request.form["tenure"]),
                "PhoneService": request.form["PhoneService"],
                "MultipleLines": request.form["MultipleLines"],
                "InternetService": request.form["InternetService"],
                "OnlineSecurity": request.form["OnlineSecurity"],
                "OnlineBackup": request.form["OnlineBackup"],
                "DeviceProtection": request.form["DeviceProtection"],
                "TechSupport": request.form["TechSupport"],
                "StreamingTV": request.form["StreamingTV"],
                "StreamingMovies": request.form["StreamingMovies"],
                "Contract": request.form["Contract"],
                "PaperlessBilling": request.form["PaperlessBilling"],
                "PaymentMethod": request.form["PaymentMethod"],
                "MonthlyCharges": float(request.form["MonthlyCharges"]),
                "TotalCharges": float(request.form["TotalCharges"])
            }

            # Use the same column order as the original dataset
            columns = [
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "tenure", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling",
                "PaymentMethod", "MonthlyCharges", "TotalCharges"
            ]

            input_df = pd.DataFrame([customer], columns=columns)

            # Apply the already-fitted preprocessing pipeline
            processed = preprocessor.transform(input_df)

            if hasattr(processed, "toarray"):
                processed = processed.toarray()

            processed = np.asarray(processed, dtype=np.float32)

            # Predict probability of churn
            probability = float(
                model.predict(processed, verbose=0)[0][0] * 100
            )

            prediction = (
                "Likely to Churn"
                if probability >= 50
                else "Less Likely to Churn"
            )

        except (ValueError, KeyError) as exc:
            error = (
                "Please enter valid customer details. "
                "Total Charges must be numeric."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=False)

