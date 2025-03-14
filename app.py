from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import joblib
import io

# Load the trained model and encoders
model = joblib.load("model.pkl")
le_region = joblib.load("le_region.pkl")
le_specialty = joblib.load("le_specialty.pkl")

# Load dataset
df = pd.read_excel("dummy_npi_data.xlsx", sheet_name="Dataset")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get hour input from the user
        hour = int(request.args.get("hour"))

        # Prepare dataset
        df["Login Hour"] = pd.to_datetime(df["Login Time"]).dt.hour
        df["Region"] = le_region.transform(df["Region"])
        df["Speciality"] = le_specialty.transform(df["Speciality"])

        # Extract features
        X = df[["Login Hour", "Usage Time (mins)", "Region", "Speciality", "Count of Survey Attempts"]]

        # Make Predictions
        df["Prediction"] = model.predict(X)

        # Filter best doctors
        best_doctors = df[(df["Prediction"] == 1) & (df["Login Hour"] == hour)][["NPI", "Region", "Speciality", "Count of Survey Attempts"]]

        # If no doctors found, return a message
        if best_doctors.empty:
            return jsonify({"message": "No doctors found for this time."})

        # Convert DataFrame to JSON for the frontend table
        results_json = best_doctors.to_dict(orient="records")

        # Save CSV for downloading
        output = io.BytesIO()
        best_doctors.to_csv(output, index=False)
        output.seek(0)

        return jsonify({"data": results_json, "csv_url": "/download_csv"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/download_csv')
def download_csv():
    df_filtered = df[df["Prediction"] == 1][["NPI", "Region", "Speciality", "Count of Survey Attempts"]]
    output = io.BytesIO()
    df_filtered.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="predicted_doctors.csv")

# if __name__ == '__main__':
#     app.run(debug=True)
