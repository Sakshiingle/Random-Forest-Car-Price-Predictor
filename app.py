from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained pipeline/model
model = pickle.load(open("randomForest_model.pkl", "rb"))

@app.route("/")
def home():
    return "Automobile Price Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data])

        prediction = model.predict(df)

        return jsonify({
            "Predicted Price": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "Error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
