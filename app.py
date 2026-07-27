from flask import Flask, request

import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("randomForest_model.pkl", "rb"))


@app.route("/")
def home():

    return """
    <h2>Automobile Price Prediction</h2>

    <form action="/predict" method="post">

    Year:
    <input type="number" name="Year"><br><br>

    Engine Size:
    <input type="number" step="0.1" name="Engine_Size"><br><br>

    Mileage:
    <input type="number" name="Mileage"><br><br>

    Horsepower:
    <input type="number" name="Horsepower"><br><br>

    Torque:
    <input type="number" name="Torque"><br><br>

    Owners:
    <input type="number" name="Owners"><br><br>

    Fuel Efficiency:
    <input type="number" step="0.1" name="Fuel_Efficiency"><br><br>

    Make:
    <input type="text" name="Make"><br><br>

    Model:
    <input type="text" name="Model"><br><br>

    Fuel Type:
    <input type="text" name="Fuel_Type"><br><br>

    Transmission:
    <input type="text" name="Transmission"><br><br>

    Accident History:
    <input type="text" name="Accident_History"><br><br>

    Service History:
    <input type="text" name="Service_History"><br><br>

    Color:
    <input type="text" name="Color"><br><br>

    Body Type:
    <input type="text" name="Body_Type"><br><br>

    Drivetrain:
    <input type="text" name="Drivetrain"><br><br>

    Location:
    <input type="text" name="Location"><br><br>

    <input type="submit" value="Predict">

    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():

    data = pd.DataFrame({

        "Make":[request.form["Make"]],
        "Model":[request.form["Model"]],
        "Year":[int(request.form["Year"])],
        "Fuel_Type":[request.form["Fuel_Type"]],
        "Transmission":[request.form["Transmission"]],
        "Engine_Size":[float(request.form["Engine_Size"])],
        "Mileage":[float(request.form["Mileage"])],
        "Horsepower":[float(request.form["Horsepower"])],
        "Torque":[float(request.form["Torque"])],
        "Owners":[int(request.form["Owners"])],
        "Accident_History":[request.form["Accident_History"]],
        "Service_History":[request.form["Service_History"]],
        "Color":[request.form["Color"]],
        "Body_Type":[request.form["Body_Type"]],
        "Drivetrain":[request.form["Drivetrain"]],
        "Fuel_Efficiency":[float(request.form["Fuel_Efficiency"])],
        "Location":[request.form["Location"]]

    })

    prediction = model.predict(data)

    return f"<h1>Predicted Price : ${prediction[0]:,.2f}</h1>"


if __name__ == "__main__":
    app.run(debug=True)
