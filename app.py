from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load saved model (pipeline)
model = joblib.load("model/titanic_survival_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        pclass = int(request.form["Pclass"])
        sex = request.form["Sex"]
        age = float(request.form["Age"])
        fare = float(request.form["Fare"])
        embarked = request.form["Embarked"]

        # Create DataFrame for the pipeline
        sample = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "Embarked": embarked
        }])

        # Make prediction
        prediction = model.predict(sample)[0]

        # Prepare result string
        result = "Survived ✅" if prediction == 1 else "Did Not Survive ❌"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
