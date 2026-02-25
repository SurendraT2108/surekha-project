from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        salary = float(request.form["salary"])
        experience = float(request.form["experience"])
        job_role = int(request.form["job_role"])

        features = np.array([[age, salary, experience, job_role]])
        result = model.predict(features)[0]

        output = "Attrition Risk" if result == 1 else "No Attrition"
        return render_template("index.html", prediction=output)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
