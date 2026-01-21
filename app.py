import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        x = float(request.form["x"])
        x_scaled = scaler.transform([[x]])
        prediction = model.predict(x_scaled)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
