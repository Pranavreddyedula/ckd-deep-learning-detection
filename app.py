from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("models/ckd_dl_model.h5")
scaler = joblib.load("models/scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect values in correct order
        values = [float(request.form.get(f, 0)) for f in request.form]
        data = scaler.transform([values])

        pred = model.predict(data)[0][0]

        if pred > 0.5:
            result = "Chronic Kidney Disease Detected"
            image = "ckd_kidney.png"
        else:
            result = "Healthy Kidney"
            image = "healthy_kidney.png"

        return render_template(
            'result.html',
            result=result,
            image=image
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
