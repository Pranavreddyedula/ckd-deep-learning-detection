from flask import Flask, render_template, request
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("models/ckd_dl_model.h5")
scaler = joblib.load("models/scaler.pkl")

FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        values = []
        for f in FEATURES:
            v = request.form.get(f)
            if v == "" or v is None:
                v = 0
            values.append(float(v))

        data = scaler.transform([values])
        pred = model.predict(data)[0][0]

        if pred > 0.5:
            result = "Chronic Kidney Disease Detected"
            image = "ckd_kidney.png"
            risk = "High"
        else:
            result = "Healthy Kidney"
            image = "healthy_kidney.png"
            risk = "Low"

        confidence = round(float(pred) * 100, 2)

        return render_template(
            "result.html",
            result=result,
            image=image,
            risk=risk,
            confidence=confidence,
            accuracy_img="accuracy.png",
            cm_img="confusion_matrix.png"
        )

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
