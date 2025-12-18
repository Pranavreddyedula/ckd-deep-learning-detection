from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------- LAZY LOAD ARTIFACTS ----------
model = None
scaler = None

def load_artifacts():
    global model, scaler
    if model is None or scaler is None:
        model = load_model("models/ckd_dl_model.h5")
        scaler = joblib.load("models/scaler.pkl")


FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
]

# ---------- MAIN ROUTE ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        load_artifacts()  # 🔥 LOAD MODEL ONLY WHEN NEEDED

        values = []
        for f in FEATURES:
            v = request.form.get(f)
            if v is None or v == "":
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
        model_accuracy = 96.8

        return render_template(
            "result.html",
            result=result,
            image=image,
            risk=risk,
            confidence=confidence,
            accuracy=model_accuracy,
            accuracy_img="accuracy.png",
            cm_img="confusion_matrix.png"
        )

    return render_template("index.html")


# ---------- DOWNLOAD REPORT ----------
@app.route("/download")
def download_report():
    report = f"""
CHRONIC KIDNEY DISEASE REPORT
----------------------------
Result      : {request.args.get('result')}
Risk Level  : {request.args.get('risk')}
Confidence  : {request.args.get('confidence')} %

Model Accuracy : 96.8 %

NOTE:
This AI-based system is for educational
and research purposes only.
"""

    with open("CKD_Report.txt", "w") as f:
        f.write(report)

    return send_file("CKD_Report.txt", as_attachment=True)


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
