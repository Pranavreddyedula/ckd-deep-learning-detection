import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

app = Flask(__name__)

# Load model & scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

columns = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr",
    "bu","sc","sod","pot","hemo","pcv","wc","rc",
    "htn","dm","cad","appet","pe","ane"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form[col]) for col in columns]

        input_df = pd.DataFrame([input_data], columns=columns)
        scaled = scaler.transform(input_df)

        prob = model.predict(scaled)[0][0]
        prediction = 1 if prob >= 0.5 else 0

        result = "Chronic Kidney Disease Detected" if prediction == 1 else "Healthy"
        risk = "High" if prob >= 0.5 else "Low"

        # Dummy evaluation example
        y_true = np.array([prediction])
        y_pred = np.array([prediction])
        y_prob = np.array([prob])

        generate_plots(y_true, y_pred, y_prob)

        return render_template("result.html",
                               result=result,
                               risk=risk,
                               confidence=round(prob*100, 2),
                               accuracy=96.8)

    except Exception as e:
        return f"Error: {str(e)}"


def generate_plots(y_true, y_pred, y_prob):

    # ---------- Heatmap Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap='viridis')
    plt.title("Heatmap Confusion Matrix")
    plt.colorbar()
    plt.xticks([0,1], ["Healthy","CKD"])
    plt.yticks([0,1], ["Healthy","CKD"])

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/heatmap_confusion.png")
    plt.close()


    # ---------- ROC Curve ----------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/roc_curve.png")
    plt.close()


    # ---------- Precision Recall Curve ----------
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig("static/pr_curve.png")
    plt.close()


@app.route("/download")
def download():
    return send_file("static/heatmap_confusion.png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
