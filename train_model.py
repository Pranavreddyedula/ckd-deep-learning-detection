# =====================================
# CKD Prediction Final Training Script
# =====================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------
# Create folders
# -------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# -------------------------------------
# Load Dataset
# -------------------------------------
data = pd.read_csv("ckd.csv")

# Safe ID drop
if "id" in data.columns:
    data = data.drop("id", axis=1)

# -------------------------------------
# Replace ? with NaN
# -------------------------------------
data = data.replace("?", np.nan)

# -------------------------------------
# Clean Classification Column
# -------------------------------------
data["classification"] = data["classification"].str.strip()

data["classification"] = data["classification"].map({
    "ckd": 1,
    "notckd": 0
})

data = data.dropna(subset=["classification"])

# -------------------------------------
# Handle Missing Values
# -------------------------------------
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

# -------------------------------------
# Convert Categorical Columns
# -------------------------------------
categorical_cols = data.select_dtypes(include="object").columns

for col in categorical_cols:
    data[col] = pd.factorize(data[col])[0]

# -------------------------------------
# Split Features and Target
# -------------------------------------
X = data.drop("classification", axis=1)
y = data["classification"]

# -------------------------------------
# Train Test Split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------
# Feature Scaling
# -------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# -------------------------------------
# Build Deep Learning Model
# -------------------------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------
# Early Stopping
# -------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -------------------------------------
# Train Model
# -------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------------
# Save Model
# -------------------------------------
model.save("models/ckd_dl_model.keras")

# -------------------------------------
# Accuracy Graph
# -------------------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.savefig("static/accuracy.png")
plt.close()

# -------------------------------------
# Predictions
# -------------------------------------
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

# -------------------------------------
# Confusion Matrix
# -------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "CKD"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")
plt.close()

# -------------------------------------
# Heatmap Confusion Matrix
# -------------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=["Healthy", "CKD"],
    yticklabels=["Healthy", "CKD"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.savefig("static/confusion_matrix_heatmap.png")
plt.close()

# -------------------------------------
# ROC Curve
# -------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--')
plt.legend()
plt.savefig("static/roc_curve.png")
plt.close()

# -------------------------------------
# Precision Recall Curve
# -------------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.title("Precision Recall Curve")
plt.savefig("static/precision_recall_curve.png")
plt.close()

# -------------------------------------
# Classification Report
# -------------------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
