import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import joblib


# Load data
data = pd.read_csv("ckd.csv")
X = data.drop("class", axis=1)
y = data["class"]


# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# Model
model = Sequential([
Dense(64, activation='relu', input_shape=(24,)),
Dropout(0.3),
Dense(32, activation='relu'),
Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


model.save("models/ckd_dl_model.h5")


# Accuracy Graph
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.legend()
plt.savefig('static/accuracy.png')
