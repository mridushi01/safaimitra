import tensorflow as tf
import numpy as np
import cv2
import serial
import time
import csv
from datetime import datetime

# ==============================
# Load AI Model
# ==============================

model = tf.keras.models.load_model("keras_model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ==============================
# Connect Arduino
# ==============================

arduino = serial.Serial('/dev/cu.usbserial-110', 9600, timeout=1)
time.sleep(2)

# ==============================
# Start Camera
# ==============================

cap = cv2.VideoCapture(0)

# ==============================
# Dashboard File
# ==============================

dashboard_file = "dashboard.csv"

# Create file if not exists
try:
    open(dashboard_file, 'x').close()
    with open(dashboard_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "City", "Waste Type", "Reward", "Time"])
except:
    pass


print("\n===== SMART DUSTBIN SYSTEM =====")

# ==============================
# Ask User Details
# ==============================

name = input("Enter your name: ")
city = input("Enter your city (Indore/Gangtok): ")

print("\nHello", name)
print("City:", city)
print("Please place waste in front of camera...")

time.sleep(3)

# ==============================
# Capture Image
# ==============================

ret, frame = cap.read()

img = cv2.resize(frame, (224, 224))
img_array = np.asarray(img)
img_array = img_array / 255.0
img_array = np.reshape(img_array, (1, 224, 224, 3))

# ==============================
# Predict Waste
# ==============================

prediction = model.predict(img_array, verbose=0)
class_name = labels[np.argmax(prediction)]

print("\nDetected Waste:", class_name)

# ==============================
# Motor Control
# ==============================

if "bio" in class_name.lower():
    arduino.write(b'B')
    reward = 10

elif "non" in class_name.lower():
    arduino.write(b'N')
    reward = 5

# ==============================
# Reward System
# ==============================

print("Reward Earned:", reward, "points")

# ==============================
# Save to Dashboard
# ==============================

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(dashboard_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([name, city, class_name, reward, current_time])

print("Dashboard Updated")

# ==============================
# Show Result on Screen
# ==============================

cv2.putText(frame, "Name: " + name, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.putText(frame, "City: " + city, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.putText(frame, "Waste: " + class_name, (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.putText(frame, "Reward: " + str(reward), (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("Smart Dustbin", frame)

cv2.waitKey(5000)

# ==============================
# Cleanup
# ==============================

cap.release()
cv2.destroyAllWindows()
arduino.close()

print("\nThank you for keeping city clean!")
