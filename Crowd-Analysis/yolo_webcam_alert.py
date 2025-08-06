import tkinter as tk
from threading import Thread
import tkinter.messagebox

import cv2
from ultralytics import YOLO
import os
def show_popup(message):
    def popup():
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        tk.messagebox.showwarning("Crowd Alert!", message)
        root.destroy()

    Thread(target=popup).start()


# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt etc.

# Setup webcam
cap = cv2.VideoCapture(0)

# Create output folder
os.makedirs("output_frames", exist_ok=True)

frame_count = 0
alert_threshold = 2  # Set your threshold for number of people

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to grab frame")
        break

    # YOLO detection
    results = model(frame, imgsz=640, conf=0.3)
    annotated_frame = results[0].plot()

    # Save the frame to disk
    #cv2.imwrite(f"output_frames/frame_{frame_count}.jpg", annotated_frame)

    # Count number of people detected
    person_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        # Show class & confidence in terminal
        print(f"Detected: {class_name} ({conf:.2f})")

        # Count only persons
        if class_name.lower() == "person":
            person_count += 1

    # Trigger alert if threshold exceeded
    if person_count > alert_threshold:
        print(f"🚨 Alert! Crowd detected: {person_count} people")
        show_popup(f"Crowd Detected!\nPeople: {person_count}")

    
    # if person_count > alert_threshold:
    #     print(f"🚨 Alert! Crowd detected: {person_count} people")

    # Display frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    frame_count += 1

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
