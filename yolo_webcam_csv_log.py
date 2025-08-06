import cv2
from ultralytics import YOLO
import csv
import time

# Load model
model = YOLO("yolov8n.pt")  # or yolov8s.pt

# Webcam setup
cap = cv2.VideoCapture(0)

# CSV setup
csv_file = open("crowd_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp (s)", "Frame", "People Count"])

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.3)
    annotated_frame = results[0].plot()

    # Count people
    person_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name.lower() == "person":
            person_count += 1

    # Show frame with count
    cv2.putText(annotated_frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("YOLOv8 Crowd Detection", annotated_frame)

    # Log to CSV
    timestamp = round(time.time() - start_time, 1)
    csv_writer.writerow([timestamp, frame_count, person_count])

    frame_count += 1

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()
