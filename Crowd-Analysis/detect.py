from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Or yolov8s.pt / yolov8m.pt if you have them

# Open the input video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]

    # Draw detections on frame
    annotated_frame = results.plot()

    # Show frame (optional)
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Write frame to output
    out.write(annotated_frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Detection completed. Output saved as 'output_video.mp4'")
