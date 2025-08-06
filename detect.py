from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button

def run_yolo_detection(video_path):
    model = YOLO("yolov8n.pt")  # Change to yolov8s.pt or yolov8m.pt if needed

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()

        cv2.imshow("YOLOv8 Detection", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Detection completed. Output saved as 'output_video.mp4'")


def browse_and_run():
    filepath = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
    )
    if filepath:
        label.config(text=f"Selected: {filepath}")
        run_yolo_detection(filepath)


# GUI Setup
root = tk.Tk()
root.title("YOLOv8 Video Detection")
root.geometry("500x200")

label = Label(root, text="No video selected", font=("Arial", 12))
label.pack(pady=20)

browse_btn = Button(root, text="Browse Video", command=browse_and_run, font=("Arial", 14))
browse_btn.pack(pady=10)

exit_btn = Button(root, text="Exit", command=root.quit, font=("Arial", 12))
exit_btn.pack(pady=10)

root.mainloop()
