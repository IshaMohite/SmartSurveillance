from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox

def run_yolo_detection(video_path):
    model = YOLO("yolov8n.pt")  # You can switch model size here

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
    messagebox.showinfo("Done", "✅ Detection completed.\nOutput saved as 'output_video.mp4'")


def browse_and_run():
    filepath = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
    )
    if filepath:
        file_label.config(text=f"📂 Selected:\n{filepath}", foreground="green")
        run_yolo_detection(filepath)


# ---------- GUI Setup ----------
root = tk.Tk()
root.title("🎯 YOLOv8 Video Detection")
root.geometry("600x300")
root.configure(bg="#f4f4f4")  # Light gray background

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12), padding=10)
style.configure("TLabel", font=("Segoe UI", 11), background="#f4f4f4")

# Title
title_label = ttk.Label(root, text="👁️ Smart Surveillance - Crowd Detection", font=("Segoe UI", 16, "bold"))
title_label.pack(pady=(20, 10))

# File selection label
file_label = ttk.Label(root, text="No video selected", wraplength=500, justify="center")
file_label.pack(pady=10)

# Browse button
browse_btn = ttk.Button(root, text="📁 Browse Video", command=browse_and_run)
browse_btn.pack(pady=10)

# Exit button
exit_btn = ttk.Button(root, text="❌ Exit", command=root.quit)
exit_btn.pack(pady=10)

# Run GUI
root.mainloop()
