import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
import tflite_runtime.interpreter as tflite

# --- Config ---
MODEL_PATH = "MobileNetv2.tflite"
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ["Class1", "Class2", "Class3"]

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup Legacy Camera (cv2)
cap = cv2.VideoCapture(0)

# --- Global Zoom ---
zoom_factor = 1.0  # default no zoom

# --- Functions ---
def zoom_frame(frame, zoom_factor=1.0):
    """ Apply digital zoom by cropping and resizing """
    if zoom_factor == 1.0:
        return frame  # no zoom
    
    h, w, _ = frame.shape
    # Calculate cropping box
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2, y2 = x1 + new_w, y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_normalized = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

def update_preview():
    """ Continuously update preview """
    ret, frame = cap.read()
    if not ret:
        return

    # Apply zoom
    frame = zoom_frame(frame, zoom_factor)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (480, 200))  # smaller preview

    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)

    if live_preview_running:
        label_preview.after(30, update_preview)

def capture_and_grade():
    """ Capture image, stop preview, and show split screen result """
    global live_preview_running
    live_preview_running = False  # stop live feed

    ret, frame = cap.read()
    if not ret:
        return

    # Apply zoom before processing
    frame = zoom_frame(frame, zoom_factor)

    processed = preprocess_image(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    # Left side: captured mango image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (240, 200))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)
    label_preview.place(x=0, y=20, width=240, height=200)

    # Right side: grading text
    result_text = f"Prediction:\n{CLASS_NAMES[predicted_index]}\n\nConfidence:\n{confidence:.2f}"
    label_result.config(text=result_text, font=("Arial", 12))
    label_result.place(x=240, y=20, width=240, height=200)

    # Switch buttons
    btn_capture.place_forget()
    btn_again.place(x=80, y=240, width=150, height=30)
    btn_exit.place(x=260, y=240, width=120, height=30)

def reset_preview():
    """ Return to preview mode """
    global live_preview_running
    live_preview_running = True

    # Expand preview again
    label_preview.place(x=0, y=20, width=480, height=200)

    # Clear text on right side
    label_result.config(text="")

    # Switch buttons
    btn_again.place_forget()
    btn_capture.place(x=80, y=240, width=150, height=30)
    btn_exit.place(x=260, y=240, width=120, height=30)

    update_preview()

# --- Zoom Controls ---
def zoom_in():
    global zoom_factor
    zoom_factor = min(zoom_factor + 0.2, 3.0)  # max 3x zoom

def zoom_out():
    global zoom_factor
    zoom_factor = max(zoom_factor - 0.2, 1.0)  # min normal

# --- UI Setup ---
root = tk.Tk()
root.title("Mango Grader")
root.geometry("480x320")  # TFT screen

# Preview
label_preview = tk.Label(root, bg="black")
label_preview.place(x=0, y=20, width=480, height=200)

# Right-side result (hidden during preview)
label_result = tk.Label(root, text="", font=("Arial", 12), justify="center")

# Buttons
btn_capture = tk.Button(root, text="📸 Capture", command=capture_and_grade, font=("Arial", 12))
btn_capture.place(x=80, y=240, width=150, height=30)

btn_again = tk.Button(root, text="🔄 Capture Again", command=reset_preview, font=("Arial", 12))

btn_exit = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
btn_exit.place(x=260, y=240, width=120, height=30)

# Zoom buttons
btn_zoom_in = tk.Button(root, text="➕ Zoom In", command=zoom_in, font=("Arial", 10))
btn_zoom_in.place(x=10, y=280, width=100, height=30)

btn_zoom_out = tk.Button(root, text="➖ Zoom Out", command=zoom_out, font=("Arial", 10))
btn_zoom_out.place(x=120, y=280, width=100, height=30)

# Start preview
live_preview_running = True
update_preview()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
