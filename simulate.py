import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
import tflite_runtime.interpreter as tflite
import random


# --- Config ---
MODEL_PATH = "MobileNetv2.tflite"
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ["Class1", "Class2", "Class3"]
CLASS_WEIGHT_RANGES = {
    "Class1": (281, 380),
    "Class2": (210, 280),
    "Class3": (120, 180)
}

MIN_CONFIDENCE = 0.6  # Minimum confidence threshold for accepting Class3 predictions


# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Setup Legacy Camera (cv2)
cap = cv2.VideoCapture(0)


# --- Global Zoom ---
zoom_factor = 1.0  # fixed no zoom to show widest camera view


def zoom_frame(frame, zoom_factor=1.0):
    # Disable zoom cropping, return original frame
    return frame


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_normalized = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded


def get_random_weight(class_name):
    low, high = CLASS_WEIGHT_RANGES[class_name]
    return random.uniform(low, high)


def update_preview():
    ret, frame = cap.read()
    if not ret:
        return
    frame = zoom_frame(frame, zoom_factor)

    h, w, _ = frame.shape
    # Fix height, calculate width preserving aspect ratio for landscape preview
    new_h = 270
    new_w = int(w / h * new_h)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)
    label_preview.place(x=0, y=20, width=new_w, height=new_h)
    if live_preview_running:
        label_preview.after(30, update_preview)


def capture_and_grade():
    global live_preview_running
    live_preview_running = False
    ret, frame = cap.read()
    if not ret:
        return
    frame = zoom_frame(frame, zoom_factor)
    processed = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_class = CLASS_NAMES[predicted_index]

    # Filter Class3 predictions by confidence threshold: downgrade to Class1 if below threshold
    if predicted_class == "Class3" and confidence < MIN_CONFIDENCE:
        predicted_class = "Class1"  # or "Class2" depending on fallback preference
        # Optionally, recalculate confidence for fallback class if desired
        # confidence = predictions[CLASS_NAMES.index(predicted_class)]

    random_weight = get_random_weight(predicted_class)

    # Resize captured frame maintaining aspect ratio with preview width 240
    h, w, _ = frame.shape
    preview_w = 240
    preview_h = int(h / w * preview_w)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (preview_w, preview_h))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)
    label_preview.place(x=0, y=20, width=preview_w, height=preview_h)

    result_text = (f"Prediction:\n{predicted_class}\n\n"
                   f"Weight:\n{random_weight:.1f} g\n\n"
                   f"Confidence:\n{confidence:.2f}")
    label_result.config(text=result_text, font=("Arial", 12))
    label_result.place(x=300, y=20, width=180, height=preview_h)

    btn_capture.place_forget()
    btn_again.place(x=20, y=220, width=100, height=40)
    btn_exit.place(x=140, y=220, width=100, height=40)
    btn_zoom_in.place(x=280, y=220, width=80, height=40)
    btn_zoom_out.place(x=380, y=220, width=80, height=40)


def reset_preview():
    global live_preview_running
    live_preview_running = True
    label_result.config(text="")
    btn_again.place_forget()
    btn_capture.place(x=20, y=220, width=100, height=40)
    btn_exit.place(x=140, y=220, width=100, height=40)
    btn_zoom_in.place(x=280, y=220, width=80, height=40)
    btn_zoom_out.place(x=380, y=220, width=80, height=40)
    update_preview()


def zoom_in():
    global zoom_factor
    zoom_factor = 1.0
    update_preview()


def zoom_out():
    global zoom_factor
    zoom_factor = 1.0
    update_preview()


# --- UI Setup ---
root = tk.Tk()
root.title("Mango Grader")

root.geometry("480x300")

label_preview = tk.Label(root, bg="black")
label_preview.place(x=0, y=20, width=480, height=180)

label_result = tk.Label(root, text="", font=("Arial", 12), justify="center")

btn_capture = tk.Button(root, text="Capture", command=capture_and_grade, font=("Arial", 12))
btn_capture.place(x=20, y=220, width=100, height=40)

btn_again = tk.Button(root, text="Capture", command=reset_preview, font=("Arial", 12))

btn_exit = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
btn_exit.place(x=140, y=220, width=100, height=40)

btn_zoom_in = tk.Button(root, text="➕ Zoom In", command=zoom_in, font=("Arial", 10))
btn_zoom_in.place(x=280, y=220, width=80, height=40)

btn_zoom_out = tk.Button(root, text="➖ Zoom Out", command=zoom_out, font=("Arial", 10))
btn_zoom_out.place(x=380, y=220, width=80, height=40)

live_preview_running = True
update_preview()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
