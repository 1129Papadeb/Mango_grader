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


# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Setup Legacy Camera (cv2)
cap = cv2.VideoCapture(0)


# --- Global Zoom ---
zoom_factor = 1.0  # default no zoom


def zoom_frame(frame, zoom_factor=1.0):
    if zoom_factor == 1.0:
        return frame
    h, w, _ = frame.shape
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


def get_random_weight(class_name):
    low, high = CLASS_WEIGHT_RANGES[class_name]
    return random.uniform(low, high)


def update_preview():
    ret, frame = cap.read()
    if not ret:
        return
    frame = zoom_frame(frame, zoom_factor)

    h, w, _ = frame.shape
    # Portrait preview: fix width and scale height
    new_w = 270
    new_h = int(h / w * new_w)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)
    label_preview.place(x=25, y=20, width=new_w, height=new_h)
    if live_preview_running:
        label_preview.after(30, update_preview)


def capture_and_grade():
    global live_preview_running
    live_preview_running = False
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture image from camera.")

        frame = zoom_frame(frame, zoom_factor)
        processed = preprocess_image(frame)

        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index]
        predicted_class = CLASS_NAMES[predicted_index]

        random_weight = get_random_weight(predicted_class)

        # Resize captured frame maintaining portrait aspect
        h, w, _ = frame.shape
        preview_w = 240
        preview_h = int(h / w * preview_w)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (preview_w, preview_h))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        label_preview.imgtk = imgtk
        label_preview.configure(image=imgtk)
        label_preview.place(x=40, y=20, width=preview_w, height=preview_h)

        result_text = (f"Prediction:\n{predicted_class}\n\n"
                       f"Weight:\n{random_weight:.1f} g\n\n"
                       f"Confidence:\n{confidence:.2f}")
        label_result.config(text=result_text, font=("Arial", 12), justify="center", fg="black")
        label_result.place(x=40, y=400, width=240, height=100)
        label_result.update()

        btn_capture.pack_forget()
        btn_again.pack(side="left", padx=10)
    except Exception as e:
        label_result.config(text=f"Error:\n{str(e)}", font=("Arial", 12), fg="red", justify="center")
        label_result.place(x=20, y=20, width=280, height=100)
        label_result.update()
        btn_capture.pack(side="left", padx=10)
        btn_again.pack_forget()
        live_preview_running = True
        update_preview()


def reset_preview():
    global live_preview_running
    live_preview_running = True
    label_result.config(text="", fg="black")
    label_result.place_forget()
    btn_again.pack_forget()
    btn_capture.pack(side="left", padx=10)
    update_preview()


def zoom_in():
    global zoom_factor
    zoom_factor = min(zoom_factor + 0.2, 3.0)
    update_preview()


def zoom_out():
    global zoom_factor
    zoom_factor = max(zoom_factor - 0.2, 1.0)
    update_preview()


# --- UI Setup ---
root = tk.Tk()
root.title("Mango Grader")

# Portrait window geometry
root.geometry("320x520")

label_preview = tk.Label(root, bg="black")
label_preview.place(x=25, y=20, width=270, height=360)

label_result = tk.Label(root, text="", font=("Arial", 12), justify="center")

# Pre-place label_result but hide initially
label_result.place(x=40, y=400, width=240, height=100)
label_result.place_forget()

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(side="bottom", fill="x", pady=10)

btn_capture = tk.Button(button_frame, text="Capture", command=capture_and_grade, font=("Arial", 12))
btn_capture.pack(side="left", padx=10)

btn_again = tk.Button(button_frame, text="Capture Again", command=reset_preview, font=("Arial", 12))
# btn_again not packed initially

btn_exit = tk.Button(button_frame, text="Exit", command=root.quit, font=("Arial", 12))
btn_exit.pack(side="left", padx=10)

btn_zoom_in = tk.Button(button_frame, text="➕ Zoom In", command=zoom_in, font=("Arial", 10))
btn_zoom_in.pack(side="left", padx=10)

btn_zoom_out = tk.Button(button_frame, text="➖ Zoom Out", command=zoom_out, font=("Arial", 10))
btn_zoom_out.pack(side="left", padx=10)

live_preview_running = True
update_preview()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
