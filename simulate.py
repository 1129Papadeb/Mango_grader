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

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Camera setup
cap = cv2.VideoCapture(0)

# Zoom variable
zoom_factor = 1.0

def zoom_frame(frame, zoom=1.0):
    if zoom == 1.0:
        return frame
    h, w, _ = frame.shape
    new_w, new_h = int(w / zoom), int(h / zoom)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    return cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

def preprocess_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_HEIGHT, IMG_WIDTH))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)

def get_random_weight(class_name):
    low, high = CLASS_WEIGHT_RANGES[class_name]
    return random.uniform(low, high)

def update_preview():
    ret, frame = cap.read()
    if not ret:
        return
    frame = zoom_frame(frame, zoom_factor)
    h, w, _ = frame.shape
    new_w = 270
    new_h = int(h / w * new_w)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (new_w, new_h))
    img = Image.fromarray(resized)
    photo = ImageTk.PhotoImage(image=img)
    label_preview.config(image=photo)
    label_preview.image = photo
    label_preview.place(x=25, y=20, width=new_w, height=new_h)
    if live_preview_running:
        label_preview.after(30, update_preview)

def capture_and_grade():
    global live_preview_running
    live_preview_running = False
    print("Capture started")
    try:
        ret, frame = cap.read()
        print("Camera capture ret:", ret)
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")

        frame = zoom_frame(frame, zoom_factor)
        processed = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        print("Raw predictions:", predictions)
        pred_idx = np.argmax(predictions)
        confidence = predictions[pred_idx]
        pred_class = CLASS_NAMES[pred_idx]
        print(f"Predicted class: {pred_class}, Confidence: {confidence}")

        weight = get_random_weight(pred_class)

        h, w, _ = frame.shape
        pw = 240
        ph = int(h / w * pw)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (pw, ph))
        img = Image.fromarray(resized_img)
        photo = ImageTk.PhotoImage(image=img)
        label_preview.config(image=photo)
        label_preview.image = photo
        label_preview.place(x=40, y=20, width=pw, height=ph)

        result_text = f"Prediction:\n{pred_class}\n\nWeight:\n{weight:.1f} g\n\nConfidence:\n{confidence:.2f}"
        label_result.config(text=result_text, font=("Arial", 12), fg="black")
        label_result.place(x=40, y=400, width=240, height=100)
        label_result.update()
        print("Result label updated")

        btn_capture.pack_forget()
        btn_again.pack(side="left", padx=10)

    except Exception as e:
        print("Exception:", e)
        label_result.config(text=f"Error:\n{str(e)}", font=("Arial", 12), fg="red")
        label_result.place(x=20, y=20, width=280, height=100)
        label_result.update()
        btn_capture.pack(side="left", padx=10)
        btn_again.pack_forget()
        live_preview_running = True
        update_preview()

def reset_preview():
    global live_preview_running
    live_preview_running = True
    label_result.config(text="")
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

root = tk.Tk()
root.title("Mango Grader")
root.geometry("320x520")

label_preview = tk.Label(root, bg="black")
label_preview.place(x=25, y=20, width=270, height=360)

label_result = tk.Label(root, text="", font=("Arial", 12), justify="center")
label_result.place(x=40, y=400, width=240, height=100)
label_result.place_forget()

button_frame = tk.Frame(root)
button_frame.pack(side="bottom", fill="x", pady=10)

btn_capture = tk.Button(button_frame, text="Capture", command=capture_and_grade, font=("Arial", 12))
btn_capture.pack(side="left", padx=10)

btn_again = tk.Button(button_frame, text="Capture Again", command=reset_preview, font=("Arial", 12))

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
