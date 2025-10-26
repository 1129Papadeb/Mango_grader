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
    "Class1": (221, 280),
    "Class2": (181, 220),
    "Class3": (120, 180)
}


# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Setup Camera
cap = cv2.VideoCapture(0)


# --- Global Zoom ---
zoom_factor = 1.0  # Start with no zoom to capture full mango


def zoom_frame(frame, zoom_factor=1.0):
    # Crop center region to simulate zoom
    if zoom_factor == 1.0:
        return frame
    h, w, _ = frame.shape
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1 = max((w - new_w) // 2, 0)
    y1 = max((h - new_h) // 2, 0)
    x2, y2 = x1 + new_w, y1 + new_h
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))


def enhance_lighting(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def preprocess_image(img):
    img = enhance_lighting(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized.astype('float32') / 255.0
    img_normalized = (img_normalized - 0.5) * 2.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded


def get_random_weight(class_name):
    low, high = CLASS_WEIGHT_RANGES[class_name]
    return random.uniform(low, high)


def update_preview():
    global live_preview_running
    if not live_preview_running:
        return
    ret, frame = cap.read()
    if not ret:
        label_preview.after(100, update_preview)
        return
    frame = zoom_frame(frame, zoom_factor)

    # Consistent 90 degrees clockwise rotation for preview
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    h, w, _ = frame.shape
    preview_w = 480
    preview_h = int(h / w * preview_w)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (preview_w, preview_h))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)

    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)
    label_preview.place(x=0, y=20, width=preview_w, height=preview_h)

    label_result.config(text="")
    label_preview.after(30, update_preview)


def capture_and_grade():
    global live_preview_running
    live_preview_running = False

    ret, frame = cap.read()
    if not ret:
        label_result.config(text="Failed to capture image.")
        return
    frame = zoom_frame(frame, zoom_factor)

    # Same rotation here as preview for consistency
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    processed = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_class = CLASS_NAMES[predicted_index]
    random_weight = get_random_weight(predicted_class)

    h, w, _ = frame.shape
    preview_w = label_preview.winfo_width()
    preview_h = label_preview.winfo_height()

    aspect_ratio = h / w
    target_h = int(preview_w * aspect_ratio)
    if target_h > preview_h:
        target_h = preview_h
        preview_w = int(preview_h / aspect_ratio)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (preview_w, target_h))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)

    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)
    label_preview.place(x=0, y=20, width=preview_w, height=target_h)

    result_text = (f"Prediction:\n{predicted_class}\n\n"
                   f"Weight:\n{random_weight:.1f} g\n\n"
                   f"Confidence:\n{confidence:.2f}")
    label_result.config(text=result_text, font=("Arial", 12))
    label_result.place(x=200, y=20, width=180, height=target_h)

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
    # Limit max zoom to 2.0 to keep mango fully visible
    zoom_factor = min(zoom_factor + 0.2, 2.0)
    if live_preview_running:
        update_preview()


def zoom_out():
    global zoom_factor
    zoom_factor = max(zoom_factor - 0.2, 1.0)
    if live_preview_running:
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
