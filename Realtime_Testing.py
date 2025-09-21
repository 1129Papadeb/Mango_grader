import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# --- Config ---
MODEL_PATH = "Mobilenetv2.tflite"
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ["Class1", "Class2", "Class3"]

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# --- Functions ---
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

def capture_and_grade():
    frame = picam2.capture_array()  # Take picture
    processed = preprocess_image(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    result = f"Prediction: {CLASS_NAMES[predicted_index]} ({confidence:.2f})"
    label_result.config(text=result)

    # Ask user if they want another capture
    again = messagebox.askyesno("Continue?", "Do you want to capture another mango?")
    if not again:
        root.quit()

# --- UI ---
root = tk.Tk()
root.title("Mango Grader")

label_instr = tk.Label(root, text="Press Capture to grade a mango", font=("Arial", 14))
label_instr.pack(pady=10)

btn_capture = tk.Button(root, text="📸 Capture", command=capture_and_grade, font=("Arial", 14))
btn_capture.pack(pady=10)

label_result = tk.Label(root, text="", font=("Arial", 14))
label_result.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 14))
btn_exit.pack(pady=10)

root.mainloop()
