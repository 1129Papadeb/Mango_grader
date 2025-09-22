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

# --- Functions ---
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

def update_preview():
    """ Continuously update live preview """
    ret, frame = cap.read()
    if not ret:
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (480, 180))

    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)

    if live_preview_running:
        label_preview.after(30, update_preview)

def capture_and_grade():
    """ Capture image, run grading, show result """
    global live_preview_running
    live_preview_running = False  # stop live feed

    ret, frame = cap.read()
    if not ret:
        return

    processed = preprocess_image(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    # Add grading text on image
    frame_out = frame.copy()
    cv2.putText(frame_out,
                f"{CLASS_NAMES[predicted_index]} ({confidence:.2f})",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    # Show output on TFT
    frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (480, 180))
    img = Image.fromarray(frame_resized)
    imgtk = ImageTk.PhotoImage(image=img)
    label_preview.imgtk = imgtk
    label_preview.configure(image=imgtk)

    # Show result text
    result = f"Prediction: {CLASS_NAMES[predicted_index]} ({confidence:.2f})"
    label_result.config(text=result)

    # Switch buttons
    btn_capture.place_forget()
    btn_again.place(x=150, y=220, width=180, height=40)

def reset_preview():
    """ Return to live preview """
    global live_preview_running
    live_preview_running = True
    btn_again.place_forget()
    btn_capture.place(x=150, y=220, width=180, height=40)
    update_preview()

# --- UI Setup ---
root = tk.Tk()
root.title("Mango Grader")
root.geometry("480x320")  # TFT resolution

label_instr = tk.Label(root, text="Align the mango and press Capture", font=("Arial", 10))
label_instr.place(x=10, y=5)

# Preview area
label_preview = tk.Label(root)
label_preview.place(x=0, y=30, width=480, height=180)

# Capture button
btn_capture = tk.Button(root, text="📸 Capture", command=capture_and_grade, font=("Arial", 12))
btn_capture.place(x=150, y=220, width=180, height=40)

# Capture Again button (hidden initially)
btn_again = tk.Button(root, text="🔄 Capture Again", command=reset_preview, font=("Arial", 12))

# Result label
label_result = tk.Label(root, text="", font=("Arial", 12))
label_result.place(x=10, y=270)

# Exit button
btn_exit = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
btn_exit.place(x=400, y=270, width=70, height=30)

# Start live preview
live_preview_running = True
update_preview()

root.mainloop()

# Release camera when closing
cap.release()
cv2.destroyAllWindows()
