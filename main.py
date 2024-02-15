import tkinter as tk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageTk, Image

# Load the trained model
model = load_model('C:\\Users\\Hp\\PycharmProjects\\pythonProject\\Resources\\Model\\model6.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Define global variables
is_detecting = False

# Define a function for real-time object detection
def detect_objects():
    global is_detecting
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    frame_reshaped = np.reshape(frame_normalized, [-1, 224, 224, 3])

    predictions = model.predict(frame_reshaped)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    if class_index == 0:
        predicted_class = "Electronic Waste"
    elif class_index == 1:
        predicted_class = "Food Waste"
    elif class_index == 2:
        predicted_class = "Glass"
    elif class_index == 3:
        predicted_class = "Metal"
    elif class_index == 4:
        predicted_class = "Paper"
    elif class_index == 5:
        predicted_class = "Plastic"

    if is_detecting:
        label.config(text=f"Predicted Class: {predicted_class}\nConfidence: {confidence * 100:.2f}%")
    else:
        label.config(text="Object detection stopped")

    # Update the video feed
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if is_detecting:
        video_label.after(10, detect_objects)

# Function to start object detection
def start_detection():
    global is_detecting
    is_detecting = True
    detect_objects()

# Function to stop object detection
def stop_detection():
    global is_detecting
    is_detecting = False

# Create the Tkinter window
window = tk.Tk()
window.title("Real-Time Waste Classification")
window.geometry("800x600")

# Create a label for the video feed
video_label = tk.Label(window)
video_label.pack(pady=10)

# Create a label to display the predicted class and confidence
label = tk.Label(window, font=("Arial", 16))
label.pack(pady=10)

# Create start and stop buttons
start_button = tk.Button(window, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

stop_button = tk.Button(window, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)

# Apply a stylish template and design
window.configure(bg='#F0F0F0')  # Set background color

#title_label = tk.Label(window, text="GreenSort", font=("Arial", 30, "bold"), bg='#F0F0F0')
#title_label.pack(pady=20)

# Set button colors and font
button_color = '#4CAF50'  # Green
button_font = ("Arial", 14, "bold")

start_button.configure(bg=button_color, fg='white', font=button_font)
stop_button.configure(bg=button_color, fg='white', font=button_font)

# Insert an image related to waste classification
image_path = 'C:\\Users\\Hp\\PycharmProjects\\pythonProject\\Resources\\abc.png'
image = Image.open(image_path)
image = image.resize((400, 300), Image.Resampling.LANCZOS)
image = ImageTk.PhotoImage(image)

image_label = tk.Label(window, image=image)
image_label.pack(pady=20)

# Exit the program when the window is closed
window.protocol("WM_DELETE_WINDOW", window.quit)

# Run the Tkinter event loop
window.mainloop()

# Release the webcam.
cap.release()
