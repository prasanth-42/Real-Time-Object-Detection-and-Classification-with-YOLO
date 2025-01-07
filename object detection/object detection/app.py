from flask import Flask, request, send_file, render_template, Response, send_from_directory
from io import BytesIO
import os
import cv2
import torch
import contextlib
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Flask app setup
app = Flask(__name__)

# Temporary folder for uploaded files
os.makedirs("temp", exist_ok=True)

# Suppress print statements temporarily
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            yield

# Load YOLO model
print("Loading YOLOv5 model...")
with suppress_stdout():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print("Model loaded successfully!")

# Function to detect objects in an image
def detect_objects_in_image(file_path):
    img = cv2.imread(file_path)
    results = model(img)

    # Draw bounding boxes and labels on the image
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, class_id = result[:6]
        label = f"{model.names[int(class_id)]}: {conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode image as JPEG for response
    _, buffer = cv2.imencode('.jpg', img)
    return BytesIO(buffer)

# Function to generate frames for video detection
def generate_frames(video_source):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection on each frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, class_id = result[:6]
            label = f"{model.names[int(class_id)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for the main page
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')
@app.route('/styles.css')
def serve_css():
    return send_from_directory(os.getcwd(), 'styles.css')

# Route for object detection in images
@app.route('/detect_image', methods=['POST'])
def detect_image():
    file = request.files['file']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Perform detection and get processed image
    processed_image = detect_objects_in_image(file_path)

    # Clean up uploaded file
    os.remove(file_path)

    return send_file(processed_image, mimetype='image/jpeg')

# Route for video file detection
@app.route('/detect_video', methods=['POST'])
def detect_video():
    file = request.files['file']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Stream the video with detection
    return Response(generate_frames(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for live webcam detection
@app.route('/live_detection')
def live_detection():
    # Stream the webcam with detection
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)