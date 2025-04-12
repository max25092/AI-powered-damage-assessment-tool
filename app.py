from flask import Flask, render_template, request, jsonify, Response
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Configure the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}

# Load your trained model
model = tf.keras.models.load_model('damage_assessment_model2.h5')

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process the image (resize and normalize it)
def process_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
    img = Image.fromarray(img)
    img = img.resize((224, 224))  # Adjust to model input size
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to process video frame
def process_video_frame(frame):
    img_array = process_image(frame)
    prediction = model.predict(img_array)
    damage_percentage = prediction[0][0] * 100  # Return damage percentage (multiply by 100 to get the percentage)
    return damage_percentage

# Route for handling the live video
@app.route('/video_feed')
def video_feed():
    def gen():
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame and classify
            damage_percentage = process_video_frame(frame)

            # Add result text to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Damage Percentage: {damage_percentage:.2f}", (10, 30),
                        font, 1, (0, 255, 0) if damage_percentage <= 50 else (0, 0, 255), 2)

            # Encode the frame in JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break

            # Yield the frame in a format that can be streamed as HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Handle file upload and damage assessment
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    damage_percentage = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the file (image)
            img_array = process_image(cv2.imread(file_path))
            prediction = model.predict(img_array)

            # Get the damage percentage (scaled between 0-1)
            damage_percentage = prediction[0][0] * 100  # Scale it back to 0-100% damage

            # Return the result to the front end
            return render_template('index.html', filename=filename, damage_percentage=damage_percentage)

    return render_template('index.html', result=result, filename=filename, damage_percentage=damage_percentage)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
