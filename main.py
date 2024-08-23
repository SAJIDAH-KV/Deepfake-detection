from flask import Flask, render_template, request, redirect, url_for, flash, get_flashed_messages
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generating a random secret key

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable for MAX_SEQ_LENGTH
MAX_SEQ_LENGTH = 20

# Load the image detection model
image_model = tf.keras.models.load_model('detection_model.h5')

# Load the video detection model
video_model = tf.keras.models.load_model('deepfake_video_model.h5')

# Define the build_feature_extractor function for video detection
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((224, 224, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# Function to check if file extension is allowed
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Detection route for image
@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Perform image detection
        result = analyze_image(filepath)
        return render_template('result.html', result=result)
    else:
        flash('Invalid file type. Only JPG, JPEG, and PNG images are allowed.', 'error')
        return redirect(request.url)

# Detection route for video
@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Perform video detection
        result = analyze_video(filepath)
        return render_template('result.html', result=result)
    else:
        flash('Invalid file type. Only MP4 videos are allowed.', 'error')
        return redirect(request.url)

# Function to analyze image for deepfake content
def analyze_image(filepath):
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = image_model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "REAL"
    else:
        return "FAKE"

# Function to analyze video for deepfake content
def analyze_video(filepath):
    # Function to load and preprocess video frames
    def load_and_preprocess_video(video_path, max_frames=0, resize=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)
                frames.append(frame)
                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)

    # Load video frames and preprocess
    frames = load_and_preprocess_video(filepath, max_frames=MAX_SEQ_LENGTH)
    frames = frames.astype(np.float32) / 255.0

    # Ensure the correct shape for frame features input
    frame_features_input = np.zeros((1, MAX_SEQ_LENGTH, 2048))

    # Create mask input (assuming all frames are valid)
    mask_input = np.ones((1, MAX_SEQ_LENGTH), dtype=bool)

    # Define the feature extractor model
    feature_extractor = build_feature_extractor()

    # Extract features for each frame and populate frame_features_input
    for i, frame in enumerate(frames):
        frame_features_input[0, i, :] = feature_extractor.predict(frame.reshape(1, 224, 224, 3))

    # Make predictions
    predictions = video_model.predict([frame_features_input, mask_input])

    # Assuming binary classification (0 for real, 1 for fake)
    if predictions.mean() >= 0.5:
        return "Fake"
    else:
        return "Real"

if __name__ == '__main__':
    app.run(debug=True)






# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename
# import os
# import cv2
# import numpy as np
# from tensorflow import keras
#
# app = Flask(__name__)
# app.secret_key = os.urandom(24)  # Generating a random secret key
#
# # Configure upload folder
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# ALLOWED_VIDEO_EXTENSIONS = {'mp4'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# # Create upload folder if it doesn't exist
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# # Global variable for MAX_SEQ_LENGTH
# MAX_SEQ_LENGTH = 20
#
# # Load the image detection model
# image_model = keras.models.load_model('detection_model.h5')
#
# # Load the video detection model
# video_model = keras.models.load_model('deepfake_video_model_new.h5')
#
# # Define the build_feature_extractor function for video detection
# def build_feature_extractor():
#     feature_extractor = keras.applications.InceptionV3(
#         weights="imagenet",
#         include_top=False,
#         pooling="avg",
#         input_shape=(224, 224, 3),
#     )
#     preprocess_input = keras.applications.inception_v3.preprocess_input
#
#     inputs = keras.Input((224, 224, 3))
#     preprocessed = preprocess_input(inputs)
#
#     outputs = feature_extractor(preprocessed)
#     return keras.Model(inputs, outputs, name="feature_extractor")
#
# # Function to check if file extension is allowed
# def allowed_file(filename, allowed_extensions):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
#
# # Home page
# @app.route('/')
# def home():
#     return render_template('index.html')
#
# # Detection route for image upload
# @app.route('/detect_image', methods=['POST'])
# def detect_image():
#     if 'file' not in request.files:
#         flash('No file part', 'error')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No selected file', 'error')
#         return redirect(request.url)
#     if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         # Perform image detection
#         result = analyze_image(filepath)
#         return render_template('result.html', result=result)
#     else:
#         flash('Invalid file type. Only JPG, JPEG, and PNG images are allowed.', 'error')
#         return redirect(request.url)
#
# # Detection route for video upload
# @app.route('/detect_video', methods=['POST'])
# def detect_video():
#     if 'file' not in request.files:
#         flash('No file part', 'error')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No selected file', 'error')
#         return redirect(request.url)
#     if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         # Perform video detection
#         result = analyze_video(filepath)
#         return render_template('result.html', result=result)
#     else:
#         flash('Invalid file type. Only MP4 videos are allowed.', 'error')
#         return redirect(request.url)
#
# # Route for real-time video detection
# @app.route('/detect_realtime_video')
# def detect_realtime_video():
#     # Perform real-time video detection
#     result = analyze_realtime_video()
#     return render_template('result.html', result=result)
#
# # Route for real-time image detection
# @app.route('/detect_realtime_image')
# def detect_realtime_image():
#     # Perform real-time image detection
#     result = analyze_realtime_image()
#     return render_template('result.html', result=result)
#
# # Function to analyze image for deepfake content
# def analyze_image(filepath):
#     img = keras.preprocessing.image.load_img(filepath, target_size=(256, 256))
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     prediction = image_model.predict(img_array)
#     if prediction[0][0] > 0.5:
#         return "REAL"
#     else:
#         return "FAKE"
#
#
# # Define the function to preprocess video frames
# def preprocess_frame(frame):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = cv2.resize(frame, (224, 224))
#     frame = frame.astype(np.float32) / 255.0
#     return frame
#
#
# # Function to analyze video for deepfake content
# def analyze_video(filepath):
#     # Open the video file
#     cap = cv2.VideoCapture(filepath)
#     if not cap.isOpened():
#         return "Error: Unable to open video file"
#
#     predictions = []
#     while True:
#         ret, frame = cap.read()  # Read a frame from the video
#         if not ret:
#             break
#
#         # Preprocess the frame
#         preprocessed_frame = preprocess_frame(frame)
#
#         # Make prediction on the frame
#         prediction = video_model.predict(np.expand_dims(preprocessed_frame, axis=0))
#         predictions.append(prediction)
#
#     cap.release()
#
#     # Perform classification based on predictions
#     mean_prediction = np.mean(predictions)
#     if mean_prediction >= 0.5:
#         return "FAKE"
#     else:
#         return "REAL"
#
#
# # Function to analyze real-time video for deepfake content
# def analyze_realtime_video():
#     # Open a video capture object for the default camera (0)
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         return "Error: Unable to open camera feed"
#
#     predictions = []
#     while True:
#         ret, frame = cap.read()  # Read a frame from the video feed
#         if not ret:
#             break
#
#         # Preprocess the frame
#         preprocessed_frame = preprocess_frame(frame)
#
#         # Make prediction on the frame
#         prediction = video_model.predict(np.expand_dims(preprocessed_frame, axis=0))
#         predictions.append(prediction)
#
#         # Display the frame
#         cv2.imshow('Real-time Video Detection', frame)
#
#         # Check for key press to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     # Perform classification based on predictions
#     mean_prediction = np.mean(predictions)
#     if mean_prediction >= 0.5:
#         return "FAKE"
#     else:
#         return "REAL"
#
# # Function to analyze real-time image for deepfake content
# def analyze_realtime_image():
#     # Open a video capture object for the default camera (0)
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()  # Read a frame from the video feed
#         if not ret:
#             break
#         # Preprocess the frame (resize, normalize, etc.)
#         frame = cv2.resize(frame, (256, 256))  # Resize frame to match model input size
#         frame = frame / 255.0  # Normalize pixel values
#         frame = np.expand_dims(frame, axis=0)  # Add batch dimension
#         # Predict whether the frame contains a deepfake or not
#         prediction = image_model.predict(frame)
#         # Define threshold for classification
#         threshold = 0.5
#         if prediction[0][0] > threshold:
#             result = "FAKE"
#         else:
#             result = "REAL"
#         # Display the result on the frame
#         cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         # Display the frame
#         cv2.imshow('Real-time Image Detection', frame)
#         # Check for key press to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     # Release the video capture object and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     app.run(debug=True)
