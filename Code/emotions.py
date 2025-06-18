import cv2
import numpy as np
import dlib
from imutils import face_utils
import face_recognition
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from keras.optimizers import Adam

# Flag to determine if webcam or video file is used as source
USE_WEBCAM = False

# Path to the emotion detection model
emotion_model_path = '/Users/srilekhanampelli/Desktop/DeepLearning/CrowdAnalysis/Analysis/models/emotion_model.hdf5'

# Load the emotion labels
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# Load face detection model
face_detector = dlib.get_frontal_face_detector()

# Load the emotion model without compiling
emotion_classifier = load_model(emotion_model_path, compile=False)

# Define the learning rate
learning_rate = 0.0001

# Initialize the optimizer with the learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the emotion model with the defined optimizer
emotion_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting lists for calculating modes
emotion_window = []

# Start video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)  # Webcam source
else:
    cap = cv2.VideoCapture('/Users/srilekhanampelli/Desktop/DeepLearning/CrowdAnalysis/Analysis/test/testvdo.mp4')  # Video file source

# Loop to process video frames
while cap.isOpened(): 
    ret, frame = cap.read()

    # Convert image to grayscale and RGB
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = face_detector(rgb_frame)

    # Process each detected face
    for face_coordinates in faces:

        # Apply offsets to face coordinates
        x1, y1, x2, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_frame[y1:y2, x1:x2]
        
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        # Preprocess the face image
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        # Predict emotions
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        detected_emotion = emotion_labels[emotion_label_arg]
        emotion_window.append(detected_emotion)

        # Maintain a sliding window of emotions
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
            
        try:
            current_emotion = mode(emotion_window)
        except:
            continue

        # Determine color based on detected emotion
        if detected_emotion == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif detected_emotion == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif detected_emotion == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif detected_emotion == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        # Convert color to int and list
        color = color.astype(int)
        color = color.tolist()

        # Draw bounding box and text on the image
        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_frame, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_frame, current_emotion,
                  color, 0, -45, 1, 1)

    # Convert RGB image back to BGR
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Display the processed image
    cv2.imshow('window_frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
