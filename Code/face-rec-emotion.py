import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
from statistics import mode
import face_recognition
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from keras.optimizers import Adam
import time

# Flag for using webcam or video file as source
USE_WEBCAM = True  

# Path to the emotion detection model
emotion_model_path = '/Users/srilekhanampelli/Desktop/DeepLearning/CrowdAnalysis/Analysis/models/emotion_model.hdf5'

# Load the emotion labels
emotion_labels = get_labels('fer2013')

# Parameters for bounding box shape
frame_window = 10
emotion_offsets = (20, 40)

# Load face detection model
face_detector = dlib.get_frontal_face_detector()

# Load the emotion detection model
emotion_classifier = load_model(emotion_model_path, compile=False)

# Define optimizer
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)

# Compile the emotion model
emotion_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Get input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Lists for storing emotion data
emotion_window = []

# Load known face encodings and their names
obama_image = face_recognition.load_image_file("images/Obama.jpg")
obama_encoding = face_recognition.face_encodings(obama_image)[0]

trump_image = face_recognition.load_image_file("images/Trump.jpg")
trump_encoding = face_recognition.face_encodings(trump_image)[0]

modi_image = face_recognition.load_image_file("images/Modi.jpg")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

srilekha_image = face_recognition.load_image_file("images/Srilekha.jpeg")
srilekha_encoding = face_recognition.face_encodings(srilekha_image)[0]

rishi_image = face_recognition.load_image_file("images/Rishi.jpeg")
rishi_encoding = face_recognition.face_encodings(rishi_image)[0]

known_encodings = [
    obama_encoding,
    trump_encoding,
    modi_encoding,
    srilekha_encoding,
    rishi_encoding
]
known_names = [
    "Barack Obama",
    "Trump",
    "Modi",
    "Srilekha",
    "Rishi"
]

# Initialize variables
detected_face_locations = []
detected_face_encodings = []
detected_face_names = []

start_time = time.time()

# Flag for processing frames
process_frame = True

# Function to recognize faces
def recognize_faces(rgb_image, process_frame):
    print("Recognizing faces")
    if process_frame:
        detected_face_locations = face_recognition.face_locations(rgb_image)
        detected_face_encodings = face_recognition.face_encodings(rgb_image, detected_face_locations)
        detected_face_names = []
        for face_encoding in detected_face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            detected_face_names.append(name)
    else:
        detected_face_names = []
    return detected_face_names

# Open webcam or video file
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)  # Webcam source
else:
    cap = cv2.VideoCapture('/Users/srilekhanampelli/Desktop/DeepLearning/CrowdAnalysis/Analysis/test/dinner.mp4')  # Video file source

while cap.isOpened():  
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from camera")
        break

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb_image)
    detected_face_names = recognize_faces(rgb_image, process_frame)
    
    # Process detected faces
    for face_coordinates, name in zip(faces, detected_face_names):
        x1, y1, x2, y2 = face_coordinates.left(), face_coordinates.top(), face_coordinates.right(), face_coordinates.bottom()
        x1, x2, y1, y2 = apply_offsets((x1, y1, x2 - x1, y2 - y1), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        
        # Preprocess the face image
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        
        gray_face = preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        # Predict emotions
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        
        # Maintain a sliding window of emotions
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        
        # Set color based on detected emotion
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        
        color = color.astype(int)
        color = color.tolist()
        
        # Display name and emotion on bounding box
        if name == "Unknown":
            label = emotion_text
        else:
            label = f"{name} is {emotion_text}"
        
        draw_bounding_box((x1, y1, x2 - x1, y2 - y1), rgb_image, color)
        draw_text((x1, y1), rgb_image, label, color, 0, -45, 1, 1)
    
    process_frame = not process_frame  # Toggle the process flag
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 10:
        break

cap.release()
cv2.destroyAllWindows()

# Print the emotions of all detected faces
print("Emotions detected for each face:")
for name in set(detected_face_names):
    emotion = mode([emotion for face, emotion in zip(detected_face_names, emotion_window) if face == name])
    print(f"{name} is {emotion}")

print("\n")

