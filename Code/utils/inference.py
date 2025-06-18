import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    """Load image from file and convert to numpy array.
    
    Args:
        image_path (str): Path to the image file.
        grayscale (bool, optional): Whether to convert image to grayscale (default is False).
        target_size (tuple, optional): Size to resize the image (width, height).
    
    Returns:
        numpy.ndarray: Image array.
    """
    pil_image = image.load_img(image_path, grayscale, target_size)  # Load image using Keras
    return image.img_to_array(pil_image)  # Convert PIL image to numpy array

def load_detection_model(model_path):
    """Load face detection model.
    
    Args:
        model_path (str): Path to the face detection model file.
    
    Returns:
        cv2.CascadeClassifier: Face detection model.
    """
    detection_model = cv2.CascadeClassifier(model_path)  # Load face detection model using OpenCV
    return detection_model

def detect_faces(detection_model, gray_image_array):
    """Detect faces in an image using the specified detection model.
    
    Args:
        detection_model (cv2.CascadeClassifier): Face detection model.
        gray_image_array (numpy.ndarray): Grayscale image array.
    
    Returns:
        list of tuples: List of face coordinates (x, y, width, height).
    """
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)  # Detect faces using OpenCV

def draw_bounding_box(face_coordinates, image_array, color):
    """Draw bounding box around the detected face.
    
    Args:
        face_coordinates (tuple): Coordinates of the detected face (x, y, width, height).
        image_array (numpy.ndarray): Image array.
        color (tuple): Color of the bounding box (B, G, R).
    """
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)  # Draw rectangle around face

def apply_offsets(face_coordinates, offsets):
    """Apply offsets to face coordinates.
    
    Args:
        face_coordinates (tuple): Coordinates of the detected face (x, y, width, height).
        offsets (tuple): Offset values for x and y (x_offset, y_offset).
    
    Returns:
        tuple: Adjusted face coordinates (x1, x2, y1, y2).
    """
    x, y, width, height = face_coordinates
    x_offset, y_offset = offsets
    return (x - x_offset, x + width + x_offset, y - y_offset, y + height + y_offset)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=0.5, thickness=1):
    """Draw text on the image.
    
    Args:
        coordinates (tuple): Coordinates for placing text (x, y).
        image_array (numpy.ndarray): Image array.
        text (str): Text to be displayed.
        color (tuple): Text color (B, G, R).
        x_offset (int, optional): Horizontal offset for the text position (default is 0).
        y_offset (int, optional): Vertical offset for the text position (default is 0).
        font_scale (float, optional): Font scale factor (default is 0.5).
        thickness (int, optional): Thickness of the text (default is 1).
    """
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)  # Draw text on image

def get_colors(num_classes):
    """Generate an array of colors for visualizing different classes.
    
    Args:
        num_classes (int): Number of classes.
    
    Returns:
        numpy.ndarray: Array of colors.
    """
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()  # Generate colors using HSV colormap
    colors = np.asarray(colors) * 255  # Convert colors to BGR format
    return colors
