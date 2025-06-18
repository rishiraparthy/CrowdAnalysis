import numpy as np
import cv2 as cv

def preprocess_input(x, v2=True):
    """Preprocess input images.
    
    Args:
        x (numpy.ndarray): Input image array.
        v2 (bool, optional): Whether to apply V2 preprocessing (default is True).
    
    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    x = x.astype('float32')  # Convert image to float32
    x = x / 255.0  # Normalize pixel values to range [0, 1]
    if v2:
        x = x - 0.5  # Shift values to range [-0.5, 0.5]
        x = x * 2.0  # Scale values to range [-1, 1]
    return x

def _imread(image_name):
    """Read image from file.
    
    Args:
        image_name (str): Path to the image file.
    
    Returns:
        numpy.ndarray: Image array.
    """
    return cv.imread(image_name)  # Read image using OpenCV

def _imresize(image_array, size):
    """Resize image array.
    
    Args:
        image_array (numpy.ndarray): Input image array.
        size (tuple): New size (width, height) of the image.
    
    Returns:
        numpy.ndarray: Resized image array.
    """
    return cv.resize(image_array, size)  # Resize image using OpenCV

def to_categorical(integer_classes, num_classes=2):
    """Convert integer class labels to categorical one-hot encoding.
    
    Args:
        integer_classes (numpy.ndarray): Array of integer class labels.
        num_classes (int, optional): Number of classes (default is 2).
    
    Returns:
        numpy.ndarray: Categorical one-hot encoding of class labels.
    """
    integer_classes = np.asarray(integer_classes, dtype='int')  # Convert to numpy array
    num_samples = integer_classes.shape[0]  # Number of samples
    categorical = np.zeros((num_samples, num_classes))  # Initialize categorical array
    categorical[np.arange(num_samples), integer_classes] = 1  # Set corresponding index to 1
    return categorical  # Return one-hot encoded labels
