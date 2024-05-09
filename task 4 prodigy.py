import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load images and labels from a directory
def load_dataset(directory):
    images = []
    labels = []
    gesture_names = os.listdir(directory)
    for gesture_id, gesture_name in enumerate(gesture_names):
        gesture_dir = os.path.join(directory, gesture_name)
        for image_name in os.listdir(gesture_dir):
            image_path = os.path.join(gesture_dir, image_name)
            image = cv2.imread(image_path)
            # Preprocess image (resize, normalize, etc.)
            # Add preprocessed image to images list
            images.append(image)
            # Add label to labels list
            labels.append(gesture_id)
    return np.array(images), np.array(labels)

# Define the path to your dataset directory
dataset_directory = r"C:\Users\Kavya Bhatt\OneDrive\Desktop\TASK 4 DATASET"

# Load the dataset
X, y = load_dataset(dataset_directory)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
