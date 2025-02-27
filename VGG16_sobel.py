from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma

# Paths to the image directories
yes_image_dir = r'train\with_cancer'
no_image_dir =r'train\no_cancer 

# Apply preprocessing techniques
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224 for VGG16
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Non-Local Means (NLM) filtering for noise reduction
    sigma_est = np.mean(estimate_sigma(gray))  # Removed 'multichannel=False'
    nlm_filtered = denoise_nl_means(gray, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)
    
    # Convert back to uint8
    nlm_filtered = (nlm_filtered * 255).astype(np.uint8)
    
    # Apply Adaptive Histogram Equalization (AHE) to improve image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ahe = clahe.apply(nlm_filtered)
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(nlm_filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(nlm_filtered, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))

    # Apply edge detection using Canny
    edges = cv2.Canny(nlm_filtered, 100, 200)
    
    # Stack the processed images to create a 3-channel image
    processed_img = np.stack([ahe, sobel_combined, edges], axis=-1)
    return processed_img

# Load and preprocess images from a directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        processed_img = preprocess_image(img_path)
        images.append(processed_img)
    return np.array(images)

# Load images from both directories
yes_images = load_images_from_directory(yes_image_dir)
no_images = load_images_from_directory(no_image_dir)

# Combine images and create labels
images = np.concatenate((yes_images, no_images), axis=0)
y = np.concatenate((np.ones(len(yes_images)), np.zeros(len(no_images))), axis=0)

# Use VGG16 to extract features
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_cnn_features(images):
    cnn_features = []
    for img in images:
        img = preprocess_input(img.astype(np.float32))  # Preprocess input for VGG16
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        features = base_model.predict(img)  # Extract features using the VGG16 model
        cnn_features.append(features.flatten())  # Flatten the features
    return np.array(cnn_features)

# Extract CNN features for all images
X_cnn = extract_cnn_features(images)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),  # Flatten the output from VGG16 to a 1D vector
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('brain_cancer_cnn_model_preprocessed.h5')

# Plot training accuracy and loss
plt.figure()
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the test accuracy
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
