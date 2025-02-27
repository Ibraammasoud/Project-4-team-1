
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
from skimage import transform

# Paths to the image directories
yes_image_dir = r'C:\Users\faulk\OneDrive\Desktop\project4\train\with_cancer'  # Path to positive (tumor) images
no_image_dir = r'C:\Users\faulk\OneDrive\Desktop\project4\train\no_cancer'    # Path to negative (no tumor) images

# Load images from a directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=(64, 64))  # Resize images to 64x64
        img_array = image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Load images from both directories
yes_images = load_images_from_directory(yes_image_dir)
no_images = load_images_from_directory(no_image_dir)

# Combine the positive and negative images and create labels
images = np.concatenate((yes_images, no_images), axis=0)
y = np.concatenate((np.ones(len(yes_images)), np.zeros(len(no_images))), axis=0)

# Use VGG16 to extract features
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

def extract_cnn_features(images):
    cnn_features = []
    for img in images:
        img = transform.resize(img, (64, 64, 3))  # Ensure the image is 64x64 and 3 channels
        img = preprocess_input(img)  # Preprocess input for VGG16
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        features = base_model.predict(img)  # Extract features using the VGG16 model
        cnn_features.append(features.flatten())  # Flatten the features
    return np.array(cnn_features)

# Extract CNN features for all images
X_cnn = extract_cnn_features(images)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# Build the CNN model (note: input shape should match VGG16 output size, not (64, 64, 1))
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
model.save('brain_cancer_cnn_model.h5')

# Optionally, plot training accuracy and loss
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot training loss and validation loss
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