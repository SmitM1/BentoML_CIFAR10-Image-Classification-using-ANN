import requests
import numpy as np
import tensorflow as tf

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Select a known test image and its label
test_image = x_test[0]  # First test image
test_label = y_test[0]  # Corresponding label

# Preprocess the image (normalize the pixel values)
test_image = test_image.astype('float32') / 255.0
print(f"test img shape: {test_image.shape}")
print(f"test image after normalize:{test_image}")

# Add batch dimension (as the model expects a batch of images)
test_image = np.expand_dims(test_image, axis=0)
print(f"test img shape: {test_image.shape}")
print(f"test image after expand dims:{test_image}")

# URL of your BentoML service
url = "http://localhost:3000/predict"

# Send a POST request with the test image
response = requests.post(url, json=test_image.tolist())

# Decode the response (assuming the model returns class probabilities)
predicted_probs = np.array(response.json())

# Get the predicted class index
predicted_class = np.argmax(predicted_probs, axis=1)[0]

# Print the predicted class and the true label
print(f"Predicted class: {predicted_class}")
print(f"True label: {test_label[0]}")

# Verify if the prediction is correct
if predicted_class == test_label[0]:
    print("The prediction is correct!")
else:
    print("The prediction is incorrect.")
