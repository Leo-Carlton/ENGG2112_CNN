import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Path to the image you want to classify
img_path = '/path/to/your/image/input_photo.jpg'  ## GEORGE CHANGE THIS TO ACTUAL DIRECTORY

# Load and preprocess the image
img = tf.keras.utils.load_img(img_path, target_size=(512, 512))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch

# Predict
prediction = base_model.predict(img_array)[0][0]  # Get the raw score

# Interpret the prediction
label = "Malignant" if prediction >= 0.5 else "Benign"
confidence = prediction if prediction >= 0.5 else 1 - prediction

# Output
print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
