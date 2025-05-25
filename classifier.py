import numpy as np
from keras.preprocessing import image  # For standalone Keras
from keras.models import load_model

# Load the trained model
model = load_model('/Users/muhammadumair/Desktop/model/vgg16_brain_mri_classifier.h5')

# Function to preprocess the image
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of an image
def predict_image_class(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path, target_size=(224, 224))

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map the predicted class index to the class name
    class_names = ['Glioma', 'Meningioma', 'No-Tumor', 'Pituitary']  # Replace with your class names
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name