import os
import base64
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras

import cv2
from flask import Flask, render_template, request, redirect
from PIL import Image  # For image processing

import classifier  # Your model prediction code
import recomendation  # Your drug recommendation logic

app = Flask(__name__)

# Uploads folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
model = classifier.model

# Store last uploaded file path
last_file_path = None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global last_file_path

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        last_file_path = file_path

        # Get prediction
        predicted_class_name = classifier.predict_image_class(file_path)
        drugs = recomendation.suggest_drugs(predicted_class_name)

        # Process image for display
        image = Image.open(file.stream).convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        return render_template('index.html', suggestion=drugs, prediction=predicted_class_name, image_base64=img_base64)

    return redirect(request.url)


@app.route('/tumor', methods=['POST'])
def tumor():
    global last_file_path

    if last_file_path is None:
        return "No image uploaded for tumor detection", 400

    try:
        # Load and preprocess the image
        img_array = classifier.preprocess_image(last_file_path, target_size=(224, 224))

        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)  # Ensure shape (1, 224, 224, 3)

        # Display image
        img_display = cv2.cvtColor(cv2.imread(last_file_path), cv2.COLOR_BGR2RGB)
        img_display = cv2.resize(img_display, (224, 224))

        # Predict the class
        predicted_class_name = classifier.predict_image_class(last_file_path)
        class_names = ['Glioma', 'Meningioma', 'No-Tumor', 'Pituitary']
        has_tumor = predicted_class_name != 'No-Tumor'

        # Initialize `superimposed_base64` to avoid undefined variable error
        superimposed_base64 = None

        if has_tumor:
            # Grad-CAM heatmap generation
            grad_model = tf.keras.models.Model(
                inputs=[model.input],
                outputs=[model.get_layer("block5_conv3").output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_output, preds = grad_model(img_array)
                preds = preds[0]
                loss = preds[tf.argmax(preds)] if len(preds.shape) > 1 else preds[0]

            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_output = conv_output[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap).numpy()
            heatmap = np.maximum(heatmap, 0)

            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            heatmap_vis = cv2.resize(heatmap, (224, 224))
            heatmap_vis = np.uint8(255 * heatmap_vis)
            heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

            # Overlay heatmap on original image
            superimposed_img = cv2.addWeighted(img_display, 0.7, heatmap_vis, 0.3, 0)

            _, superimposed_encoded = cv2.imencode('.png', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
            superimposed_base64 = base64.b64encode(superimposed_encoded).decode('utf-8')

        # Encode the input image for display
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return render_template('results.html',
                               prediction=predicted_class_name,
                               has_tumor=has_tumor,
                               img_base64=img_base64,
                               superimposed_base64=superimposed_base64)

    except Exception as e:
        app.logger.error(f"Error in tumor detection: {str(e)}")
        return f"An error occurred during tumor analysis: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
