from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the original model
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    # Resize the image to 150x150 (as expected by the model)
    img = image.resize((150, 150))
    # Convert to grayscale
    img = img.convert('L')  # 'L' mode converts to grayscale
    # Convert to numpy array
    img_array = np.array(img)
    
    # Ensure the array has the channel dimension (shape should be (150, 150, 1))
    img_array = np.expand_dims(img_array, axis=-1)
    
    # Normalize the image (scale pixel values to [0, 1] as done during training)
    img_array = img_array / 255.0
    
    # Add batch dimension for prediction
    img_array = np.expand_dims(img_array, axis=0)
    
    # Print the shape after preprocessing
    print("Shape after preprocessing:", img_array.shape)
    
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    uploaded_image = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        pred = model.predict(processed_image)
        print("Raw prediction value:", pred[0][0])  # Print raw prediction to debug
        # Adjust threshold based on model performance (model tends to predict Pneumonia more)
        threshold = 0.7  # Increase threshold since model biases towards Pneumonia
        prediction = "Pneumonia Detected" if pred[0][0] > threshold else "Normal (No Pneumonia)"
        confidence = round(pred[0][0] * 100, 2) if pred[0][0] > threshold else round((1 - pred[0][0]) * 100, 2)
        uploaded_image = file.filename

    return render_template('index.html', prediction=prediction, confidence=confidence, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)