from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

model = tf.keras.models.load_model('mode.h5')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file uploaded', confidence=None, image_path=None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No file selected', confidence=None, image_path=None)
        
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                print(f"Image saved at: {file_path}")  
            except Exception as e:
                return render_template('index.html', prediction=f'Error saving image: {str(e)}', confidence=None, image_path=None)
            
            try:
                img_array = preprocess_image(file_path)
                pred = model.predict(img_array)
                predicted_class = np.argmax(pred, axis=1)[0]
                confidence_score = float(np.max(pred) * 100)
                labels = {0: 'Normal', 1: 'Pneumonia'}
                prediction = labels[predicted_class]
                confidence = round(confidence_score, 2)
                image_path = os.path.join('uploads', filename)
                print(f"Image path for display: static/{image_path}")  
            except Exception as e:
                return render_template('index.html', prediction=f'Error: {str(e)}', confidence=None, image_path=None)
    
    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)