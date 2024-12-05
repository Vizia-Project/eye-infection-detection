from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = load_model('model.h5')

labels = ["Normal", "Stye", "Uveitis"]

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return '''
    <h1>Eye Diagnosis</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <label for="image">Upload an image:</label>
        <input type="file" name="image" id="image" required>
        <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file sent.'}), 400

    img = Image.open(file.stream)
    img = img.resize((128, 128))  
    img_array = np.array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)


    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = labels[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    return f'''
    <h1>Prediction Result</h1>
    <p>Predicted Class: <strong>{predicted_label}</strong></p>
    <p>Confidence: <strong>{confidence:.2f}%</strong></p>
    <a href="/">Back to Home</a>
    '''

if __name__ == '__main__':
    app.run(debug=True)
