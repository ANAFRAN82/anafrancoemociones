from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import mediapipe as mp
import os
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Carga el modelo de clasificación de expresiones faciales
model_path = "path_to_your_model.h5"
expression_model = load_model(model_path)

label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Tristeza', 3: 'Felicidad', 4: 'Sorpresa'}

def string2array(x):
    return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')

def resize_image(image):
    img = image.reshape(48, 48)
    return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)

def analyze_face(image_path):
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")
        
        landmarks = results.multi_face_landmarks
        if len(landmarks) == 0:
            raise Exception("No landmarks found for detected face")

        landmark = landmarks[0]
        num_landmarks = len(landmark.landmark)
        print(f"Number of landmarks detected: {num_landmarks}")

        key_points = [i for i in [70, 55, 285, 300, 33, 480, 133, 362, 473, 263, 4, 185, 0, 306, 17] if i < num_landmarks]

        height, width = gray_image.shape
        
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(gray_image, cmap='gray')

        for point_idx in key_points:
            landmark_point = landmark.landmark[point_idx]
            x = int(landmark_point.x * width)
            y = int(landmark_point.y * height)
            plt.plot(x, y, 'rx')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

def analyze_face_with_expression(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise Exception("Could not load image")

        # Redimensionar la imagen a 48x48 y luego a 96x96
        resized_image = cv2.resize(image, (48, 48))
        resized_image = resize_image(resized_image)

        # Normalización
        normalized_image = resized_image / 255.0
        normalized_image = normalized_image.reshape(1, 96, 96, 1)

        # Predicción
        predictions = expression_model.predict(normalized_image)
        emotion_index = np.argmax(predictions)
        emotion_label = label_to_text[emotion_index]

        return emotion_label

    except Exception as e:
        print(f"Error in analyze_face_with_expression: {str(e)}")
        raise

@app.route('/')
def index():
    images = os.listdir(UPLOAD_FOLDER)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files and 'existing_file' not in request.form:
        return jsonify({'error': 'No file uploaded'}), 400

    image_path = ''
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
    else:
        existing_file = request.form['existing_file']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_file)

    try:
        emotion_label = analyze_face_with_expression(image_path)
        image_base64 = analyze_face(image_path)
        return jsonify({'emotion': emotion_label, 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
