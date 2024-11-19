from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import mediapipe as mp
import os
import copy
import random

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

# Data Augmentation Functions
def augment_data(keyfacial_df, columns):
    # Volteo Horizontal
    keyfacial_df_copy = copy.copy(keyfacial_df)
    keyfacial_df_copy['Image'] = keyfacial_df['Image'].apply(lambda x: np.flip(x, axis=1))
    for i in range(len(columns)):
        if i % 2 == 0:  # Coordenadas X
            keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x))
    
    # Aumento de Brillo
    brightness_copy = copy.copy(keyfacial_df)
    brightness_copy['Image'] = brightness_copy['Image'].apply(
        lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255.0)
    )

    # Volteo Vertical
    vertical_flip_copy = copy.copy(keyfacial_df)
    vertical_flip_copy['Image'] = vertical_flip_copy['Image'].apply(lambda x: np.flip(x, axis=0))
    for i in range(len(columns)):
        if i % 2 == 1:  # Coordenadas Y
            vertical_flip_copy[columns[i]] = vertical_flip_copy[columns[i]].apply(lambda x: 96. - float(x))
    
    # Concatenar todas las transformaciones
    augmented_df = np.concatenate((keyfacial_df, keyfacial_df_copy, brightness_copy, vertical_flip_copy))
    return augmented_df

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
        image_base64 = analyze_face(image_path)
        return jsonify({'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
