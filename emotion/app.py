from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import mediapipe as mp
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def analyze_face(image_path):
    try:
        # Inicializamos MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Cargamos la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        # Dimensiones originales de la imagen
        height, width = image.shape[:2]

        # Procesamos la imagen original con MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")

        landmarks = results.multi_face_landmarks[0].landmark

        # Lista de puntos clave específicos
        key_points = [70, 55, 285, 300, 33, 480, 133, 362, 473, 263, 4, 185, 0, 306, 17]

        # Coordenadas iniciales
        points = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in key_points]

        # Giramos horizontalmente
        flipped_image_h = cv2.flip(image, 1)
        points_flipped_h = [(width - x, y) for x, y in points]  # Ajustamos las coordenadas

        # Aumentamos el brillo aleatoriamente
        brightness_factor = np.random.uniform(1.5, 2.0)
        brighter_image = np.clip(flipped_image_h * brightness_factor, 0, 255).astype(np.uint8)

        # Volteamos verticalmente
        flipped_image_v = cv2.flip(brighter_image, 0)
        points_flipped_v = [(x, height - y) for x, y in points_flipped_h]  # Ajustamos las coordenadas

        # Convertimos a escala de grises para la visualización final
        gray_image = cv2.cvtColor(flipped_image_v, cv2.COLOR_BGR2GRAY)

        # Dibujamos la imagen procesada y los puntos
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(gray_image, cmap='gray')

        for x, y in points_flipped_v:
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
