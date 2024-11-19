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
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

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

        # Aplicamos transformaciones
        images_data = []

        # Imagen volteada horizontalmente
        flipped_horizontal = cv2.flip(image, 1)
        images_data.append(save_image_with_points(flipped_horizontal, "flipped_horizontal.png", face_mesh, height, width))

        # Imagen con brillo aumentado
        brightness_factor = np.random.uniform(1.5, 2.0)
        brighter_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        images_data.append(save_image_with_points(brighter_image, "brighter.png", face_mesh, height, width))

        # Imagen con brillo aumentado y volteada verticalmente
        flipped_vertical = cv2.flip(brighter_image, 0)
        images_data.append(save_image_with_points(flipped_vertical, "flipped_vertical.png", face_mesh, height, width))

        # Imagen original con puntos
        images_data.append(save_image_with_points(image, "original.png", face_mesh, height, width))

        return images_data

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise

def save_image_with_points(image, filename, face_mesh, height, width):
    """
    Genera y guarda una imagen con los puntos clave detectados superpuestos.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesamos la imagen con MediaPipe
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        raise Exception(f"No face detected in {filename}")

    # Detectamos puntos clave específicos
    landmarks = results.multi_face_landmarks[0].landmark
    key_points = [i for i in [70, 55, 285, 300, 33, 480, 133, 362, 473, 263, 4, 185, 0, 306, 17] if i < len(landmarks)]
    points = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in key_points]

    # Dibujamos los puntos clave en la imagen
    plt.clf()
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(gray_image, cmap='gray')

    for x, y in points:
        plt.plot(x, y, 'rx')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Guardamos la imagen procesada en PROCESSED_FOLDER
    output_path = os.path.join(PROCESSED_FOLDER, filename)
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())

    plt.close(fig)
    return output_path

@app.route('/')
def index():
    images = os.listdir(UPLOAD_FOLDER)
    processed_images = os.listdir(PROCESSED_FOLDER)
    return render_template('index.html', images=images, processed_images=processed_images)

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
        images_data = analyze_face(image_path)
        return jsonify({'processed_images': images_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
