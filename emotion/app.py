import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración de la carpeta de subida
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Tamaño máximo de archivo: 16MB

# Crear el directorio de subida si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para validar archivos permitidos
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para analizar la imagen y detectar emociones
def analyze_face_and_emotion(image_path):
    try:
        # Inicializar MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        mp_face_detection = mp.solutions.face_detection
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertir a RGB y escala de grises
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar emociones con MediaPipe (asumiendo detección de emociones básica)
        detection_results = face_detection.process(rgb_image)
        emotion_text = "Sin emoción detectada"
        if detection_results.detections:
            emotion_text = "¡Cara detectada! (Emoción no implementada detalladamente)"

        # Detectar puntos clave faciales
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No se detectó ninguna cara en la imagen")

        # Preparar la visualización
        height, width = gray_image.shape
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(gray_image, cmap='gray')

        # Dibujar puntos clave faciales
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                ax.plot(x, y, 'rx')

        # Mostrar el texto de la emoción en la imagen
        ax.text(10, 10, emotion_text, color='yellow', fontsize=12, backgroundcolor='black')

        # Guardar la imagen procesada
        buf = BytesIO()
        plt.axis('off')
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convertir la imagen a base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64, emotion_text

    except Exception as e:
        print(f"Error en analyze_face_and_emotion: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Manejar la subida de archivos
        if 'file' not in request.files:
            return jsonify({'error': 'No se envió ningún archivo'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de archivo no permitido'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analizar la imagen
        result_image, emotion = analyze_face_and_emotion(filepath)

        return jsonify({'success': True, 'image': result_image, 'emotion': emotion})

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
