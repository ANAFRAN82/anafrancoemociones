import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración del directorio de carga
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB máximo

# Crear directorio si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo de detección de emociones
emotion_model = tf.keras.models.load_model('path_to_your_model.h5')  # Reemplaza con la ruta de tu modelo
label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Tristeza', 3: 'Felicidad', 4: 'Sorpresa'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_face(image_path):
    try:
        # Inicializar MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertir a RGB y escala de grises
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar puntos faciales
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No se detectó ninguna cara en la imagen")

        # Recortar la cara para análisis de emociones
        height, width = rgb_image.shape[:2]
        key_landmark = results.multi_face_landmarks[0].landmark[4]  # Usa un punto clave como referencia
        x_min = int(max(key_landmark.x * width - 50, 0))
        y_min = int(max(key_landmark.y * height - 50, 0))
        x_max = int(min(key_landmark.x * width + 50, width))
        y_max = int(min(key_landmark.y * height + 50, height))
        roi = rgb_image[y_min:y_max, x_min:x_max]

        # Preprocesar la ROI para el modelo
        roi_resized = cv2.resize(roi, (48, 48))  # Tamaño esperado por el modelo
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
        roi_normalized = roi_gray / 255.0  # Normalización
        roi_reshaped = np.expand_dims(roi_normalized, axis=(0, -1))  # Expande dimensiones para el modelo

        # Predecir emoción
        emotion_prediction = emotion_model.predict(roi_reshaped)
        emotion_label = np.argmax(emotion_prediction)
        emotion_text = label_to_text[emotion_label]

        # Dibujar la emoción detectada
        plt.imshow(gray_image, cmap='gray')
        plt.title(f'Emoción: {emotion_text}')
        plt.axis('off')

        # Guardar imagen procesada
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64, emotion_text

    except Exception as e:
        print(f"Error en analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    images = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_image, emotion = analyze_face(filepath)

            return jsonify({
                'success': True,
                'image': result_image,
                'emotion': emotion
            })

        return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
