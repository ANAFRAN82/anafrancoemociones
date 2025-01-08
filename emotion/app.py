import os
from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración de la carpeta de subida
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lista de puntos clave específicos
key_points = [70, 55, 285, 300, 33, 480, 133, 362, 473, 263, 4, 185, 0, 306, 1]

# Validar archivos
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para analizar la imagen
def analyze_face_and_emotion(image_path):
    try:
        # Inicializar MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

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

        # Preparar la visualización
        height, width = gray_image.shape
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(gray_image, cmap='gray')

        # Dibujar puntos clave específicos
        for face_landmarks in results.multi_face_landmarks:
            for idx in key_points:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                ax.plot(x, y, 'rx')

        # Guardar la imagen procesada
        buf = BytesIO()
        plt.axis('off')
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convertir la imagen a base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

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
        result_image = analyze_face_and_emotion(filepath)

        return jsonify({'success': True, 'image': result_image})

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
