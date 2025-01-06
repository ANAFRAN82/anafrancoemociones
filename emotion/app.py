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
from deepface import DeepFace  # Import DeepFace for emotion detection

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_face(image_path):
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect facial landmarks
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")

        # Detect emotions using DeepFace
        emotion_analysis = DeepFace.analyze(image_path, actions=['emotion'])

        # Get the dominant emotion
        dominant_emotion = emotion_analysis[0]['dominant_emotion']
        emotion_label = { 
            'angry': 'Ira',
            'disgust': 'Odio',
            'sad': 'Tristeza',
            'happy': 'Felicidad',
            'surprise': 'Sorpresa'
        }

        detected_emotion = emotion_label.get(dominant_emotion, "Desconocida")

        # Prepare transformations
        transformations = [
            ("Original", gray_image),
            ("Horizontally Flipped", cv2.flip(gray_image, 1)),
            ("Brightened", cv2.convertScaleAbs(gray_image, alpha=1.2, beta=50)),
            ("Upside Down", cv2.flip(gray_image, 0))
        ]

        height, width = gray_image.shape

        # Initialize figure for displaying images
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for ax, (title, img) in zip(axes, transformations):
            ax.imshow(img, cmap='gray')
            num_landmarks = len(results.multi_face_landmarks[0].landmark)
            for point_idx in range(0, 468):  # Iterate through the number of facial landmarks
                if point_idx < num_landmarks:  # Check if the index is valid
                    landmark = results.multi_face_landmarks[0].landmark[point_idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    # Adjust key points according to transformations
                    if title == "Horizontally Flipped":
                        x = width - x
                    elif title == "Upside Down":
                        y = height - y
                    ax.plot(x, y, 'rx')
            ax.set_title(f"{title} - Emoción: {detected_emotion}")
            ax.axis('off')

        # Save processed image to memory
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64, detected_emotion

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    # Get list of images in upload folder
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if we're analyzing an existing file
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'File not found: {filename}'}), 404
            
        # Check if we're uploading a new file
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        
        else:
            return jsonify({'error': 'No file provided'}), 400

        # Analyze the image
        result_image, detected_emotion = analyze_face(filepath)
        
        return jsonify({
            'success': True,
            'image': result_image,
            'emotion': detected_emotion
        })

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
