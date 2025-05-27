import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import json
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = '../models/brain_tumor_model.h5'
app.config['CLASS_NAMES_PATH'] = '../models/class_names.json'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model and class names
def load_model_and_classes():
    try:
        model = load_model(app.config['MODEL_PATH'])
        with open(app.config['CLASS_NAMES_PATH'], 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        # Default class names if file not found
        return None, ["glioma", "meningioma", "notumor", "pituitary"]

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess the image for prediction
def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Get tumor information
def get_tumor_info(tumor_type):
    tumor_info = {
        "glioma": {
            "description": "Gliomas are tumors that originate in the glial cells of the brain. These cells support and protect neurons.",
            "symptoms": ["Headaches", "Seizures", "Memory loss", "Physical weakness", "Speech difficulties"],
            "treatment": "Treatment typically involves surgery, radiation therapy, and chemotherapy. The specific approach depends on the grade and location of the tumor.",
            "prognosis": "Prognosis varies widely depending on the grade of the tumor, with higher-grade gliomas having a less favorable outlook."
        },
        "meningioma": {
            "description": "Meningiomas develop in the meninges, the layers of tissue that cover the brain and spinal cord.",
            "symptoms": ["Headaches", "Vision problems", "Hearing loss", "Seizures", "Memory problems"],
            "treatment": "Small, slow-growing meningiomas may only require monitoring. Larger or symptomatic tumors typically require surgical removal, sometimes followed by radiation therapy.",
            "prognosis": "Most meningiomas are benign (non-cancerous) and have a favorable prognosis after treatment."
        },
        "notumor": {
            "description": "No tumor detected in the MRI scan.",
            "symptoms": ["N/A"],
            "treatment": "No tumor-specific treatment required. If symptoms persist, further medical evaluation is recommended.",
            "prognosis": "Excellent in the absence of a brain tumor. However, other neurological conditions should be considered if symptoms are present."
        },
        "pituitary": {
            "description": "Pituitary tumors develop in the pituitary gland, which is located at the base of the brain and regulates many hormonal functions.",
            "symptoms": ["Headaches", "Vision problems", "Hormonal imbalances", "Fatigue", "Unexplained weight changes"],
            "treatment": "Treatment options include medication to control hormone production, surgery, and radiation therapy.",
            "prognosis": "Most pituitary tumors are benign and have a good prognosis with appropriate treatment."
        }
    }
    return tumor_info.get(tumor_type, {"description": "Information not available", "symptoms": [], "treatment": "", "prognosis": ""})

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Load model and class names
        model, class_names = load_model_and_classes()
        if model is None:
            return jsonify({'error': 'Model not found. Please train the model first.'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image and make prediction
        processed_image = preprocess_image(file_path)
        predictions = model.predict(processed_image)[0]
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[predicted_class_index])
        
        # Get information about the tumor type
        tumor_info = get_tumor_info(predicted_class.lower())
        
        # Prepare the response
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'file_path': os.path.join('uploads', filename),
            'tumor_info': tumor_info,
            'class_probabilities': {class_name: float(prob) for class_name, prob in zip(class_names, predictions)}
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
