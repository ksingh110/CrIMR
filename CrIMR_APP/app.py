from flask import Flask, request, jsonify, render_template, redirect, url_for
import tensorflow as tf
import os
import numpy as np
from APP_Preprocessing import onehotencoder  # Assuming this is the preprocessing function
from APP_Preprocessing import process

# Load your trained model
model = tf.keras.models.load_model("/Users/krishaysingh/Downloads/6000_5_if_new_best_model.keras") 

app = Flask(__name__)

# Configure file upload folder (make sure to create this folder in your project directory)
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/real-thing', methods=['GET'])
def real_thing():
    # The real prediction page (you can put a more comprehensive prediction form here)
    return render_template('real_thing.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains the file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if a file is selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the file (assuming it's a FASTA sequence file)
        with open(file_path, 'r', errors="ignore")  as f:  # Added encoding='latin-1'
            fasta_sequence = f.read()

        # Preprocess the sequence
        processed_data = onehotencoder(fasta_sequence, max_length=13000)
        processed_data = np.expand_dims(processed_data, axis=1)# Flatten it before passing to model
        preprocessed = process(processed_data)

        # Make prediction
        prediction = model.predict(fasta_sequence).flatten()

        # Return prediction as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
