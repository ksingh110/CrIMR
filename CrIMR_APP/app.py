from flask import Flask, request, jsonify, render_template, redirect, url_for
import tensorflow as tf
import os
import numpy as np
from APP_Preprocessing import onehotencoder  # Assuming this is the preprocessing function
from APP_Preprocessing import process
import pandas as pd
import umap 
import logging
logging.basicConfig(level=logging.DEBUG)
# Load your trained model
model = tf.keras.models.load_model("/Users/krishaysingh/Downloads/6000_5_if_new_best_model.keras") 

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Configure file upload folder (make sure to create this folder in your project directory)
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
def apply_umap_on_data(X):
    # Reshape the data to 2D for UMAP
    X_flat = X.reshape(X.shape[0], -1)
    
    # Initialize UMAP model
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    
    # Apply UMAP transformation
    X_umap = umap_model.fit_transform(X_flat)
    return X_umap
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
        # Ensure a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load data based on file type
        if file.filename.endswith('.npy'):
            input_data = np.load(file_path, allow_pickle=True)
        elif file.filename.endswith('.npz'):
            npz_data = np.load(file_path, allow_pickle=True)
            first_key = list(npz_data.files)[0]  # Get the first key
            input_data = npz_data[first_key]  # Extract the first stored array
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            input_data = df.to_numpy()
        else:
            return jsonify({'error': 'Unsupported file format. Please upload .npy, .npz, or .csv'}), 400
        
        X_umap = apply_umap_on_data(input_data)

        # Ensure input data has the correct shape (batch_size, timesteps, features)
         # Convert (samples, features) → (samples, 1, features)
        
        # Make prediction


        # Return the prediction as JSON
        prediction_prob = model.predict(X_umap).flatten()

        # Compute the probabilities for mutation and non-mutation
        mutation_prob = float(prediction_prob[0])  # Assuming the second class is "mutation"
        non_mutation_prob = 1 - mutation_prob  # Assuming only two classes: mutation and non-mutation
        # If probability > 0.5, it's DSPD, else it's non-DSPD
        prediction = 'DSPD' if mutation_prob > 0.5 else 'Non-DSPD'
        logging.debug(f"Predicted mutation_prob: {mutation_prob}, non_mutation_prob: {non_mutation_prob}")
        # Return the prediction and probabilities
        return jsonify({
            'prediction': prediction,
            'mutation_prob': mutation_prob,
            'non_mutation_prob': non_mutation_prob
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
