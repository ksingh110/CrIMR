from flask import Flask, request, jsonify, render_template, redirect, url_for
import tensorflow as tf
import os
import numpy as np
from APP_Preprocessing import onehotencoder  # Assuming this is the preprocessing function
from APP_Preprocessing import process
import pandas as pd
import umap 

# Load your trained model
model = tf.keras.models.load_model("/Users/krishaysingh/Downloads/6000_5_if_new_best_model.keras") 

app = Flask(__name__)

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
        prediction = model.predict(X_umap).flatten()

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
