from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from APP_Preprocessing import onehotencoder  # Assuming this is the preprocessing function
from flask import render_template
from APP_Preprocessing import process


# Load your trained model
model = tf.keras.models.load_model("E:/my_models/750_if_new_best_model.keras")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get file from request
        file = request.files['file']
        
        # Save the file temporarily
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the file (assuming it's a FASTA sequence file)
        with open(file_path, 'r') as f:
            fasta_sequence = f.read()

        # Preprocess the sequence
        preprocessed_data = onehotencoder(fasta_sequence, max_length=102500)
        

        # Reshape for the model if necessary
        processed_data = process(preprocessed_data)

        # Make prediction
        prediction = model.predict(processed_data)

        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
