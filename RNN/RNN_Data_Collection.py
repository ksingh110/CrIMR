import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from sklearn.ensemble import IsolationForest

checkpoint_path = "E:/my_models/750_if_new_best_model.h5"
save_path = "E:/my_plots/750_if_new_prediction_plots/"
mutated_data = np.load("E:\datasets\processeddata\MUTATION_DATA_TRAINING_750.npz", allow_pickle=True, mmap_mode='r')
nonmutated_data = np.load("E:\\datasets\\processeddata\\AUGMENTED_DATA_TRAINING_750.npz", allow_pickle=True, mmap_mode='r')
csv_file = "cry1realvariations (1).csv"  # Assuming this CSV contains mutation data for anomaly detection

# Load the pre-trained model (replace with the actual path to your model)
model = tf.keras.models.load_model('E:/my_models/750_if_new_best_model.h5')
print(f"Model loaded successfully: {model.name}")

# Function to load sequences and their labels
def load_sequences(data, label):
    encoded_sequences = None
    input_shape = None
    
    for key in data.files:
        temp_sequences = data[key]
        print(f"Checking key '{key}': shape {temp_sequences.shape}")  # Debugging

        if temp_sequences.ndim == 2:  # If 2D, reshape to 3D for LSTM
            temp_sequences = np.expand_dims(temp_sequences, axis=1)  # (samples, 1, features)

        if temp_sequences.ndim == 3:
            encoded_sequences = temp_sequences
            input_shape = (encoded_sequences.shape[1], encoded_sequences.shape[2])
            print(f"Using key '{key}' with reshaped shape {encoded_sequences.shape}")
            break
        else:
            print(f"Skipping '{key}': Unexpected shape {temp_sequences.shape}")

    if encoded_sequences is None:
        raise ValueError(f"No valid encoded sequences found in {label} file.")

    return encoded_sequences, input_shape

# Load mutated and non-mutated sequences
mutated_sequences, input_shape = load_sequences(mutated_data, "mutated")
nonmutated_sequences, _ = load_sequences(nonmutated_data, "nonmutated")

mutated_labels = np.ones(mutated_sequences.shape[0])  # 1 for mutated
nonmutated_labels = np.zeros(nonmutated_sequences.shape[0])  # 0 for non-mutated

# Anomaly Isolation Function - Use Isolation Forest to score anomalies
def get_anomaly_scores(X):
    # Fit IsolationForest model
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X.reshape(X.shape[0], -1))  # Flatten sequences to 2D (samples, features)

    # Convert predictions: 1 for normal, -1 for anomaly
    return (anomaly_scores == -1).astype(int)  # 1 for anomaly, 0 for normal

# Get anomaly scores for mutated sequences (for example)
anomaly_scores = get_anomaly_scores(mutated_sequences)

# Incorporate anomaly scores into the data
anomaly_scores_reshaped = anomaly_scores[:mutated_sequences.shape[0]].reshape(-1, 1)
anomaly_scores_reshaped = np.expand_dims(anomaly_scores_reshaped, axis=-1)  # shape (1000, 1, 1)

# Update the last feature of each mutated sequence with the anomaly score
mutated_sequences_with_anomalies = mutated_sequences.copy()
mutated_sequences_with_anomalies[:, :, -1:] = anomaly_scores_reshaped

# Concatenate mutated and non-mutated sequences
X_with_anomalies = np.concatenate([mutated_sequences_with_anomalies, nonmutated_sequences], axis=0)
y = np.concatenate([mutated_labels, nonmutated_labels], axis=0)

# Split data into train/test
x_train, x_test, y_train, y_test = train_test_split(X_with_anomalies, y, random_state=42, test_size=0.2, stratify=y)

# Ensure the model input is initialized
def get_layer_activations(model, X):
    activation_outputs = []
    layer_names = []

    # Initialize model input with dummy data
    _ = model(X)  # Dummy forward pass to initialize the model

    # Loop over layers and create models to fetch activations
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.Dense):
            layer_names.append(layer.name)
            temp_model = Model(inputs=model.input, outputs=layer.output)
            activation_outputs.append(temp_model(X))

    return activation_outputs, layer_names

# Get activations after making sure the model input is initialized
activation_outputs, layer_names = get_layer_activations(model, x_test)

# Plot ReLU and Sigmoid activation functions
plt.figure(figsize=(12, 8))

# Plot 1: ReLU activation function
x_values = np.linspace(-10, 10, 1000)
relu_values = np.maximum(0, x_values)

plt.subplot(2, 2, 1)
plt.plot(x_values, relu_values)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# Plot 2: Sigmoid activation function
sigmoid_values = 1 / (1 + np.exp(-x_values))

plt.subplot(2, 2, 2)
plt.plot(x_values, sigmoid_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)

# Plot activations of specific layers (ReLU, Dense layers)
plt.figure(figsize=(15, 12))

for i, (act, layer_name) in enumerate(zip(activation_outputs, layer_names)):
    if isinstance(model.layers[i], tf.keras.layers.ReLU):
        plt.subplot(len(activation_outputs) // 3 + 1, 3, i + 1)
        plt.plot(act[0, :])  # Assuming the output shape is (1, units)
        plt.title(f"ReLU Activation in Layer: {layer_name}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Activation Value")
        plt.grid(True)

plt.tight_layout()
plt.savefig('activation_functions_analysis_test.png')
plt.show()

# Additional plotting or analysis can be done here if needed
