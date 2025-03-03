import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
data = np.load("E:/datasets/processeddata/encoded_mutation_sequences.npz")

if len(data.files) == 0:
    raise ValueError("The file does not contain any arrays and does not work.")

encoded_sequences = None  
input_shape = None  

# Find a valid sequence
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
    raise ValueError("No valid encoded sequences found in the file.")

# Define the RNN model
def rnn_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and summarize the model
model = rnn_model(input_shape)
model.summary()
