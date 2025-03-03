import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = np.load("File Path.npz", mmap_mode='r')
if len(data.files) == 0:
    raise ValueError ("The file does not contain any arrays and does not work")

for key in data.files:
    encoded_sequences = data[key]
    if encoded_sequences.ndim == 3:  
        input_shape = (encoded_sequences.shape[1], encoded_sequences.shape[2])
    else:
        print(f"Skipping {key}: Unexpected shape {encoded_sequences.shape}")
        continue



def rnn_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),  
        Dropout(0.5),  
        Dense(1, activation='sigmoid')
    ])
        
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (encoded_sequences.shape[1], encoded_sequences.shape[2])

model = rnn_model(input_shape)
model.summary()
