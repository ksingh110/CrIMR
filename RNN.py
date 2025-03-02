import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = np.load("FILE_PATH.npz")
print("Keys in .npz file:", data.files)
encoded_sequences = data["your_key_here"]  


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
