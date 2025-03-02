import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout

df = pd.read_parquet("FILE PATH")
# Krishay, create the dataset, and then shuffle and batch the dataset

def rnn_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),  
        Dropout(0.5), 
        Dense(1, activation='sigmoid')
    ])
        
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy']) 
    return model
if 'encoded_sequences' in locals():
    input_shape = (encoded_sequences.shape[1], encoded_sequences.shape[2])

    model = rnn_model(input_shape)
    model.summary()
else:
    print("Error: 'encoded_sequences' is not defined.")
