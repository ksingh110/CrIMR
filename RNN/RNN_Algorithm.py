import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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

num_sample = encoded_sequences.shape[0]
y = np.random.randomint(0, 2, size=num_sample)

x_train, x_test, y_train, y_test =  train_test_split(encoded_sequences, y , random_state=42, test_size=0.2) 

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

checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, mode="min")
csv_log = CSVLogger("training_log.csv", append=True)

rnn_fit = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_test, y_test), callbacks=[csv_log, checkpoint])

accuracy, lose = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

predictions = model.predict(x_test).flatten()
print("The Sample predictions are: ", predictions)

save_path = "prediction_plots/"
os.makedirs(save_path, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities")
plt.savefig(save_path + "histogram_predictions.png")  # Save plot
plt.show()

