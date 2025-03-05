import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split

data = np.load("E:/datasets/processeddata/encoded_mutation_sequences.npz")

if len(data.files) == 0:
    raise ValueError("The file does not contain any arrays and does not work.")

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
    raise ValueError("No valid encoded sequences found in the file.")

y = np.random.randint(0, 2, size=encoded_sequences.shape[0])  # Random labels

x_train, x_test, y_train, y_test = train_test_split(encoded_sequences, y, random_state=42, test_size=0.2)

def rnn_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = rnn_model(input_shape)

checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, mode="min")
csv_log = CSVLogger("training_log.csv", append=True)

rnn_fit = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_test, y_test), callbacks=[csv_log, checkpoint])

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

predictions = model.predict(x_test).flatten()

print("\n🔹 First 10 Predictions vs Actual Values 🔹")
for i in range(10):
    print(f"Sample {i+1}: Actual = {y_test[i]}, Predicted Probability = {predictions[i]:.4f}")

save_path = "FILE PATH"
os.makedirs(save_path, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities")
plt.savefig(save_path + "histogram_predictions.png")  
plt.show()

print(f"\n✅ Plots saved in '{save_path}' folder.")
