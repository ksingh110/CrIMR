# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split

# Set n (number of time steps)
n = 1

# Load data
mutated_data = np.load("E:\\datasets\\processeddata\\encoded_mutation_sequences.npz", mmap_mode='r')
nonmutated_data = np.load("E:\\datasets\\processeddata\\AUGMENTEDCONTROLWORKStest31689.npz", mmap_mode='r', allow_pickle=True)

# Extract sequences from the files
mutated_sequences = mutated_data["arr_0"][:100]  # or the appropriate key
nonmutated_sequences = nonmutated_data["arr_0"]  # or the appropriate key

# Ensure mutated_sequences has 3 dimensions (samples, n, features)
mutated_sequences = mutated_sequences.reshape(mutated_sequences.shape[0], n, mutated_sequences.shape[1])

# Ensure nonmutated_sequences has 3 dimensions (samples, n, features)
if len(nonmutated_sequences.shape) == 1:  # If nonmutated_sequences has 2 dimensions
    nonmutated_sequences = nonmutated_sequences.reshape(nonmutated_sequences.shape[0], n, 1)  # Reshape as (samples, n, 1)

# Repeat the nonmutated_sequences to match the feature length of 410000
nonmutated_sequences = np.repeat(nonmutated_sequences, 410000, axis=-1)

# Now concatenate the sequences
X = np.concatenate((mutated_sequences, nonmutated_sequences), axis=0)  # Concatenate along the first axis (samples)

# Labels for mutated and non-mutated sequences
mutated_labels = np.ones(mutated_sequences.shape[0])  # 1 for mutated
nonmutated_labels = np.zeros(nonmutated_sequences.shape[0])  # 0 for non-mutated

# Concatenate sequences and labels
X = np.concatenate((mutated_sequences, nonmutated_sequences), axis=0)
y = np.concatenate((mutated_labels, nonmutated_labels), axis=0)

# Ensure the data types are in correct format (numeric)
X = X.astype(np.float32)
y = y.astype(np.int32)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

# Define RNN model
def rnn_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define input shape based on reshaped data
input_shape = (n, 410000)

# Create the model
model = rnn_model(input_shape)

# Callbacks for model saving and training log
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, mode="min")
csv_log = CSVLogger("training_log.csv", append=True)

# Train the model
rnn_fit = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_test, y_test), callbacks=[csv_log, checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
predictions = model.predict(x_test).flatten()

print("\n🔹 First 10 Predictions vs Actual Values 🔹")
for i in range(10):
    print(f"Sample {i+1}: Actual = {y_test[i]}, Predicted Probability = {predictions[i]:.4f}")

# Save predictions plots
save_path = "prediction_plots/"
os.makedirs(save_path, exist_ok=True)

# Plot histogram of predictions
plt.figure(figsize=(8, 6))
plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities")
plt.savefig(save_path + "histogram_predictions.png")
plt.show()

print(f"\n Plots saved in '{save_path}' folder.")
