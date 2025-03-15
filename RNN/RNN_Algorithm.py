import numpy as np 
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import pandas as pd
import tensorflow as tf
from tensorflow.keras.regularizers import l2

# Generate classification report, which includes precision, recall, and F1-score
tf.compat.v1.reset_default_graph()

checkpoint_path = "E:/my_models/1000_new_best_model.h5"
save_path = "E:/my_plots/1000_new_prediction_plots/"
mutated_data = np.load("E:\datasets\processeddata\MUTATION_DATA_TRAINING_1000.npz", allow_pickle=True, mmap_mode='r')
nonmutated_data = np.load("E:\\datasets\\processeddata\\AUGMENTED_DATA_TRAINING_1000.npz", allow_pickle=True, mmap_mode='r')



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

mutated_sequences, input_shape = load_sequences(mutated_data, "mutated")
nonmutated_sequences, _ = load_sequences(nonmutated_data, "nonmutated")

mutated_labels = np.ones(mutated_sequences.shape[0])  # 1 for mutated
nonmutated_labels = np.zeros(nonmutated_sequences.shape[0])  # 0 for non-mutated

X = np.concatenate((mutated_sequences, nonmutated_sequences), axis=0)
y = np.concatenate((mutated_labels, nonmutated_labels), axis=0)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

def rnn_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False, activation = "relu"),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])
    model.compile(optimizer="adamw", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def reset_rnn(input_shape):
    # Recreate the model to reset weights
    model = rnn_model(input_shape)
    return model
def plot_learning_curve(history, save_path):
    # Extract training and validation loss (or accuracy)
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']

    epochs = range(1, len(training_loss) + 1)

    # Plot loss curves
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_acc, label='Training Accuracy')
    plt.plot(epochs, validation_acc, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

# Plot learning curves

# Reset model
model = reset_rnn(input_shape)

checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, mode="min")
csv_log = CSVLogger("training_log_new_1000.csv", append=True)

rnn_fit = model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1, validation_data=(x_test, y_test), callbacks=[csv_log, checkpoint])

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

predictions = model.predict(x_test).flatten()

print("\n🔹 First 10 Predictions vs Actual Values 🔹")
for i in range(10):
    print(f"Sample {i+1}: Actual = {y_test[i]}, Predicted Probability = {predictions[i]:.4f}")


os.makedirs(save_path, exist_ok=True)

plt.figure(figsize=(14, 10))
plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities")
plt.savefig(save_path + "histogram_predictions.png")  
plt.show()

print(f"\n Plots saved in '{save_path}' folder.")


# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(save_path + "1000_new_roc_curve.png")

print(f"Area Under the Curve (AUC): {roc_auc:.4f}")

report = classification_report(y_test, (predictions > 0.5).astype(int))
print("\nClassification Report:" + report)
report = classification_report(y_test, (predictions > 0.5).astype(int), output_dict=True)

# Convert the dictionary into a DataFrame and transpose it
report_df = pd.DataFrame(report).transpose()

# Save the report to a CSV file
report_df.to_csv("1000_new_classification_report.csv")

print("Classification report saved to 'classification_report.csv'.")
print("\nClassification Report:")

plot_learning_curve(rnn_fit,  save_path="1000_new_learning_curve")
