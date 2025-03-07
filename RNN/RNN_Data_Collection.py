import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Reset TensorFlow graph
tf.compat.v1.enable_eager_execution()

# Paths and data
checkpoint_path = "E:/my_models/750_if_new_best_model.h5"
save_path = "E:/my_plots/750_if_new_prediction_plots/"
mutated_data = np.load("E:\datasets\processeddata\MUTATION_DATA_TRAINING_750.npz", allow_pickle=True, mmap_mode='r')
nonmutated_data = np.load("E:\\datasets\\processeddata\\AUGMENTED_DATA_TRAINING_750.npz", allow_pickle=True, mmap_mode='r')
csv_file = "cry1realvariations (1).csv"  # Assuming this CSV contains mutation data for anomaly detection

# Anomaly detection using Isolation Forest
def load_and_get_anomaly_scores(csv_file):
    df = pd.read_csv(csv_file)

    # Assuming the CSV file has columns 'Mutation ID' and 'Allele Frequency'
    data_id_array = df['_displayName'].values[:750].reshape(-1, 1)  # Get the first 1000 Mutation IDs
    array_data = df['AF'].values[:750].reshape(-1, 1)  # Get the first 1000 Allele Frequencies

    # Initialize and fit the Isolation Forest model
    clf = IsolationForest(contamination=0.01, random_state=42)
    clf.fit(array_data)

    # Get anomaly scores and predictions
    anomaly_scores = clf.decision_function(array_data)  # Negative scores represent outliers (anomalous)
    predictions = clf.predict(array_data)  # -1 for anomaly, 1 for normal

    # Invert the anomaly scores (as lower scores indicate anomalies, we invert so higher scores are more "normal")
    inverted_anomaly_scores = -anomaly_scores

    # Filter out NaN values and prepare the data for plotting
    valid_indices = ~np.isnan(array_data)
    valid_indices = valid_indices.flatten() 
    data_id_array = data_id_array[valid_indices]
    array_data = array_data[valid_indices]
    inverted_anomaly_scores = inverted_anomaly_scores[valid_indices]
    predictions = predictions[valid_indices]

    return inverted_anomaly_scores, data_id_array, array_data, predictions

# Get anomaly scores
anomaly_scores, _, _, _ = load_and_get_anomaly_scores(csv_file)

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

# Load data
mutated_sequences, input_shape = load_sequences(mutated_data, "mutated")
nonmutated_sequences, _ = load_sequences(nonmutated_data, "nonmutated")

mutated_labels = np.ones(mutated_sequences.shape[0])  # 1 for mutated
nonmutated_labels = np.zeros(nonmutated_sequences.shape[0])  # 0 for non-mutated

# Step 1: Generate random Mutation IDs for non-mutated data
nonmutated_data_with_ids = np.array([f"NonMut_{i}" for i in range(len(nonmutated_data.files))])

# Step 2: Add noise to the Allele Frequency for the non-mutated data
allele_frequency_mutated = pd.read_csv(csv_file)['AF'].values[:750]  # Allele Frequency of mutated data

# Generate random noise (Gaussian noise with mean 0 and std deviation same as mutated data's AF)
noise = np.random.normal(0, np.std(allele_frequency_mutated), len(nonmutated_data.files))

# Add noise to non-mutated allele frequencies (assuming a similar structure in 'nonmutated_data')
nonmutated_allele_frequency = np.random.normal(np.mean(allele_frequency_mutated), np.std(allele_frequency_mutated), len(nonmutated_data.files))

# Step 3: Incorporate noise into non-mutated data for Isolation Forest
df_nonmutated_with_noise = pd.DataFrame({
    '_displayName': nonmutated_data_with_ids,
    'AF': nonmutated_allele_frequency
})

# Step 4: Apply Isolation Forest on non-mutated data
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(df_nonmutated_with_noise['AF'].values.reshape(-1, 1))

# Get anomaly scores and predictions for non-mutated data
nonmutated_anomaly_scores = clf.decision_function(df_nonmutated_with_noise['AF'].values.reshape(-1, 1))
nonmutated_predictions = clf.predict(df_nonmutated_with_noise['AF'].values.reshape(-1, 1))

# Invert the anomaly scores (lower scores = anomalies)
nonmutated_inverted_anomaly_scores = -nonmutated_anomaly_scores

# Append anomaly scores to features (only for the mutated sequences)
anomaly_scores_reshaped = anomaly_scores[:mutated_sequences.shape[0]].reshape(-1, 1)
# Expand anomaly_scores_reshaped to 3D (1000, 1, 1) to match the timesteps axis of X
anomaly_scores_reshaped = np.expand_dims(anomaly_scores_reshaped, axis=-1)  # shape (1000, 1, 1)

# Update the last feature of each sequence with the anomaly score
mutated_sequences_with_anomalies = mutated_sequences.copy()  # Make a copy of mutated_sequences
mutated_sequences_with_anomalies[:, :, -1:] = anomaly_scores_reshaped  # Replace the last feature with the anomaly score
# Concatenate anomaly scores with mutated sequences only

# Concatenate the mutated and non-mutated sequences (with anomalies added to mutated)
X_with_anomalies = np.concatenate([mutated_sequences_with_anomalies, nonmutated_sequences], axis=0)
y = np.concatenate([mutated_labels, nonmutated_labels], axis=0)

x_train, x_test, y_train, y_test = train_test_split(X_with_anomalies, y, random_state=42, test_size=0.2, stratify=y)

model1_path = "E:\\my_models\\750_if_new_best_model.h5"
model2_path = "E:\\my_models\\750_best_model.h5"

# Load the saved RNN models
model1 = load_model(model1_path)
model2 = load_model(model2_path)

# Evaluate models on test set
loss1, acc1 = model1.evaluate(x_test, y_test)
loss2, acc2 = model2.evaluate(x_test, y_test)

# Get predictions
y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)

# Calculate ROC Curve and AUC
fpr1, tpr1, _ = roc_curve(y_test, y_pred1)
fpr2, tpr2, _ = roc_curve(y_test, y_pred2)
roc_auc1 = roc_auc_score(y_test, y_pred1)
roc_auc2 = roc_auc_score(y_test, y_pred2)

# Classification Reports
report1 = classification_report(y_test, (y_pred1 > 0.5))
report2 = classification_report(y_test, (y_pred2 > 0.5))

# Plotting comparison of accuracies
plt.figure(figsize=(10, 6))

# Plot training accuracy
plt.plot(history1.history['accuracy'], label='Model 1 (Training)', color='blue')
plt.plot(history2.history['accuracy'], label='Model 2 (Training)', color='orange')

# Plot validation accuracy
plt.plot(history1.history['val_accuracy'], label='Model 1 (Validation)', color='blue', linestyle='--')
plt.plot(history2.history['val_accuracy'], label='Model 2 (Validation)', color='orange', linestyle='--')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.legend(loc='lower right')

# Show the plot
plt.show()

# ROC Curves plot
plt.figure(figsize=(10, 6))
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'Model 1 AUC = {roc_auc1:.2f}')
plt.plot(fpr2, tpr2, color='orange', lw=2, label=f'Model 2 AUC = {roc_auc2:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Print classification reports
print("Model 1 Classification Report:")
print(report1)
print("\nModel 2 Classification Report:")
print(report2)
