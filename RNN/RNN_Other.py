import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import pandas as pd

# Paths for data and model
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

# Generate random Mutation IDs for non-mutated data
nonmutated_data_with_ids = np.array([f"NonMut_{i}" for i in range(len(nonmutated_data.files))])

# Add noise to the Allele Frequency for the non-mutated data
allele_frequency_mutated = pd.read_csv(csv_file)['AF'].values[:750]  # Allele Frequency of mutated data

# Generate random noise (Gaussian noise with mean 0 and std deviation same as mutated data's AF)
noise = np.random.normal(0, np.std(allele_frequency_mutated), len(nonmutated_data.files))

# Add noise to non-mutated allele frequencies (assuming a similar structure in 'nonmutated_data')
nonmutated_allele_frequency = np.random.normal(np.mean(allele_frequency_mutated), np.std(allele_frequency_mutated), len(nonmutated_data.files))

# Incorporate noise into non-mutated data for Isolation Forest
df_nonmutated_with_noise = pd.DataFrame({
    '_displayName': nonmutated_data_with_ids,
    'AF': nonmutated_allele_frequency
})

# Apply Isolation Forest on non-mutated data
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(df_nonmutated_with_noise['AF'].values.reshape(-1, 1))

# Get anomaly scores and predictions for non-mutated data
nonmutated_anomaly_scores = clf.decision_function(df_nonmutated_with_noise['AF'].values.reshape(-1, 1))
nonmutated_predictions = clf.predict(df_nonmutated_with_noise['AF'].values.reshape(-1, 1))

# Invert the anomaly scores (lower scores = anomalies)
nonmutated_inverted_anomaly_scores = -nonmutated_anomaly_scores

# Append anomaly scores to features (only for the mutated sequences)
anomaly_scores_reshaped = anomaly_scores[:mutated_sequences.shape[0]].reshape(-1, 1)
anomaly_scores_reshaped = np.expand_dims(anomaly_scores_reshaped, axis=-1)  # shape (1000, 1, 1)

# Update the last feature of each sequence with the anomaly score
mutated_sequences_with_anomalies = mutated_sequences.copy()
mutated_sequences_with_anomalies[:, :, -1:] = anomaly_scores_reshaped

# Concatenate the mutated and non-mutated sequences (with anomalies added to mutated)
X_with_anomalies = np.concatenate([mutated_sequences_with_anomalies, nonmutated_sequences], axis=0)
y = np.concatenate([mutated_labels, nonmutated_labels], axis=0)

x_train, x_test, y_train, y_test = train_test_split(X_with_anomalies, y, random_state=42, test_size=0.2, stratify=y)

# Load the pretrained RNN model (LSTM in this case)
model = load_model(checkpoint_path)

# Step 1: Get predictions from the pretrained RNN model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
predictions = model.predict(x_test).flatten()
print("\n🔹 First 10 Predictions vs Actual Values 🔹")
for i in range(10):
    print(f"Sample {i+1}: Actual = {y_test[i]}, Predicted Probability = {predictions[i]:.4f}")

# Save plots
os.makedirs(save_path, exist_ok=True)

# Histogram of predictions
plt.figure(figsize=(14, 10))
plt.hist(predictions, bins=20, edgecolor='black', alpha=0.7)

# Calculate percentages
counts, bins = np.histogram(predictions, bins=20)
percentages = (counts / len(predictions)) * 100

# Add percentage text below the x-axis
bin_centers = (bins[:-1] + bins[1:]) / 2
for i in range(len(counts)):
    plt.text(bin_centers[i], counts[i] + 0.5, f'{percentages[i]:.1f}%', ha='center', color='black', fontsize=12)

plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities")

# Show number of samples on bars
for i in range(len(counts)):
    plt.text(bin_centers[i], counts[i] + 0.5, str(counts[i]), ha='center', color='black', fontsize=12)

# Save and show plot
plt.savefig(save_path + "histogram_predictions_with_percentage.png")
plt.show()
