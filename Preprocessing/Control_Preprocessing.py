import gzip
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Output file path
output_path = "E:\\datasets\\processeddata\\CRY1ENCODEDCONTROLWORKS.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load existing data if available
if os.path.exists(output_path):
    existing_data = np.load(output_path, allow_pickle=True)['arr_0'].tolist()  # Load existing sequences
else:
    existing_data = []

# One-hot encoding function
def onehotencoder(fasta_sequence, max_length=102500, max_columns=14):
    sequence_array = np.array(list(fasta_sequence))

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sequence_array)

    onehotencoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    integer_encoded = integer_encoded.reshape(-1, 1)
    onehot_sequence = onehotencoder.fit_transform(integer_encoded)

    # Ensure uniform length by padding or truncating sequences to `max_length`
    if onehot_sequence.shape[0] < max_length:
        pad_size = max_length - onehot_sequence.shape[0]
        padding = np.zeros((pad_size, onehot_sequence.shape[1]), dtype=np.float32)
        onehot_sequence = np.vstack([onehot_sequence, padding])
    elif onehot_sequence.shape[0] > max_length:
        onehot_sequence = onehot_sequence[:max_length, :]  # Truncate if too long

    # Ensure uniform number of columns
    if onehot_sequence.shape[1] < max_columns:
        pad_columns = max_columns - onehot_sequence.shape[1]
        padding_columns = np.zeros((onehot_sequence.shape[0], pad_columns), dtype=np.float32)
        onehot_sequence = np.hstack([onehot_sequence, padding_columns])  # Pad columns
    elif onehot_sequence.shape[1] > max_columns:
        onehot_sequence = onehot_sequence[:, :max_columns]  # Truncate columns if too many

    return onehot_sequence

# Process sequences in batches
batch = 200
outputcontrol = []  # List to store processed sequences
with gzip.open("E:\\datasets\\processeddata\\AUGMENTEDCONTROLWORKS.gz", "rb") as f:  # Open in binary mode
    sequences = np.load(f, allow_pickle=True)  # Load the file as a NumPy array

    for idx, seq in enumerate(sequences):
        # Apply one-hot encoding
        encoded_seq = onehotencoder(seq)
        
        # Check that the encoded sequence has the correct shape (102500, X)
        if encoded_seq.shape[0] == 102500:
            outputcontrol.append(encoded_seq)
            print(f"Processed sequence {idx + 1}, Shape: {encoded_seq.shape}")
        else:
            print(f"Skipping sequence {idx + 1} due to incorrect shape: {encoded_seq.shape}")

        # Save in batches
        if len(outputcontrol) >= batch:
            # Stack the batch into a single array for uniformity
            batch_array = np.array(outputcontrol)  # Ensure it's a NumPy array
            existing_data.extend(batch_array)  # Append new data
            # Save the stacked array
            np.savez_compressed(output_path, arr_0=np.array(existing_data))
            print(f"Saved {len(outputcontrol)} sequences to disk.")
            outputcontrol = []  # Clear batch after saving

    # Save any remaining sequences that weren't saved in the last batch
    if outputcontrol:
        batch_array = np.array(outputcontrol)  # Stack the remaining batch
        existing_data.extend(batch_array)
        np.savez_compressed(output_path, arr_0=np.array(existing_data))
        print(f"Saved remaining sequences to disk.")

print("All sequences have been processed and saved.")
