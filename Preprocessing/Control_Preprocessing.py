import gzip
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Output file path
output_path = "E:\\datasets\\processeddata\\CRY1ENCODEDCONTROLWORKS1.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load existing data if available
if os.path.exists(output_path):
    existing_data = np.load(output_path, allow_pickle=True)['arr_0']
    print(f"Loaded existing data: {existing_data.shape}")
else:
    existing_data = np.empty((0, 102500, 14), dtype=np.float32)

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
batch_size = 1000
outputcontrol = []  # List to store processed sequences

with gzip.open("E:\\datasets\\processeddata\\AUGMENTEDCONTROLWORKS.gz", "rb") as f:  # Open in binary mode
    sequences = np.load(f, allow_pickle=True)  # Load the file as a NumPy array
    print(f"Loaded {len(sequences)} sequences from AUGMENTEDCONTROLWORKS.gz")

    for idx, seq in enumerate(sequences):
        # Apply one-hot encoding
        encoded_seq = onehotencoder(seq)

        # Ensure correct shape before saving
        if encoded_seq.shape == (102500, 14):
            outputcontrol.append(encoded_seq)
            print(f"Processed sequence {idx + 1}, Shape: {encoded_seq.shape}")
        else:
            print(f"Skipping sequence {idx + 1} due to incorrect shape: {encoded_seq.shape}")

        # Save in batches
        if len(outputcontrol) >= batch_size:
            batch_array = np.array(outputcontrol, dtype=np.float32)  # Convert list to NumPy array

            # Debugging: Print current count
            print(f"Appending batch of {batch_array.shape} to existing_data (Current size: {existing_data.shape})")

            existing_data = np.concatenate((existing_data, batch_array), axis=0)  # Append batch to existing data

            # Save updated dataset
            np.savez_compressed(output_path, arr_0=existing_data)
            print(f"Saved {batch_array.shape[0]} new sequences. Total saved: {existing_data.shape[0]}")

            outputcontrol = []  # Clear batch after saving

# Save any remaining sequences that weren't saved in the last batch
if outputcontrol:
    batch_array = np.array(outputcontrol, dtype=np.float32)  # Stack remaining sequences
    print(f"Appending final batch of {batch_array.shape} to existing_data")

    existing_data = np.concatenate((existing_data, batch_array), axis=0)  # Append remaining data

    np.savez_compressed(output_path, arr_0=existing_data)
    print(f"Final save: {batch_array.shape[0]} sequences. Total saved: {existing_data.shape[0]}")

print("All sequences have been processed and saved.")
