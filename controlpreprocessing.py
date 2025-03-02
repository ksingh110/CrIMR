
import gzip
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
output_path = "E:\\datasets\processeddata\cry1controlsequencesencoded.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
def onehotencoder(fasta_sequence, max_length=102500):
    sequence_array = np.array(list(fasta_sequence))

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sequence_array)

    onehotencoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    integer_encoded = integer_encoded.reshape(-1, 1)
    onehot_sequence = onehotencoder.fit_transform(integer_encoded)

    # Ensure uniform length
    if onehot_sequence.shape[0] < max_length:
        pad_size = max_length - onehot_sequence.shape[0]
        padding = np.zeros((pad_size, onehot_sequence.shape[1]), dtype=np.float32)
        onehot_sequence = np.vstack([onehot_sequence, padding])
    else:
        onehot_sequence = onehot_sequence[:max_length, :]  # Truncate if too long

    return onehot_sequence.flatten()
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
file_path = "E:\\cry1controlsequences (1).gz"
outputcontrol = []

with gzip.open(file_path, "rt") as f:  # Read as text
    current_entry = None
    for line in f:
        line = line.strip()
        if line.startswith(">"):  # Header line
            if current_entry:  # Save previous entry
                print(1)
                outputcontrol.append(onehotencoder(current_entry["sequence"]))
                print(len(onehotencoder(current_entry["sequence"])))
            current_entry = {"header": line, "sequence": ""}
        elif current_entry:
            current_entry["sequence"] += line  # Append sequence lines
        
    if current_entry:  # Append last entry
        outputcontrol.append(onehotencoder(current_entry["sequence"]))
print(outputcontrol)
 # Append new data
expected_shape = None
good_sequences = []
for i, seq in enumerate(outputcontrol):
    encoded_seq = onehotencoder(seq)

    if expected_shape is None:
        expected_shape = encoded_seq.shape  # Set first shape as reference
        print(f"Expected shape: {expected_shape}")

    if encoded_seq.shape != expected_shape:
        print(f"Warning: Sequence {i+1} has shape {encoded_seq.shape}, expected {expected_shape}. Skipping.")
        continue  # Skip inconsistent sequences

    good_sequences.append(encoded_seq)
    print("appended")
output_array = np.stack(good_sequences, axis=0)
np.savez_compressed(output_path, output_array)  # Save back to file
print(f"Saved control sequences to disk.")
print(len(good_sequences))
