
import gzip
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Define output path and ensure directory exists
output_path = r"E:\datasets\processeddata\cry1controlsequencesencoded.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# One-hot encoding function
def onehotencoder(fasta_sequence, max_length=102500):
    sequence_array = np.array(list(fasta_sequence))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sequence_array)

    onehotencoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_sequence = onehotencoder.fit_transform(integer_encoded).astype(np.float32)

    # Padding or truncating to max_length
    num_features = onehot_sequence.shape[1]  # Number of one-hot columns
    if onehot_sequence.shape[0] < max_length:
        pad_size = max_length - onehot_sequence.shape[0]
        padding = np.zeros((pad_size, num_features), dtype=np.float32)
        onehot_sequence = np.vstack([onehot_sequence, padding])
    else:
        onehot_sequence = onehot_sequence[:max_length, :]
    
    return onehot_sequence  # No flattening to keep shape consistent
        
file_path = "E:\\cry1controlsequences.gz"
outputcontrol = []

with gzip.open(file_path, "rt") as f:  # Read as text
    current_entry = None
    for line in f:
        line = line.strip()
        if line.startswith(">"):  # Header line
            if current_entry:  # Save previous entry
                outputcontrol.append(onehotencoder(current_entry["sequence"]))
            current_entry = {"header": line, "sequence": ""}
        elif current_entry:
            current_entry["sequence"] += line  # Append sequence lines
        
    if current_entry:  # Append last entry
        outputcontrol.append(onehotencoder(current_entry["sequence"]))
 # Append new data
output_array = np.stack(outputcontrol, axis=0)
np.savez_compressed(output_path, output_array)  # Save back to file
print(f"Saved control sequences to disk.")


