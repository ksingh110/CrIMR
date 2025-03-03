import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import gzip
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
output_path = "E:\\datasets\processeddata\cry1controlsequencesencodedreall.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
if os.path.exists(output_path):
    existing_data = np.load(output_path, allow_pickle=True).tolist()  # Convert to list for easy appending
else:
    existing_data = []
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
        
file_path = "E:\datasets\processeddata\_augmentedcry1controlsequencesencoded.gz"
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
batch = 200
expected_shape = None
good_sequences = []
for i, seq in enumerate(outputcontrol):
    encoded_seq = onehotencoder(seq)


    good_sequences.append(encoded_seq)
    print("appended")
    if len(good_sequences) >= batch:  # Write to disk every 50 sequences
                  existing_data.extend(good_sequences)  # Append new data
                  np.savez_compressed(output_path, np.array(existing_data))  # Save back to file
                  print(f"Saved {len(good_sequences)} sequences to disk.")
                  good_sequences= []  # Clear batch
    np.savez_compressed(output_path, np.array(existing_data))
print("All files")




