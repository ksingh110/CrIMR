import numpy as np
import requests
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import random
import os

# Function to fetch CRY1 gene sequence from UCSC API
def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107093549")
    if response.status_code == 200:
        return response.json().get("dna", "").upper()
    else:
        return None

# One-hot encode sequence and return a 1D array
def onehotencoder(fasta_sequence, max_length=102500):
    sequence_array = np.array(list(fasta_sequence))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sequence_array)
    onehotencoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_sequence = onehotencoder.fit_transform(integer_encoded).astype(np.float32)
    
    if onehot_sequence.shape[0] < max_length:
        pad_size = max_length - onehot_sequence.shape[0]
        padding = np.zeros((pad_size, onehot_sequence.shape[1]))
        onehot_sequence = np.vstack([onehot_sequence, padding])
    else:
        onehot_sequence = onehot_sequence[:max_length, :]
    return onehot_sequence.flatten()

# Augment sequence by introducing substitutions, deletions, or insertions
def augment_sequence(seq, substitution_prob=0.1, deletion_prob=0.1, insertion_prob=0.1):
    augmented_seq = list(seq)
    seq_len = len(augmented_seq)

    for i in range(seq_len):
        # Substitution
        if random.random() < substitution_prob:
            augmented_seq[i] = random.choice(['A', 'G', 'C', 'T'])
        
        # Deletion
        elif random.random() < deletion_prob:
            augmented_seq[i] = ''
        
        # Insertion
        elif random.random() < insertion_prob:
            augmented_seq.insert(i, random.choice(['A', 'G', 'C', 'T']))
            seq_len += 1  # Increase sequence length after insertion
            i += 1  # Skip the next index after insertion to avoid double counting

    # Join the list back to a string after mutation
    augmented_seq = ''.join(augmented_seq)
    return augmented_seq

# Process augmented sequence and save to output file
def process_data_augmentation(cry1_seq, output_path, num_augmented_sequences=750):
    # Initialize existing data for appending
    if os.path.exists(output_path):
        existing_data = np.load(output_path, allow_pickle=True)["arr_0"].tolist()
    else:
        existing_data = []

    seq_count = 0
    batch = 250
    rows_save = []

    # Generate augmented sequences and save
    while seq_count < num_augmented_sequences:
        # Augment the sequence
        augmented_seq = augment_sequence(cry1_seq)
        print(f"Processed augmented sequence {seq_count + 1}.")
        if augmented_seq:
            encoded_seq = onehotencoder(augmented_seq)
            seq_count += 1
            rows_save.append(encoded_seq)

            if len(rows_save) >= batch:
                existing_data.extend(rows_save)
                np.savez_compressed(output_path, arr_0=np.array(existing_data))
                rows_save = []

    if rows_save:
        existing_data.extend(rows_save)
        np.savez_compressed(output_path, arr_0=np.array(existing_data))
    
    print(f"Processed {seq_count} augmented sequences and saved to {output_path}.")

# Path to output file
output_file = "E:\\datasets\\processeddata\\AUGMENTED_DATA_TRAINING_750.npz"

# Fetch CRY1 sequence and process data with augmentation
cry1_seq = get_CRY1_gene()
if cry1_seq:
    process_data_augmentation(cry1_seq, output_file)
else:
    print("Failed to fetch CRY1 gene sequence.")
