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
def onehotencoder(fasta_sequence, max_length=13000):
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
def augment_sequence(seq, substitution_prob=0.33, deletion_prob=0.33, insertion_prob=0.33):
    augmented_seq = list(seq)
    seq_len = len(augmented_seq)

    for i in range(seq_len):
        if random.random() < substitution_prob:  # Substitution
            augmented_seq[i] = random.choice(['A', 'G', 'C', 'T'])
        elif random.random() < deletion_prob:  # Deletion
            augmented_seq[i] = ''
        elif random.random() < insertion_prob:  # Insertion
            augmented_seq.insert(i, random.choice(['A', 'G', 'C', 'T']))
            seq_len += 1  
            i += 1  

    return ''.join(augmented_seq)

# Generate multiple augmented sequences and save separately
def generate_and_save_sequences(cry1_seq, num_sequences=5):
    for i in range(1, num_sequences + 1):
        augmented_seq = augment_sequence(cry1_seq)  # Generate a new augmented sequence each time
        encoded_seq = onehotencoder(augmented_seq)
        output_file = f"testing{i}.npz"
        np.savez_compressed(output_file, arr_0=np.array([encoded_seq]))  # Save each sequence separately
        print(f"Saved {output_file}.")

# Fetch CRY1 sequence and generate 5 different augmented test sequences
cry1_seq = get_CRY1_gene()
if cry1_seq:
    generate_and_save_sequences(cry1_seq, num_sequences=5)
else:
    print("Failed to fetch CRY1 gene sequence.")
