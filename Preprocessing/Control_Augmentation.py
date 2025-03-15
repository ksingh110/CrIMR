import gzip
import random
import numpy as np
import nlpaug.augmenter.char as nac
import requests
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os

# File paths
input_file = "E:\\cry1controlsequences (1).gz"
output_file = "E:\\datasets\\processeddata\\AUGMENTEDCONTROLWORKS12345678.npz"

# Load existing data correctly
if os.path.exists(output_file):
    with np.load(output_file, allow_pickle=True) as data:
        existing_data = list(data["arr_0"])  # Retrieve stored array as list
else:
    existing_data = []

def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107093549")
    if response.status_code == 200:
        return response.json().get("dna", "").upper()
    else:
        return None

cry1 = get_CRY1_gene()

# Read original sequences
def read_fasta(file_path):
    with gzip.open(file_path, "rt") as handle:
        return [str(record.seq) for record in SeqIO.parse(handle, "fasta")]

# Augment sequence function
def augment_sequence(sequence, aug):
    return aug.augment(sequence)  # Returns a list of sequences

# One-Hot Encoding Function
def onehotencoder(fasta_sequence, max_length = 102500):
    sequence_array = np.array(list(fasta_sequence))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sequence_array)
    onehotencoder = OneHotEncoder(sparse_output=False, dtype = np.float32)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_sequence = onehotencoder.fit_transform(integer_encoded).astype(np.float32)
    if onehot_sequence.shape[0] < max_length:
        pad_size = max_length - onehot_sequence.shape[0]
        padding = np.zeros((pad_size, onehot_sequence.shape[1]))
        onehot_sequence = np.vstack([onehot_sequence, padding])
    else:
        onehot_sequence = onehot_sequence[:max_length, :] 
    return(onehot_sequence.flatten())

# Mutation types
type_mutation = ("insert", "substitute", "swap", "delete")

# Read sequences
sequences = read_fasta(input_file)
augmented_sequences = []
batch = 50
num_augmentations_per_seq = 4000
total_saved = 0

original_seq = cry1
for _ in range(num_augmentations_per_seq):
    mutation = random.choice(type_mutation)

    if mutation == "insert":
        aug = nac.RandomCharAug(action="insert", aug_char_p=0.33)
    elif mutation == "swap":
        aug = nac.RandomCharAug(action="swap", aug_char_p=0.33)
    elif mutation == "delete":
        aug = nac.RandomCharAug(action="delete", aug_char_p=0.33)

    augmented_seq_list = augment_sequence(original_seq, aug)

    print(f"Generated {len(augmented_seq_list)} augmented sequences.")

    for seq in augmented_seq_list:
        encoded_seq = onehotencoder(seq)  # Process each sequence
        augmented_sequences.append(encoded_seq)

    # Save in batches
    if len(augmented_sequences) >= batch:
        existing_data.extend(augmented_sequences)
        print(f"Saving {len(augmented_sequences)} sequences...")
        
        # Use dictionary format to avoid overwriting the `.npz` file
        np.savez_compressed(output_file, sequences=np.array(existing_data, dtype=object))
        
        total_saved += len(augmented_sequences)
        print(f"Total saved so far: {total_saved}")
        
        augmented_sequences = []  # Reset batch after saving

# Save any remaining sequences


print(f"Final augmented sequences count: {len(existing_data)}")
