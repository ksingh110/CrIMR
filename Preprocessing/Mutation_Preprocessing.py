import numpy as np
import pandas as pd
import os
import requests
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

# Function to add mutation to the CRY1 sequence
def add_mutation(seq, start, end, refnuc, altnuc):
    seq_start = start
    seq_end = end
    if seq[seq_start:seq_end] != refnuc:
        print(f"Mutation mismatch at positions {start}-{end}.")
        return None
    else:
        mutated_seq = seq[:seq_start] + altnuc + seq[seq_end:]
        return mutated_seq


def generate_mutated_sequences(csv_path, cry1_seq, num_files=5):
    df = pd.read_csv(csv_path, usecols=['chromEnd', 'ref', 'alt', 'AF', 'genes', 'variation_type', '_displayName'])
    df = df[df["variation_type"].str.contains("intron_variant", na=False, case=False)]

    # Shuffle mutations to create diverse mutation sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    mutations_per_file = len(df) // num_files  # Divide mutations across files

    for i in range(num_files):
        output_path = f"Test_M{i+1}.npz"
        start_idx = i * mutations_per_file
        end_idx = (i + 1) * mutations_per_file if i < num_files - 1 else len(df)
        mutation_subset = df.iloc[start_idx:end_idx]

        mutated_seq = cry1_seq
        for _, row in mutation_subset.iterrows():
            gnomAD_ID = row["_displayName"]
            refnuc = str(row["ref"]) if pd.notna(row["ref"]) else ""
            start = row["chromEnd"] - 106991364 - len(row["ref"])
            end = row["chromEnd"] - 106991364
            altnuc = str(row["alt"]) if pd.notna(row["alt"]) else ""

            new_mutation = add_mutation(mutated_seq, start, end, refnuc, altnuc)
            if new_mutation:
                mutated_seq = new_mutation
                print(f"Applied mutation {gnomAD_ID} at positions {start}-{end}.")

        encoded_seq = onehotencoder(mutated_seq)
        np.savez_compressed(output_path, arr_0=np.array([encoded_seq]))
        print(f"Saved {output_path} with unique mutations.")

# Path to CSV file with mutations
csv_file = "cry1realvariations (1).csv"

# Fetch CRY1 sequence and generate 5 distinct mutation files
cry1_seq = get_CRY1_gene()
if cry1_seq:
    generate_mutated_sequences(csv_file, cry1_seq, num_files=5)
else:
    print("Failed to fetch CRY1 gene sequence.")
