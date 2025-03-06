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

# Process the mutation CSV and apply mutations to the CRY1 sequence
def process_mutations(csv_path, cry1_seq, output_path):
    df = pd.read_csv(csv_path, usecols=['chromEnd', 'ref', 'alt', 'AF', 'genes', 'variation_type', '_displayName'])
    df = df[df["variation_type"].str.contains("intron_variant", na=False, case=False)]

    # Initialize existing data for appending
    if os.path.exists(output_path):
        existing_data = np.load(output_path, allow_pickle=True)["arr_0"].tolist()
    else:
        existing_data = []

    # Process each row in the CSV
    seq_count = 0
    batch = 250
    rows_save = []
    max_seq_count = 750
    for index, row in df.iterrows():
        if seq_count >= max_seq_count:
            print("Reached maximum sequence count.")
            break
        gnomAD_ID = row["_displayName"]
        refnuc = str(row["ref"]) if pd.notna(row["ref"]) else ""
        start = row["chromEnd"] - 106991364 - len(row["ref"])
        end = row["chromEnd"] - 106991364
        altnuc = str(row["alt"]) if pd.notna(row["alt"]) else ""

        # Add mutation to the sequence
        mutated_seq = add_mutation(cry1_seq, start, end, refnuc, altnuc)
        print(f"Processed mutation {gnomAD_ID} at positions {start}-{end}.")
        if mutated_seq:
            encoded_seq = onehotencoder(mutated_seq)
            seq_count += 1
            rows_save.append(encoded_seq)

            if len(rows_save) >= batch:
                existing_data.extend(rows_save)
                np.savez_compressed(output_path, arr_0=np.array(existing_data))
                rows_save = []
    
    if rows_save:
        existing_data.extend(rows_save)
        np.savez_compressed(output_path, arr_0=np.array(existing_data))
    print(f"Processed {seq_count} mutated sequences and saved to {output_path}.")

# Path to CSV file with mutations and output path
csv_file = "cry1realvariations (1).csv"
output_file = "E:\\datasets\\processeddata\\MUTATION_DATA_TRAINING_750.npz"

# Fetch CRY1 sequence and process mutations
cry1_seq = get_CRY1_gene()
if cry1_seq:
    process_mutations(csv_file, cry1_seq, output_file)
else:
    print("Failed to fetch CRY1 gene sequence.")
