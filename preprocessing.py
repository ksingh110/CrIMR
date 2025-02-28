from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pyfastx
import glob
import os
import numpy as np
import pandas as pd
import requests
import h5py
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
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def processdata(file):
    encoded_sequence = onehotencoder(file, max_length =  102500)
    return encoded_sequence


def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107093549")
    if response.status_code == 200:
        return response.json().get("dna", "").upper()
    else:
        return None

# Function to add CRY1 mutation to the regular CRY1 sequence
def add_mutation(seq, start, end, refnuc, altnuc):
 
    seq_start = start 
    seq_end = end
    if seq[seq_start:seq_end] != refnuc:
        print(f"The positions of the sequences do not match, or the sequence does not exist.")
        return None
    else:
        mutated_seq = seq[:seq_start] + altnuc + seq[seq_end:]
        print(f"Matches")
     
        return mutated_seq

# Read the CSV file and store it in a variable
csv = "cry1realvariations (1).csv"
df = pd.read_csv(csv, usecols=['chromEnd', 'ref', 'alt', 'AF', 'genes', 'variation_type', '_displayName'])
df = df[df["variation_type"].str.contains("intron_variant", na=False, case=False)]
# output file
output_path = "E:\\datasets\processeddata\encoded_mutation_sequences_real.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
if os.path.exists(output_path):
    existing_data = np.load(output_path, allow_pickle=True).tolist()  # Convert to list for easy appending
else:
    existing_data = []
# Get the CRY1 sequence
cry1_seq = get_CRY1_gene()
print(cry1_seq)
encoded_sequences = []
seq_count = 0
batch = 200
rows_save = []
if cry1_seq:
        for index, row in df.iterrows():
         #   if seq_count> max_seq_count:
         #       print(f"Processed {seq_count} sequences. Stopping.")
        #        break
            gnomAD_ID = row["_displayName"]
            print(f"gnomAD_ID: {gnomAD_ID}")  # Corrected print statement
            refnuc = str(row["ref"]) if pd.notna(row["ref"]) else ""
            start = row["chromEnd"] - 106991364- len(row["ref"]) 
            end = row["chromEnd"] - 106991364
            altnuc = str(row["alt"]) if pd.notna(row["alt"]) else ""
                            
                # Make sure chromnum is correctly formatted (if needed)
                # For instance, ensure it starts with "chr" or adjust as required
            mutated_seq = add_mutation(cry1_seq, start, end, refnuc, altnuc)
            encoded_seq = onehotencoder(mutated_seq)
            seq_count += 1
            print(f"At sequence: {seq_count}")  

            rows_save.append(encoded_seq)

            if len(rows_save) >= batch:  # Write to disk every 50 sequences
                  existing_data.extend(rows_save)  # Append new data
                  np.savez_compressed(output_path, np.array(existing_data))  # Save back to file
                  print(f"Saved {len(rows_save)} sequences to disk.")
                  rows_save = []  # Clear batch
           

print("Saved to disk")
