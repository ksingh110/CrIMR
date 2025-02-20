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
    print (seq_start)
    print (seq_end) 
    print (refnuc)
    print (seq[seq_start:seq_end])
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

# Get the CRY1 sequence
cry1_seq = get_CRY1_gene()
print(cry1_seq)
encoded_sequences = []
seq_count = 0
max_seq_count = 3000
if cry1_seq:
        for index, row in df.iterrows():
            if seq_count> max_seq_count:
                print(f"Processed {seq_count} sequences. Stopping.")
                break
            gnomAD_ID = row["_displayName"]
            print(f"gnomAD_ID: {gnomAD_ID}")  # Corrected print statement
            refnuc = str(row["ref"]) if pd.notna(row["ref"]) else ""
            start = row["chromEnd"] - 106991364- len(row["ref"]) 
            end = row["chromEnd"] - 106991364
            altnuc = str(row["alt"]) if pd.notna(row["alt"]) else ""
                            
                # Make sure chromnum is correctly formatted (if needed)
                # For instance, ensure it starts with "chr" or adjust as required
                
            mutated_sequence = add_mutation(cry1_seq, start, end, refnuc, altnuc)
            if mutated_sequence:
                ohe_data = processdata(mutated_sequence)
                encoded_sequences.append(ohe_data)
                seq_count += 1
                print(f"At sequence: {seq_count}")
os.chdir("E:/data/")  
output_path = "datasets/processeddata/encoded_mutation_sequences.h5"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
print("output made")
num_variants = len(df)
encoded_shape = (num_variants, 410000)  # Each sequence is (102500 * 4)

# Create a memory-mapped array
encoded_sequences = np.memmap(output_path, dtype=np.float16, mode="w+", shape=encoded_shape)

for idx, row in df.iterrows():
    gnomAD_ID = row["_displayName"]
    refnuc = str(row["ref"]) if pd.notna(row["ref"]) else ""
    start = row["chromEnd"] - 106991364 - len(row["ref"])
    end = row["chromEnd"] - 106991364
    altnuc = str(row["alt"]) if pd.notna(row["alt"]) else ""

    mutated_sequence = add_mutation(cry1_seq, start, end, refnuc, altnuc)
    if mutated_sequence:
        ohe_data = processdata(mutated_sequence)
        encoded_sequences[idx] = ohe_data  # Directly write to disk
        print(f"At sequence: {idx + 1}")
encoded_sequences.flush()  # Ensure data is written

print(f"Encoded shape: {encoded_sequences.shape}")