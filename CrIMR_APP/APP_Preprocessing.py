import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import requests

# Function to fetch CRY1 gene sequence from UCSC API
def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107093549")
    if response.status_code == 200:
        return response.json().get("dna", "").upper()  # Return the DNA sequence in uppercase
    else:
        return None  # Return None if the request fails

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
    return onehot_sequence.flatten() # Return the sequence without flattening for compatibility with LSTM

# Function to process sequence (one-hot encoded) and return it in a format suitable for further use
def process(sequence):
    encoded_seq = onehotencoder(sequence)  # One-hot encode the input sequence
    return np.array([encoded_seq])  # Return the one-hot encoded sequence as a 2D array (1 sequence)

# Example usage
if __name__ == "__main__":
    # Fetch the CRY1 gene sequence
    cry1_sequence = get_CRY1_gene()

    if cry1_sequence:
        # Process the CRY1 sequence and print the result
        processed_data = process(cry1_sequence)
        print(f"Processed data shape: {processed_data.shape}")  # Shape should be (1, max_length, 4) for LSTM input
    else:
        print("Failed to retrieve CRY1 gene sequence.")
