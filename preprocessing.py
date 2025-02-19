from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pyfastx
import glob
import os
import numpy as np


def onehotencoder(fasta_sequence, max_length = 500):
    sequence_array = np.array(list(fasta_sequence))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sequence_array)
    onehotencoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_sequence = onehotencoder.fit_transform(integer_encoded)
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

def processdata():
    encoded_sequences = []
    folder = os.path.join("datasets", "mutations")
    files = list(glob.glob(os.path.join(folder, "*.fasta")))

    for file in files:
        print(f"Processing {file}")
        fasta_init = pyfastx.Fastx(file)
        for seq in fasta_init:
            encoded_sequence = onehotencoder(seq, max_length = 500)
            encoded_sequences.append(encoded_sequence)
    path = os.path.join("datasets/processeddata", "encoded_sequences.npy")
    encoded_sequences = np.array(encoded_sequences)
    np.save(path, encoded_sequences)
processdata()


