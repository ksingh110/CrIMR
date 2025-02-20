import h5py
import numpy as np

# Load the saved data
file_path = "datasets/processeddata/encoded_mutation_sequences.h5"

with h5py.File(file_path, "r") as hf:
    encoded_sequences = hf["encoded_sequences"][:]  # Read the dataset into a NumPy array

print(f"Loaded data shape: {encoded_sequences.shape}")  # 