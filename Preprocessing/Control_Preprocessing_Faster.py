import numpy as np
import os
import requests
from sklearn.preprocessing import OneHotEncoder

# Function to fetch CRY1 gene sequence from UCSC API
def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107093549")
    return response.json().get("dna", "").upper() if response.status_code == 200 else None

# Faster One-Hot Encoding with Precomputed Dictionary
def onehotencoder(fasta_sequence, max_length=102500):
    nucleotides = ['A', 'C', 'G', 'T']
    label_dict = {nucleotides[i]: i for i in range(len(nucleotides))}  # Precompute encoding
    integer_encoded = np.array([label_dict.get(nuc, 0) for nuc in fasta_sequence])  # Map nucleotides

    # One-hot encode manually
    onehot_sequence = np.zeros((max_length, len(nucleotides)), dtype=np.float32)
    onehot_sequence[np.arange(len(integer_encoded[:max_length])), integer_encoded[:max_length]] = 1.0

    return onehot_sequence.flatten()

# Optimized Augmentation with Vectorized Operations
def augment_sequence(seq, substitution_prob=0.33, deletion_prob=0.33, insertion_prob=0.33):
    seq_array = np.array(list(seq))  # Convert to NumPy array
    seq_len = len(seq_array)
    
    # Generate all probabilities at once for substitution, deletion, and insertion
    random_probs = np.random.rand(seq_len)
    
    # Substitution Mask (Vectorized)
    substitution_mask = random_probs < substitution_prob
    seq_array[substitution_mask] = np.random.choice(['A', 'G', 'C', 'T'], size=substitution_mask.sum())

    # Deletion Mask (Vectorized)
    deletion_mask = (random_probs >= substitution_prob) & (random_probs < substitution_prob + deletion_prob)
    seq_array = seq_array[~deletion_mask]  # Remove elements at deletion positions

    # Insertion: We use np.where to get insertion positions and perform insertions in batches
    insertion_indices = np.where(random_probs >= substitution_prob + deletion_prob)[0]
    insertions = []
    for idx in insertion_indices:
        if np.random.rand() < insertion_prob:
            insert_nucleotide = np.random.choice(['A', 'G', 'C', 'T'])
            insert_pos = min(idx, len(seq_array))  # Ensure the position is valid
            insertions.append((insert_pos, insert_nucleotide))

    # Perform all insertions at once after collecting them
    if insertions:
        insertions = sorted(insertions, reverse=True)  # Reverse to insert at the correct index
        for idx, nucleotide in insertions:
            seq_array = np.insert(seq_array, idx, nucleotide)

    return "".join(seq_array)

# Optimized Processing Function with Faster Batch Writing
def process_data_augmentation(cry1_seq, output_path, num_augmented_sequences=1000, batch_size=100):
    existing_data = []
    if os.path.exists(output_path):
        existing_data = np.load(output_path, allow_pickle=True)["arr_0"].tolist()

    seq_count = 0
    rows_save = []

    # Pre-allocate a buffer for augmented sequences
    augmented_sequences = np.empty((num_augmented_sequences, 102500 * 4), dtype=np.float32)

    while seq_count < num_augmented_sequences:
        augmented_seq = augment_sequence(cry1_seq)
        
        # Print the sequence number being processed
        print(f"Processing augmented sequence {seq_count + 1}/{num_augmented_sequences}...")

        if augmented_seq:
            encoded_seq = onehotencoder(augmented_seq)
            augmented_sequences[seq_count] = encoded_seq
            seq_count += 1

        # Save data in larger batches
        if seq_count % batch_size == 0:
            print(f"Saving {batch_size} sequences to {output_path}...")
            existing_data.extend(augmented_sequences[:seq_count])
            np.savez_compressed(output_path, arr_0=np.array(existing_data))
            rows_save = []  # Reset buffer

    # Final batch write
    if seq_count % batch_size != 0:
        print(f"Final save of {seq_count % batch_size} sequences.")
        existing_data.extend(augmented_sequences[:seq_count])
        np.savez_compressed(output_path, arr_0=np.array(existing_data))

    print(f"Processed {seq_count} augmented sequences and saved to {output_path}.")

# Define File Path for Output
output_file = "E:\\datasets\\processeddata\\AUGMENTED_DATA_TRAINING_20034.npz"

# Fetch CRY1 Sequence and Run Augmentation
cry1_seq = get_CRY1_gene()
if cry1_seq:
    process_data_augmentation(cry1_seq, output_file)
else:
    print("Failed to fetch CRY1 gene sequence.")
