import gzip
import random
import numpy as np
import nlpaug.augmenter.char as nac
import time
import requests
from Bio import SeqIO

# File paths
input_file = "E:\\cry1controlsequences (1).gz"
output_file = "E:\\datasets\\processeddata\\AUGMENTEDCONTROLWORKS.gz"
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
    augmented_seqs = aug.augment(sequence)  # Returns a list of sequences
    return augmented_seqs

# Save sequences as a compressed NumPy array
def write_gz_file(sequences, output_file):
    with gzip.open(output_file, "wb") as handle:
        np.save(handle, np.array(sequences))

# Mutation types
type_mutation = ("insert", "substitute", "swap", "delete")

# Read sequences
sequences = read_fasta(input_file)
augmented_sequences = []
batch = 200
num_augmentations_per_seq = 4000
total_saved = 0

original_seq = cry1
for _ in range(num_augmentations_per_seq):
    mutation = random.choice(type_mutation)

    if mutation == "insert":
        aug = nac.RandomCharAug(action="insert", aug_char_p=0.02)

    elif mutation == "swap":
        aug = nac.RandomCharAug(action="swap", aug_char_p=0.02)
    elif mutation == "delete":
        aug = nac.RandomCharAug(action="delete", aug_char_p=0.02)

    augmented_seq_list = augment_sequence(original_seq, aug)
    print(augmented_seq_list)
    print(f"Generated {len(augmented_seq_list)} augmented sequences.")

        # Debugging: Check before extending
    print(f"Before extending: {len(augmented_sequences)} stored sequences.")
    augmented_sequences.extend(augmented_seq_list)

        # Debugging: Check after extending
    print(f"After extending: {len(augmented_sequences)} stored sequences.")
        
        # Save in batches
    if len(augmented_sequences) >= batch:
        print(f"Saving {len(augmented_sequences)} sequences...")
        write_gz_file(augmented_sequences, output_file)
        total_saved += len(augmented_sequences)
        print(f"Total saved so far: {total_saved}")
            
        augmented_sequences = []  # Reset batch

# Save any remaining sequences
if augmented_sequences:
    print(f"Final save: {len(augmented_sequences)} sequences.")
    write_gz_file(augmented_sequences, output_file)
    total_saved += len(augmented_sequences)
    print(f"Final total saved: {total_saved}")
