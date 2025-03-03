import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import nlpaug.augmenter.char as nac
import random

# File paths
input_file = "E:\\cry1controlsequences (1).gz"
output_file = "E:\\datasets\\processeddata\\_augmentedcry1controlsequencesencoded.gz"

# Function to read FASTA sequences
def read_fasta(file_path):
    with gzip.open(file_path, "rt") as handle:
        sequences = list(SeqIO.parse(handle, "fasta"))
    return sequences

# Function to augment DNA sequence
def augment_sequence(sequence, mutation_type, aug):
    augmented_seqs = aug.augment(str(sequence.seq))  # Returns a list
    return [Seq(seq) for seq in augmented_seqs]  # Convert to Biopython Seq objects

# Function to write augmented sequences to FASTA
def write_fasta(sequences, output_file):
    with gzip.open(output_file, "wt") as handle:
        SeqIO.write(sequences, handle, "fasta")

# Read original sequences
sequences = read_fasta(input_file)

# Mutation types
type_mutation = ("insert", "substitute", "swap", "delete")

# Generate augmented sequences
augmented_sequences = []
num_augmentations_per_seq = 64  # Number of augmented sequences per input sequence

for seq in sequences:
    for _ in range(num_augmentations_per_seq):  # Augment each sequence multiple times
        mutation = random.choice(type_mutation)  # Randomly select a mutation type

        if mutation == "insert":
            aug = nac.RandomCharAug(action="insert", aug_char_p=0.1)
        elif mutation == "substitute":
            aug = nac.RandomCharAug(action="substitute", aug_char_p=0.1)
        elif mutation == "swap":
            aug = nac.RandomCharAug(action="swap", aug_char_p=0.1)
        elif mutation == "delete":
            aug = nac.RandomCharAug(action="delete", aug_char_p=0.1)

        augmented_seq_list = augment_sequence(seq, mutation, aug)  # Returns a list
        for augmented_seq in augmented_seq_list:
            augmented_sequences.append(SeqRecord(augmented_seq, id=f"{seq.id}_aug_{mutation}", description="Augmented DNA sequence"))

# Write to file
write_fasta(augmented_sequences, output_file)

print(f"Augmented DNA saved to '{output_file}'")
