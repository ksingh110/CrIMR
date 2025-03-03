import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def augment_dna(sequence, copies):
    """Generates augmented DNA sequences."""
    return [SeqRecord(sequence.seq, id=f"{sequence.id}_augmented_{i+1}", description="Augmented DNA")
            for i in range(copies)]

# Define input/output file paths
input_file = "E:\\cry1controlsequences.gz"  # Set actual input file path
output_file = "E:\\augmented_cry1controlsequences.gz"  # Set actual output file path

aug_copies = 1000  # Number of augmented copies

# Read input file
with gzip.open(input_file, "rt") as input_handle:
    sequences = list(SeqIO.parse(input_handle, "fasta"))

# Augment sequences
augmented_sequences = []
for sequence in sequences:
    augmented_sequences.extend(augment_dna(sequence, aug_copies))

# Write augmented sequences to output file
with gzip.open(output_file, "wt") as output_handle:
    SeqIO.write(augmented_sequences, output_handle, "fasta")

print(f"Augmented sequences saved to {output_file}")