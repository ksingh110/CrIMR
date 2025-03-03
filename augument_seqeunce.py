import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import nlpaug.augmenter.char as nac
import random

input_file = "FILE PATH"
output_file = "FILE PATH"
def read_fasta(file_path):
    with gzip.open(file_path, "rt") as handle:
        sequences = list(SeqIO.parse(handle, "fasta"))
    return sequences

def augment_sequence(sequence, mutation_type, aug):
    if mutation_type == "insert":
        augmented_seq = aug.augment(str(sequence.seq))
    elif mutation_type == "substitute":
        augmented_seq = aug.augment(str(sequence.seq))
    elif mutation_type == "swap":
        augmented_seq = aug.augment(str(sequence.seq))
    elif mutation_type == "delete":
        augmented_seq = aug.augment(str(sequence.seq))
    return augmented_seq

def write_fasta(sequences, output_file):
    with gzip.open(output_file, "wt") as handle:
        SeqIO.write(sequences, handle, "fasta")

sequences = read_fasta(input_file)
        
type_mutation = ("insert", "substitute", "swap", "delete")
mutation = random.choice(type_mutation)
nucleotides = ("A", "T", "G", "C")

augmented_sequences = []
for seq in sequences:
    mutation = random.choice(type_mutation)  # Randomly select a mutation type
    if mutation == "insert":
        aug = nac.RandomCharAug(action="insert", aug_char_p=0.1)
    elif mutation == "substitute":
        aug = nac.RandomCharAug(action="substitute", aug_char_p=0.1)
    elif mutation == "swap":
        aug = nac.RandomCharAug(action="swap", aug_char_p=0.1)
    elif mutation == "delete":
        aug = nac.RandomCharAug(action="delete", aug_char_p=0.1)

    augmented_seq = augment_sequence(seq, mutation, aug) 
    augmented_sequences.append(SeqRecord(augmented_seq, id=f"{seq.id}_augmented", description="Augmented DNA sequence"))

write_fasta(augmented_sequences, output_file)

print(f"Augmented DNA saved to '{output_file}'")