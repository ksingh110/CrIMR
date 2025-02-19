import pandas as pd
import requests

# The function to get the CRY1 gene sequence code
def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107093549")
    if response.status_code == 200:
        return response.json().get("dna", "").upper()
    else:
        return None

# Function to add CRY1 mutation to the regular CRY1 sequence
def add_mutation(seq, chromnum, pos, refnuc, altnuc):
    seq_start = pos - 1
    seq_end = seq_start + len(refnuc)
    if seq[seq_start:seq_end] != refnuc:
        print("The positions of the sequences do not match, or the sequence does not exist.")
        return None
    else:
        mutated_seq = seq[:seq_start] + altnuc + seq[seq_end:]
        return mutated_seq

# Read the CSV file and store it in a variable
csv = "C:\\Users\\srivi\\Downloads\\gnomAD_v4.1.0_ENSG00000008405_2025_02_16_20_37_32gnomAD_v4.1.csv"
df = pd.read_csv(csv, names=['A'], header=None)

# Get the CRY1 sequence
cry1_seq = get_CRY1_gene()

if cry1_seq:
    with open("CRY1_mutations.fasta", "w") as fasta_file:
        for index, row in df.iterrows():
            gnomAD_ID = row["A"]
            print("gnomAD_ID: {gnomAD_ID}")  
            parts = gnomAD_ID.split("-")
            if len(parts) == 4:
                chromnum, pos, refnuc, altnuc = parts
                pos = int(pos)
                
                mutated_sequence = add_mutation(cry1_seq, chromnum, pos, refnuc, altnuc)
                if mutated_sequence:
                    fasta_entry = f">CRY1_mutated {chromnum}:{pos} {refnuc}>{altnuc}\n{mutated_sequence}\n"
                    fasta_file.write(fasta_entry)
    print("FASTA file: CRY1_mutations.fasta")
else:
    print("Could not get CRY1 gene sequence")
