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
    seq_start = pos - 106991364
    seq_end = seq_start + len(refnuc)
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
df = pd.read_csv(csv, names=['H'], header=None)

# Get the CRY1 sequence
cry1_seq = get_CRY1_gene()

if cry1_seq:
    with open("CRY1_mutations.fasta", "w") as fasta_file:
        for index, row in df.iterrows():
            gnomAD_ID = row["H"]
            print(f"gnomAD_ID: {gnomAD_ID}")  # Corrected print statement
            parts = str(gnomAD_ID).split("-")
            if len(parts) == 4:
                chromnum, pos, refnuc, altnuc = parts
            
                pos = int(pos)
                
                # Make sure chromnum is correctly formatted (if needed)
                # For instance, ensure it starts with "chr" or adjust as required
                
                mutated_sequence = add_mutation(cry1_seq, chromnum, pos, refnuc, altnuc)
                if mutated_sequence:
                    fasta_entry = f">CRY1_mutated {chromnum}:{pos} {refnuc}>{altnuc}\n{mutated_sequence}\n"
                    fasta_file.write(fasta_entry)
    print("FASTA file: CRY1_mutations.fasta")
else:
    print("Could not get CRY1 gene sequence")