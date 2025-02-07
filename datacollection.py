#need to incorporate more datasets, could test with this much right now
import os
import time
import re
from Bio import Entrez, SeqIO


Entrez.email = "krishaysingh2022@gmail.com"
Entrez.api_key = "cb1c8ecbec68818042c259903f4aa5b7f908"

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
control_output_directory = "datasets/sequences/control"
ensure_directory(control_output_directory)
if not os.path.exists(os.path.join(control_output_directory, "NC_000012.12.fasta")):
    
    control_handle = Entrez.efetch(db="nucleotide", id="NC_000012.12", rettype="fasta", retmode="text")
    control_record = SeqIO.read(control_handle, "fasta")
    control_handle.close()

    control_output_file = os.path.join(control_output_directory, "NC_000012.12.fasta")
    SeqIO.write(control_record, control_output_file, "fasta")
    print(f"Control sequence saved")
    
else:
    print("Control exists")

def mutation_continuous_fetch():
    search_terms = [
        "CRY1[Gene] AND variant[All Fields] AND Homo sapiens[Organism]",
    ]
    
    cry1_output_directory = "datasets/sequences/mutations"
    ensure_directory(cry1_output_directory)
    repeat = 0
    while True:
        
        for term in search_terms:
            
            handle = Entrez.esearch(db="nucleotide", term=term, retmax=100)  
            record = Entrez.read(handle)
            handle.close()
            
           
            if not record["IdList"]:
                print(f"No results found for: {term}")
                continue
            repeats = 0
            while repeats<10:
             
                for sequence_id in record["IdList"]:
                    
                    print(f"Sequence ID: {sequence_id}")
                    # Fetch the sequence record
                    handle = Entrez.efetch(db="nucleotide", id=sequence_id, rettype="fasta", retmode="text")
                    seq_record = SeqIO.read(handle, "fasta")
                    handle.close()

                    # Save the sequence
                    ref_name = seq_record.id
                    ref_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ref_name)
                    
                    filename = f"{ref_name}.fasta"
                    output_path = os.path.join(cry1_output_directory, filename)
                    if os.path.exists(output_path):
                        print(f"Sequence {ref_name} already exists")
                        repeat+=1
                        if repeat>= 10:
                            print("No more unique cases")
                            return
                        continue
                    try:
                        with open(output_path, "w") as output_file:
                            SeqIO.write(seq_record, output_file, "fasta")
                        print(f"Sequence saved to {output_path}")
                    except Exception as e:
                        print(f"Error saving sequence {ref_name}: {e}")
                        os.remove(output_path)
                time.sleep(2)  
        
                break


mutation_continuous_fetch()
