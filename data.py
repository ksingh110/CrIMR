import os
import time, re
from Bio import Entrez, SeqIO

# Provide your email and API key for NCBI
Entrez.email = "krishaysingh2022@gmail.com"
Entrez.api_key = "cb1c8ecbec68818042c259903f4aa5b7f908"

# Function to ensure the directory exists
def ensure_directory(directory):
    """
    Creates the directory if it doesn't already exist.
    
    Args:
        directory (str): The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_sequence(search_term, output_directory, database="nucleotide", file_format="fasta"):
    """
    Downloads the first GenBank record matching the search term and saves it in the specified format.
    
    Args:
        search_term (str): Query for NCBI Entrez search.
        output_directory (str): Full path (including filename) to save the sequence.
        database (str): The NCBI database to search. Defaults to "nucleotide".
        file_format (str): The format to save the sequence (fasta or genbank). Defaults to "fasta".
    """
    print(f"Searching for: {search_term}")
    
    # Ensure the output directory exists
    ensure_directory(output_directory)

    # Search for the sequence in the specified database
    handle = Entrez.esearch(db=database, term=search_term, retmax=1)
    record = Entrez.read(handle)
    handle.close()
    
    # Get the first sequence ID from the search results
    if not record["IdList"]:
        print(f"No results found for: {search_term}")
        return
    sequence_id = record["IdList"][0]
    print(f"Found sequence with ID: {sequence_id}")
    
    # Fetch the sequence record
    handle = Entrez.efetch(db=database, id=sequence_id, rettype="gb", retmode="text")
    seq_record = SeqIO.read(handle, "genbank")
    handle.close()
    
    # Use the accession number (reference) as the filename
    ref_name = seq_record.id  # This is the GenBank accession number
    
    # Sanitize the filename by removing any characters not allowed in file names
    ref_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ref_name)
    
    filename = f"{ref_name}.{file_format}"  # Create a filename using the reference ID
    output_path = os.path.join(output_directory, filename)
    
    # Save the sequence to a file in the specified format
    try:
        with open(output_path, "w") as output_file:
            SeqIO.write(seq_record, output_file, file_format)
        print(f"Sequence saved to {output_path}")
    except Exception as e:
        print(f"Error saving sequence {ref_name}: {e}")

# Part 2: Control Sequence Download (FASTA format)
control_output_directory = "datasets/sequences/control"
ensure_directory(control_output_directory)

# Fetch and save control sequence in FASTA format
control_handle = Entrez.efetch(db="nucleotide", id="NC_000012.12", rettype="fasta", retmode="text")
control_record = SeqIO.read(control_handle, "fasta")
control_handle.close()

control_output_file = os.path.join(control_output_directory, "NC_000012.12.fasta")
SeqIO.write(control_record, control_output_file, "fasta")
print(f"Control sequence saved to {control_output_file}")
def continuous_fetch():
    search_terms = [
        "CRY1[Gene] AND variant[All Fields] AND Homo sapiens[Organism]",
    ]
    
    # Folder to save downloaded sequences
    cry1_output_directory = "datasets/sequences/mutations"
    ensure_directory(cry1_output_directory)

    control_output_directory = "datasets/sequences/control"
    ensure_directory(control_output_directory)
    
    # Infinite loop to continuously fetch and save sequences
    while True:
        for term in search_terms:
            print(f"Fetching for search term: {term}")
            
            # Increase retmax to fetch more than one result at a time
            handle = Entrez.esearch(db="nucleotide", term=term, retmax=10)  # Change retmax to a higher value
            record = Entrez.read(handle)
            handle.close()
            
            # Ensure there are results
            if not record["IdList"]:
                print(f"No results found for: {term}")
                continue
            
            # Iterate over all sequence IDs retrieved
            for sequence_id in record["IdList"]:
                print(f"Found sequence with ID: {sequence_id}")
                # Fetch the sequence record
                handle = Entrez.efetch(db="nucleotide", id=sequence_id, rettype="gb", retmode="text")
                seq_record = SeqIO.read(handle, "genbank")
                handle.close()

                # Save the sequence
                ref_name = seq_record.id
                ref_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ref_name)  # Sanitize the filename
                
                filename = f"{ref_name}.fasta"
                output_path = os.path.join(cry1_output_directory, filename)
                
                try:
                    with open(output_path, "w") as output_file:
                        SeqIO.write(seq_record, output_file, "fasta")
                    print(f"Sequence saved to {output_path}")
                except Exception as e:
                    print(f"Error saving sequence {ref_name}: {e}")
            
            # Control delay between fetches
            time.sleep(5)  # Adjust delay time as needed
            print('Completed fetching for search term')

# Start continuous fetching
continuous_fetch()


# Start continuous fetching
continuous_fetch()
