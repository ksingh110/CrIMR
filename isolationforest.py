import requests
import csv
from time import sleep
import gzip

# Function to get the coordinates for an rsID from UCSC Table Browser
def get_coordinates_from_ucsc(rsid):
    url = f"http://api.genome.ucsc.edu/getData/track?genome=hg38;track=dbSNP;table=knownGene;name={rsid}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            chrom = data[0]['chrom']  # chromosome (e.g., chr1)
            start = data[0]['chromStart']  # start position
            end = data[0]['chromEnd']  # end position
            return chrom, start, end
        else:
            print(f"No data found for {rsid}")
            return None
    else:
        print(f"Error retrieving data for {rsid}")
        return None

# Function to fetch the sequence using UCSC getDNA endpoint
def get_sequence_from_ucsc(chrom, start, end):
    chromsome= "chr" + chrom;
    
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome};chrom=chr{chromsome};start={start};end={end}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text
        return data
    else:
        print(f"Error retrieving sequence for {chrom}:{start}-{end}")
        return None

# Function to save all sequences into one compressed file (gzip)
def save_sequences_in_gzip(csv_file, output_file):
    with gzip.open(output_file, 'wt', encoding='utf-8') as fasta_file:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                rsid = row[0]  # Assuming rsID is in the first column of CSV
                print(f"Processing {rsid}...")
                
                # Step 1: Get coordinates for rsID
                coordinates = get_coordinates_from_ucsc(rsid)
                if coordinates:
                    chrom, start, end = coordinates
                    
                    # Step 2: Get sequence for those coordinates
                    sequence = get_sequence_from_ucsc(chrom, start, end)
                    
                    if sequence:
                        # Write sequence in FASTA format to the gzip file
                        fasta_header = f"> {rsid} {chrom}:{start}-{end}\n"
                        fasta_file.write(fasta_header)
                        fasta_file.write(sequence)
                        fasta_file.write("\n")
                    else:
                        print(f"Failed to get sequence for {rsid}")
                
                # Sleep to avoid hitting UCSC's API rate limits
                sleep(1)

# Run the extraction for a CSV file containing rsIDs and save in gzip
save_sequences_in_gzip(r"c:\Users\rashm\Downloads\gnomAD_v4.1.0_ENSG00000008405_2025_02_17_20_53_57.csv", "allsequences.fasta.gz")