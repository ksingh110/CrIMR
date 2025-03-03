import os
import numpy as np
import pandas as pd
import requests
def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107093549")
    if response.status_code == 200:
        return response.json().get("dna", "").upper()
    else:
        return None
csv = "cry1realvariations (1).csv"
df = pd.read_csv(csv, usecols=['chromEnd', 'ref', 'alt', 'AF', 'genes', 'variation_type', '_displayName'])
df = df[df["variation_type"].str.contains("intron_variant", na=False, case=False)]
output_path = "E:\\datasets\processeddata\cry1mutationallelefrequency.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_path_id = "E:\\datasets\processeddata\cry1mutationid.npz"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cry1_seq = get_CRY1_gene()
allele_freq_array = []
genome_id_array = []
if cry1_seq:
        for index, row in df.iterrows():
            gnomAD_ID = row["_displayName"]
            genome_id_array.append(gnomAD_ID)
            print("Genome ID appended: ", gnomAD_ID)
            allelefreq = row["AF"]
            allele_freq_array.append(allelefreq)
            print("Allele frequency appended: ", allelefreq)
np.savez_compressed(output_path, np.array(allele_freq_array))
print("Allele frequency data saved to: ", output_path)
np.savez_compressed(output_path_id, np.array(genome_id_array))
print("Mutation ID Data saved to: ", output_path)
