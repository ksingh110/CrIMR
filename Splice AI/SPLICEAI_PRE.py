import pandas as pd
import re

# --- Step 1: Read CSV and Create VCF File ---

# Replace with your actual CSV file path containing your CRY1 mutation data
csv_file = "cry1_mutations(1).csv"
mutations = pd.read_csv(csv_file)

# Define VCF header
vcf_header = """##fileformat=VCFv4.2
##INFO=<ID=SPLICEAI,Number=.,Type=String,Description="SpliceAI scores">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"""

vcf_rows = []
for _, row in mutations.iterrows():
    # Replace these column names with those in your CSV file
    chrom = row['chromosome']  # e.g., "12" or "chr12"
    pos = row['startpos']      # Genomic position of the mutation
    ref = row['ref']           # Reference nucleotide
    alt = row['alt']           # Alternate nucleotide

    # Format a VCF row (using '.' for ID, QUAL, and FILTER fields)
    vcf_rows.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.")

# Output VCF file name (this is the input for SpliceAI)
input_vcf = "cry1_variants.vcf"
with open(input_vcf, "w") as f:
    f.write(vcf_header + "\n")
    f.write("\n".join(vcf_rows))

print(f"VCF file created: {input_vcf}")

# --- Step 2: Simulate SpliceAI Prediction in Python ---

# This function simulates SpliceAI score computation from mutation data (as a placeholder).
def simulate_spliceai_score(chrom, pos, ref, alt):
    # For demonstration purposes, we generate a random score between 0 and 1.
    # You can replace this with a real calculation if you have data or logic for SpliceAI-like scoring.
    import random
    return round(random.uniform(0, 1), 4)

# Simulate the SpliceAI score for each mutation and add to the DataFrame
mutations['SpliceAI_max'] = mutations.apply(
    lambda row: simulate_spliceai_score(row['chromosome'], row['startpos'], row['ref'], row['alt']),
    axis=1
)

# --- Step 3: Apply Hill Function for DSPS Prediction ---

def hill_function(spliceai_score, kd_base=1.0, n=2.0):
    kd_mutated = kd_base / (1 + spliceai_score)
    repression_probability = 1 / (1 + (spliceai_score / kd_mutated) ** n)
    return repression_probability

# Apply Hill function to compute DSPS probability
mutations['DSPS_probability'] = mutations['SpliceAI_max'].apply(hill_function)

# --- Step 4: Save the Results ---

# Save the results to a new CSV file
mutations.to_csv("cry1_spliceai_predictions.csv", index=False)

print("SpliceAI-like predictions and DSPS probabilities saved to cry1_spliceai_predictions.csv")
print(mutations.head())
