import requests
import numpy as np
import tensorflow as tf

# Load the RNN model
rnn_model = tf.keras.models.load_model("/Users/krishaysingh/Downloads/6000_5_if_new_best_model.keras")

# Define base encoding for DNA sequences (one-hot encoding)
base_encoding = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}

# Function to one-hot encode the sequence
def one_hot_encode_sequence(sequence):
    return [base_encoding.get(base, [0, 0, 0, 0]) for base in sequence]  # One-hot encoding

# Function to fetch the reference sequence from UCSC Genome Browser API
def get_CRY1_gene():
    response = requests.get("https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr12;start=106991364;end=107004364")
    if response.status_code == 200:
        return response.json().get("dna", "").upper()
    else:
        return None

# Function to find mutation locations by comparing the reference sequence and mutation sequence
def get_mutation_location(ref_sequence, mutation_sequence):
    # Ensure both sequences are the same length
    if len(ref_sequence) != len(mutation_sequence):
        raise ValueError("Reference and mutation sequences must have the same length")

    # Compare each base and find the mutation position
    mutation_positions = []
    mutations = []
    for i, (ref_base, mut_base) in enumerate(zip(ref_sequence, mutation_sequence)):
        if ref_base != mut_base:
            mutation_positions.append(i)
            mutations.append((ref_base, mut_base))  # Store ref and alt alleles at this position
    
    return mutation_positions, mutations  # Returns mutation positions and (ref, alt) pairs

# Function to predict mutation probability using the RNN
def mutation_prob(sequence):
    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    return rnn_model.predict(sequence)[0][0]  # Predict and return the mutation probability

# Hill function to calculate DSPD probability
def hill_function(spliceai_score, kd_base=1.0, n=2.0):
    kd_mutated = kd_base / (1 + spliceai_score)
    repression_probability = 1 / (1 + (spliceai_score / kd_mutated) ** n)
    return repression_probability

# Define the start and end positions for Intron 10
INTRON_10_START = 106991364
INTRON_10_END = 107004364

# Load mutation sequence from the .npz file
mutation_sequence_data = np.load("test1.npz", allow_pickle=True)
mutation_sequence = mutation_sequence_data['arr_0'][0]  # Adjust the key based on the file content

# Fetch the reference sequence for the CRY1 gene region from UCSC
ref_sequence = get_CRY1_gene()

# Check lengths of both sequences
print(f"Length of reference sequence: {len(ref_sequence)}")
print(f"Length of mutation sequence: {len(mutation_sequence)}")

# Check if lengths match
if len(ref_sequence) != len(mutation_sequence):
    # Handle length mismatch: truncate or pad the sequences
    min_length = min(len(ref_sequence), len(mutation_sequence))
    ref_sequence = ref_sequence[:min_length]  # Truncate reference sequence
    mutation_sequence = mutation_sequence[:min_length]  # Truncate mutation sequence
    print(f"Sequences have been truncated to length: {min_length}")

    # Alternatively, you could pad the shorter sequence, for example:
    # if len(ref_sequence) < len(mutation_sequence):
    #     mutation_sequence = mutation_sequence[:len(ref_sequence)]
    # else:
    #     ref_sequence = ref_sequence[:len(mutation_sequence)]

# Compare the reference sequence with the mutation sequence
mutation_positions, mutations = get_mutation_location(ref_sequence, mutation_sequence)

# Output the mutation positions and the corresponding reference and alternate alleles
for pos, (ref, alt) in zip(mutation_positions, mutations):
    print(f"Mutation at position {pos + INTRON_10_START}:")
    print(f"  Reference Allele: {ref}, Alternate Allele: {alt}")

    # Extract the corresponding substring for the mutation
    mutated_subsequence = mutation_sequence[pos]
    
    # One-hot encode the mutated sequence
    encoded_mutated_sequence = one_hot_encode_sequence(mutation_sequence)
    encoded_mutated_sequence = np.array(encoded_mutated_sequence, dtype=np.float32)  # Convert to numpy array
    
    # Predict mutation probability using the RNN (SpliceAI score)
    mutation_prob_value = mutation_prob(encoded_mutated_sequence)  # Pass the mutation sequence to the model
    print(f"Mutation Probability (SpliceAI score): {mutation_prob_value}")
    
    # Apply Hill function to calculate DSPD probability
    dsps_prob = hill_function(mutation_prob_value)
    print(f"DSPD Probability: {dsps_prob}")
