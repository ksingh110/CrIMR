import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
#Krishay add
def augument_dna (sequence, copies):
    return [SeqRecord(sequence.seq, id=f"{sequence.id}_augmented_{i+1}", description="Augumented DNA")
        for i in range (copies)]
aug_copies = 1000 #Add the number of sequences needed
#Krishay add this
input_file =""
ouput_file = ""
with gzip.open('input_file','rt') as input_handle:
    sequences = list(SeqIO.parse(input_handle, "fasta"))
augument_dna = []
for sequences in sequences: 
    augument_dna.extend(augument_dna(sequences ,aug_copies))
with gzip.open('output_file','wt') as output_handle:
    sequences = list(SeqIO.parse(output_handle, "fasta"))
