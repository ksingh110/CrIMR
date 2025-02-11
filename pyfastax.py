import pyfastx
import glob
import os

folder = os.path.join("datasets", "mutations")
files = list(glob.glob(os.path.join(folder, "*.fasta")))

for file in files:
    fasta_init = pyfastx.Fastx(file)
    for name, seq in fasta_init:
        print(name, seq)