import pandas as pd
import numpy as np
import gzip

# Specify the path to the .gz file
gz_file_path = "E:\\datasets\\processeddata\\augmentedcontrolencoded.gz"

# Load the .gz file
with gzip.open(gz_file_path, 'rb') as handle:
    loaded_sequences = np.load(handle)

# Print the shape and some sample data
print(loaded_sequences.shape)  # Shape of the array
print(loaded_sequences[:5]) 