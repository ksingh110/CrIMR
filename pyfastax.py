import numpy as np
import pandas as pd

# Load the .npy file
data = np.load('datasets\processeddata\encoded_sequences.npy', allow_pickle=True)

# Convert to DataFrame (if the data is a 2D array)
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('encoded_sequences.csv', index=False)

print("Saved as encoded_sequences.csv")