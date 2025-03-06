import numpy as np

# Check for the integrity of the file and print keys


# Load the .npz file
file_path = "E:\\datasets\\processeddata\\AUGMENTEDCONTROLWORKStest31689.npz"  # Replace with your actual file path
npz_data = np.load(file_path, allow_pickle=True)

# Print the keys in the .npz file
print("Keys in the .npz file:", npz_data.files)
