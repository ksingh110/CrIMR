import numpy as np

# Load the .npz file
file_path = "E:\\datasets\\processeddata\\AUGMENTED_DATA.npz"
  # Replace with your actual file path
npz_data = np.load(file_path, allow_pickle=True)

# Print the keys in the .npz file
print("Keys in the .npz file:", npz_data.files)

# Print the shape of the arrays under each key
for key in npz_data.files:
    print(f"Array under key '{key}':")
    print(npz_data[key])  # Print the array for the current key
    print(f"Shape: {npz_data[key].shape}")  # Print the shape of the current array
    print()  # Print an empty line for readability
