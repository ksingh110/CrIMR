import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def onehotencoder(fasta_sequence, max_length=102500):
    # Convert the sequence into an array of characters
    sequence_array = np.array(list(fasta_sequence))

    # Label encode the characters (converting characters to integers)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sequence_array).reshape(-1, 1)

    # One-hot encode the integers
    onehotencoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    onehot_sequence = onehotencoder.fit_transform(integer_encoded)

    # Padding/truncating the sequence to the desired length
    num_classes = onehot_sequence.shape[1]  # Number of features (bases)
    
    # Padding the sequence to max_length if necessary
    if onehot_sequence.shape[0] < max_length:
        padding = np.zeros((max_length - onehot_sequence.shape[0], num_classes))
        onehot_sequence = np.vstack([onehot_sequence, padding])
    else:
        onehot_sequence = onehot_sequence[:max_length, :]

    # Ensure the final output shape: (samples, time_steps, features)
    return onehot_sequence.reshape(1, max_length, num_classes)  # Shape (1, max_length, num_classes)


def process(sequence):
    # One-hot encode and reshape the sequence for LSTM input
    encoded_seq = onehotencoder(sequence)  # Shape will be (1, max_length, num_classes)

    # If LSTM expects a 2D input, you can flatten the input (this will remove the time dimension)
    # For example, if the LSTM expects (batch_size, features) you can flatten the time_steps
    # encoded_seq = encoded_seq.flatten()  # Shape (max_length * num_classes,)

    return encoded_seq.flatten()  # Now it's ready to pass into the LSTM
