import time
from typing import List

import keras
import numpy as np
import tensorflow as tf


def create_dict_ohe(alphabet: list) -> dict:
    """
    Create a dictionary for one-hot encoding of amino acids.
    Each amino acid in the alphabet is represented as a one-hot encoded vector.
    """
    matrix = np.identity(len(alphabet))
    dict_ohe = {}
    for i in range(len(alphabet)):
        dict_ohe[alphabet[i]] = matrix[:][i]
    return dict_ohe


class DataGenerator(keras.utils.all_utils.Sequence):
    """
    Data generator for Keras models.
    This class is used to generate batches of data for training and validation.
    It supports shuffling the data at the end of each epoch and handles multiple setups.
    """

    def __init__(self, sequences: List, batch_size, ys, n_setups):
        self.sequences = sequences[:]
        self.actives = ys[:]

        def element_length(elem):
            return len(elem)

        # Find the maximum length using the max() function with the custom element_length function
        self.min_length = min(map(element_length, ys))
        self.max_length = max(map(element_length, ys))

        if batch_size > self.min_length:
            batch_size = self.min_length
            print(
                "Setting batch size to",
                batch_size,
                " since it is larger than the number of sequences",
            )
        self.time_start = time.time()
        self.batch_size = batch_size

    def on_epoch_end(self):
        n_setups = len(self.sequences)

        for setup_n in range(n_setups):
            input_tensor = self.sequences[setup_n]  # Shuffle each setup's sequences
            num_samples = input_tensor.shape[0]
            # Create indices for shuffling
            indices = tf.range(num_samples)
            # Shuffle the indices randomly
            shuffled_indices = tf.random.shuffle(indices)
            # Use the shuffled indices to shuffle the tensor
            shuffled_tensor = tf.gather(input_tensor, shuffled_indices)
            self.sequences[setup_n] = shuffled_tensor
            self.actives[setup_n] = tf.gather(self.actives[setup_n], shuffled_indices)

    def __getitem__(self, batch_idx):
        first_seq_idx = batch_idx * self.batch_size
        last_seq_idx = (batch_idx + 1) * self.batch_size
        inputs = []
        outputs = []

        for sequence, actives in zip(self.sequences, self.actives):
            batch_xs = sequence[first_seq_idx:last_seq_idx]

            batch_ys = actives[first_seq_idx:last_seq_idx]
            batch_ys = np.array(batch_ys)
            inputs.append(batch_xs)
            outputs.append(batch_ys)
        return inputs, outputs

    def __len__(self):
        return int(np.ceil(self.min_length / self.batch_size))
