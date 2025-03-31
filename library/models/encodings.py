###encoding functions
import numpy as np
import pandas as pd
import tensorflow as tf

from library.utils import utils


def aa_list_to_tens(list_of_lists: list, max_sequence_length: int, dict_encoding: dict):
    """
    list of lists: expects a list of lists such as [[a, b,f],[c,d,e]]
    max_sequence_length: maximum sequence length occuring
    dict_encoding: dictionary that translates the amino acids in the alphabet
              to a a numpy array

    The function returns a tensorflow tensor. Unknown amino acids are represented
    by 0 vectors
    """
    for peptide_i in range(len(list_of_lists)):
        peptide = list_of_lists[peptide_i]
        # for aa_i in range(len(peptide)):
        for aa_i in range(len(peptide)):
            try:
                aa = peptide[aa_i]
                # print(aa)
                # print(dict_ohe[aa])
                list_of_lists[peptide_i][aa_i] = dict_encoding[aa]
                # print('lengte vector',len(dict_ohe[aa]))
            except:
                # print('not in dict or empty', np.zeros(len(alphabet)))
                list_of_lists[peptide_i][aa_i] = np.zeros(
                    len(list(dict_encoding.values())[0])
                )

    # Define the desired length of the padded arrays
    padded_length = len(list(dict_encoding.values())[0])
    padded_list = []
    for sublist in list_of_lists:
        # print(len(sublist))
        num_pads = max_sequence_length - len(sublist)
        pad = [np.zeros(padded_length) for _ in range(num_pads)]
        padded_sublist = sublist + pad
        padded_list.append(padded_sublist)
    # Convert the padded list to a TensorFlow tensor
    # print('lengthes',len(padded_list),len(padded_list[0]))

    tensor = tf.convert_to_tensor(padded_list)  # commented out for the moment
    # return padded_list
    # print('tensorshape',tensor.shape)
    return tensor


def embed_prep(
    list_of_lists: list, alphabet: list, max_sequence_length: int, dict_embed: dict
):
    """
    list of lists: expects a list of lists such as [[a, b,f],[c,d,e]]
    max_sequence_length: maximum sequence length occuring
    dict_embed: dictionary that translates the amino acids in the alphabet
                to an integer
    The function returns a tensorflow tensor. Unknown amino acids are represented
    by 0 vectors
    """
    for peptide_i in range(len(list_of_lists)):
        peptide = list_of_lists[peptide_i]
        # for aa_i in range(len(peptide)):
        for aa_i in range(len(peptide)):
            try:
                aa = peptide[aa_i]
                # print(aa)
                # print(dict_ohe[aa])
                list_of_lists[peptide_i][aa_i] = dict_embed[aa]
            except:
                # print('not in dict or empty', np.zeros(len(alphabet)))
                list_of_lists[peptide_i][aa_i] = 0

    # Define the desired length of the padded arrays
    padded_length = len(alphabet)
    padded_list = []
    for sublist in list_of_lists:
        # print(len(sublist))
        num_pads = max_sequence_length - len(sublist)
        pad = [0 for _ in range(num_pads)]
        padded_sublist = sublist + pad
        padded_list.append(padded_sublist)
    # Convert the padded list to a TensorFlow tensor
    tensor = tf.convert_to_tensor(padded_list)  # commented out for the moment
    # return padded_list
    return tensor


##Create dictionary##
def create_dict_ohe(alphabet):
    # CREATE A DICTIONARY FOR YOUR OHE's
    matrix = np.identity(len(alphabet))
    dict_ohe = {}
    for i in range(len(alphabet)):
        dict_ohe[alphabet[i]] = matrix[:][i]
    return dict_ohe


def preprocess_data(
    data_path: str,
    n_setups_parallel: int,
    dict_encoding: dict,
):
    """
    Preprocesses the data for the model training.
    Args:
        data_path (str): Path to the data directory.
        n_setups_parallel (int): Number of setups to process in parallel.
        dict_encoding (dict): Dictionary for encoding amino acids.
    Returns:
        xs_train (list): List of training sequences.
        ys_train (list): List of training labels.
        xs_val (list): List of validation sequences.
        ys_val (list): List of validation labels.
    """
    xs_train, ys_train = [], []
    xs_val, ys_val = [], []
    for setup_n in range(1, n_setups_parallel + 1):
        df_train = pd.read_csv(f"{data_path}-{setup_n}/train.csv")

        encoded_sequences_train = utils.dataframe_to_list_of_lists(df_train["sequence"])
        encoded_sequences_train = aa_list_to_tens(
            list_of_lists=encoded_sequences_train,
            max_sequence_length=15,
            dict_encoding=dict_encoding,
        )

        xs_train.append(encoded_sequences_train)
        ys_train.append(df_train["label"])

        df_val = pd.read_csv(f"{data_path}-{setup_n}/val.csv")

        encoded_sequences_val = utils.dataframe_to_list_of_lists(df_val["sequence"])
        encoded_sequences_val = aa_list_to_tens(
            list_of_lists=encoded_sequences_val,
            max_sequence_length=15,
            dict_encoding=dict_encoding,
        )

        xs_val.append(encoded_sequences_val)
        ys_val.append(df_val["label"])
    return xs_train, ys_train, xs_val, ys_val
