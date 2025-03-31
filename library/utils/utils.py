import ast

import numpy as np


def convert_dict_string_to_dict(dict_string):
    """
    Convert a string representation of a dictionary containing numpy arrays back to a dictionary.

    Args:
        dict_string (str): String representation of the dictionary

    Returns:
        dict: Converted dictionary with numpy arrays
    """
    # First, safely evaluate the string to get the basic structure
    try:
        dict_string = dict_string.replace("array(", "").replace(")", "")
        raw_dict = ast.literal_eval(dict_string)

        # Convert the lists back to numpy arrays
        converted_dict = {}
        for key, value in raw_dict.items():
            converted_dict[key] = np.array(value)

        return converted_dict
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error converting dictionary string: {str(e)}")


def dataframe_to_list_of_lists(df):
    """
    Convert a DataFrame to a list of lists.
    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        list: List of lists, where each inner list is a row from the DataFrame
    """
    X_train_list = list(df)
    X_train_list = [list(str(peptide)) for peptide in X_train_list]
    return X_train_list
