import pickle

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from library.models.encodings import aa_list_to_tens, embed_prep
from library.utils.utils import dataframe_to_list_of_lists


def compute_permutation_multi_ensemble(models, X_trains):
    """Computes the permutation for a given model and training data.

    Args:
      model: A Keras model.
      X_train: A NumPy array of training data.

    Returns:
      A NumPy array of permutation values, with shape (n_features, n_outputs).
    """

    y_preds = []
    for model, X_train in zip(models, X_trains):
        y_pred = model.predict(X_train)
        y_preds.append(y_pred)

    permutation_ensemble = np.zeros((X_trains[0].shape[1], y_preds[0].shape[1]))
    permutation_mlp = np.zeros((X_trains[0].shape[1], y_preds[0].shape[1]))
    permutation_cnn = np.zeros((X_trains[1].shape[1], y_preds[1].shape[1]))
    permutation_gru = np.zeros((X_trains[2].shape[1], y_preds[2].shape[1]))

    n_repeats = 10
    for repetition in range(n_repeats):
        for output_idx in range(y_preds[0].shape[1]):
            for feature_idx in range(X_trains[0].shape[1]):
                X_train_perms = []

                # Generate a single permutation for the feature indices
                permuted_indices = np.random.permutation(int(X_trains[0].shape[0]))

                for X_train in X_trains:
                    X_train_perm = tf.identity(X_train)  # Create a copy of the original
                    X_train_perm = np.array(X_train_perm)

                    X_train_perm[:, feature_idx] = X_train_perm[
                        permuted_indices, feature_idx
                    ]

                    X_train_perms.append(X_train_perm)
                y_preds_perm = []
                for model, X_train_perm in zip(models, X_train_perms):
                    y_pred_perm = model.predict(X_train_perm)
                    y_preds_perm.append(y_pred_perm)
                # Compute the change in predictions due to the permutation of the feature.
                delta_y_preds = []
                for y_pred, y_pred_perm in zip(y_preds, y_preds_perm):
                    delta_y_pred = y_pred - y_pred_perm
                    delta_y_preds.append(delta_y_pred)
                delta_y_preds_ensemble = np.mean(delta_y_preds, axis=0)

                permutation_ensemble[feature_idx, output_idx] += np.sum(
                    delta_y_preds_ensemble[:, output_idx] ** 2
                )
                permutation_mlp[feature_idx, output_idx] += np.sum(
                    delta_y_preds[0][:, output_idx] ** 2
                )
                permutation_cnn[feature_idx, output_idx] += np.sum(
                    delta_y_preds[1][:, output_idx] ** 2
                )
                permutation_gru[feature_idx, output_idx] += np.sum(
                    delta_y_preds[2][:, output_idx] ** 2
                )
    # Normalize the permutation by the number of training examples.
    permutation_ensemble /= int(int(y_preds[0].shape[0]) * n_repeats)
    permutation_mlp /= int(int(y_preds[0].shape[0]) * n_repeats)
    permutation_cnn /= int(int(y_preds[1].shape[0]) * n_repeats)
    permutation_gru /= int(int(y_preds[2].shape[0]) * n_repeats)
    return permutation_ensemble, permutation_mlp, permutation_cnn, permutation_gru


tf.random.set_seed(42)


model_mlp = keras.models.load_model(
    "../../models/single_model_lab_results_26_okt_factor_1178.h5"
)
model_cnn = keras.models.load_model(
    "../../models/model_lab_30_oktober1random_seed4269.h5"
)
model_gru = keras.models.load_model(
    "../../models/gru_blos_BN_False4_sep_with_end_result_lab.h5"
)
max_sequence_length = 15
alphabet = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "s",
    "t",
]
matrix = list(range(1, len(alphabet) + 1))
dict_embed = {}
for i in range(len(alphabet)):
    dict_embed[alphabet[i]] = matrix[:][i]


dict_encoding = dict_embed


xs_train = []
ys_train = []
xs_val = []
ys_val = []

data_path = "../../data/setup-"

setup_n = 1
df_train = pd.read_csv(data_path + str(setup_n) + "/train.csv")

df_val = pd.read_csv(data_path + str(setup_n) + "/val.csv")
df_test = pd.read_csv(data_path + str(setup_n) + "/test.csv")

df_train = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
encoded_sequences_train = dataframe_to_list_of_lists(df_train["sequence"])
encoded_sequences_train = embed_prep(
    encoded_sequences_train,
    alphabet=alphabet,
    max_sequence_length=max_sequence_length,
    dict_embed=dict_embed,
)

xs_train.append(encoded_sequences_train)
ys_train.append(df_train["label"])

xs_train_mlp = xs_train
# %% COmpute permutation for blosum

with open(
    "../../library/models/dict_encodings/blosom_dict.pkl",
    "rb",
) as f:
    dict_encoding = pickle.load(f)


xs_train = []
ys_train = []
xs_val = []
ys_val = []
setup_n = 1
df_train = pd.read_csv(data_path + str(setup_n) + "/train.csv")

df_val = pd.read_csv(data_path + str(setup_n) + "/val.csv")
df_test = pd.read_csv(data_path + str(setup_n) + "/test.csv")

df_train = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
encoded_sequences_train = dataframe_to_list_of_lists(df_train["sequence"])
encoded_sequences_train = aa_list_to_tens(
    list_of_lists=encoded_sequences_train,
    max_sequence_length=15,
    dict_encoding=dict_encoding,
)

xs_train.append(encoded_sequences_train)
ys_train.append(df_train["label"])


models = [model_mlp, model_cnn, model_gru]
X_trains = [xs_train_mlp[0], xs_train[0], xs_train[0]]
(
    permutation_ensemble_train,
    permutation_mlp_train,
    permutation_cnn_train,
    permutation_gru_train,
) = compute_permutation_multi_ensemble(models, X_trains)


importance_df = pd.DataFrame(index=range(15), columns=["MLP", "CNN", "GRU", "Ensemble"])
importance_df["MLP"] = permutation_mlp_train.mean(axis=1)
importance_df["CNN"] = permutation_cnn_train.mean(axis=1)
importance_df["GRU"] = permutation_gru_train.mean(axis=1)
importance_df["Ensemble"] = permutation_ensemble_train.mean(axis=1)

importance_df.to_csv("permutation.csv")
