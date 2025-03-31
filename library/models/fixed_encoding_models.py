################IMPORT STATEMENTS ######################################################
import datetime
import json
import os
import pickle
import time
import typing

import pandas as pd
import tensorflow as tf

from library.models import encodings, multimodels
from library.models.encodings import create_dict_ohe
from library.utils import data_generator, early_stopping, evaluation
from library.utils.save_results import save_results


##MAIN RUNNING OF EXPERIMENT##
def run_model_ohe_blosum_handcrafted(
    alphabet: typing.List[str],
    hyper_space_name: str,
    folder_for_configurations: str,
    encoding: str,
    n_setups: int,
    n_setups_parallel: int,
    batch_normalisation: bool,
    model_name: str,
):
    """
    Function to run the model with different hyperparameters and save the results.
    In this case the model is a fixed encoding model, which means that the encoding
    of the amino acids is fixed and not learned during training: one-hot encoding,
    blosum62 or handcrafted.
    Args:
        alphabet (list): List of amino acids.
        hyper_space_name (str): Name of the hyperparameter space file.
        folder_for_configurations (str): Folder to save the configurations.
        encoding (str): Encoding type ('ohe', 'blosum', 'handcrafted').
        n_setups (int): Number of setups.
        n_setups_parallel (int): Number of setups to run in parallel.
        batch_normalisation (bool): Whether to use batch normalization.
        model_name (str): Name of the model ('mlp', 'cnn', 'gru').

    Steps in this function:
        1. Create a folder for the configurations and hyperparameters.
        2. Encode specific elements and create the dictionary for the encoding.
        3. Fit the model with the hyperparameters.
        4. Evaluate the model.
        5. Save the history of the model.
        6. Save the results in the results file.

    """
    tf.random.set_seed(42)
    ##1. Creating making folder for the configurations and hyperparameters##
    hyper_space = pd.read_csv(hyper_space_name)

    results_file = "configurations/" + folder_for_configurations + "/results.csv"
    ## Check if the results file already exists, if not create it ##
    try:
        results_df = pd.read_csv(
            results_file
        )  # check if the results file was already created
        rows_without_val_f1 = results_df[
            results_df["val_f1"].isnull() | results_df["val_f1"].eq("")
        ]
        indexes = rows_without_val_f1.index.astype(int).tolist()
    except:
        results_df = hyper_space.copy(deep=True)
        indexes = hyper_space.index
    max_sequence_length = 15

    ##CREATING THE RIGHT FOLDERS:
    os.makedirs(
        "configurations/" + folder_for_configurations + "/histories", exist_ok=True
    )

    ##2. Encoding specific elements and creating the dictionary for the encoding##
    if encoding == "ohe":
        dict_encoding = create_dict_ohe(alphabet=alphabet)
    elif encoding == "blosum":
        with open(
            "../../library/shared_functions/dict_encodings/blosom_dict.pkl",
            "rb",
        ) as f:
            dict_encoding = pickle.load(f)
    elif encoding == "handcrafted":
        with open(
            "../../library/shared_functions/dict_encodings/dict_hand_craft_13_10_2023_extended_normed.pkl",
            "rb",
        ) as f:
            dict_encoding = pickle.load(f)
    with open(
        "configurations/" + folder_for_configurations + "/information_on_configuration",
        "w",
    ) as outfile:
        json_object = json.dumps(
            {
                "encoding": str(dict_encoding),
                "date": str(datetime.datetime.now()),
                "hyperfile": hyper_space_name,
                "folder_for_configurations": folder_for_configurations,
            },
            indent=4,
        )
        outfile.write(json_object)

    xs_train, ys_train, xs_val, ys_val = encodings.preprocess_data(
        data_path="../data/setup",
        n_setups_parallel=n_setups_parallel,
        dict_encoding=dict_encoding,
    )

    ##3. Fit the model with the hyperparameters##
    for row_n in indexes:
        row = hyper_space.iloc[row_n]

        t = time.time()

        filename = (
            "configurations/"
            + folder_for_configurations
            + "/configuration"
            + str(row_n)
            + ".json"
        )

        ## 3a. Model Specific Part ###
        dictionary, model = multimodels.build_model(
            model_name,
            row,
            dict_encoding,
            batch_normalisation,
            max_sequence_length,
            alphabet,
            n_setups,
            row_n,
            embedding_layer=False,
            len_encoding=len(list(dict_encoding.values())[0]),
        )

        ##3b. Save the information on the configuration in a json file##

        json_object = json.dumps([dictionary], indent=4)

        with open(filename, "w") as outfile:
            outfile.write(json_object)

        file = open(filename)
        json_s = json.load(file)

        ## 3c. Create the data generator for the training and validation set ##

        data_generator_train = data_generator.DataGenerator(
            sequences=xs_train,
            batch_size=int(row["fit_batch_size"]),
            ys=ys_train,
            n_setups=len(xs_train),
        )

        lens_val = [len(x) for x in xs_val]
        min_val_size = int(min(lens_val))

        data_generator_val = data_generator.DataGenerator(
            sequences=xs_val,
            batch_size=min_val_size,
            ys=ys_val,
            n_setups=len(xs_val),
        )

        custom_early_stopping = early_stopping.CustomEarlyStopping(
            patience=5, start_epoch=5, num_setups=len(xs_train), row_n=row_n
        )
        history = model.fit(
            data_generator_train,
            validation_data=data_generator_val,
            callbacks=[custom_early_stopping],
            epochs=int(row["fit_epochs"]),
            verbose=0,
        )

        ##4. Evaluation of the model##
        inputs_train = evaluation.pad_for_evaluation(xs_train)
        predictions = model.predict(inputs_train)

        inputs_val = evaluation.pad_for_evaluation(xs_val)
        predictions_val = model.predict(inputs_val)

        threshold = 0.5
        for prediction, y_values, prediction_val, y_val in zip(
            predictions, ys_train, predictions_val, ys_val
        ):
            results = {}

            name = "training"
            results_ = evaluation.performance_calculator_nan_proof(
                prediction, y_values, threshold
            )
            results[name] = results_
            results[name]["setup"] = len(json_s)

            name = "validation"
            results_ = evaluation.performance_calculator_nan_proof(
                prediction_val, y_val, threshold
            )
            results[name] = results_
            results[name]["setup"] = len(json_s)

            elapsed = time.time() - t
            results["time"] = elapsed
            results["epochs used"] = len(history.history["loss"])

            json_s.append(results)

            json_object = json.dumps(json_s, indent=4)
            with open(filename, "w") as outfile:
                outfile.write(json_object)

        ##5. Save the history of the model##
        hist_df = pd.DataFrame(history.history)

        hist_json_file = (
            "configurations/"
            + folder_for_configurations
            + "/histories/history"
            + str(row_n)
            + ".json"
        )

        try:
            his_till_now = json.load(hist_json_file)
            his_till_now.append(json.dumps(hist_df, indent=4))
            with open(hist_json_file, mode="w") as f:
                hist_df.to_json(f)
        except:
            with open(hist_json_file, mode="w") as f:
                hist_df.to_json(f)
        file = open(filename)
        json_s = json.load(file)

        averages = evaluation.averages_calculator(json_s, n_setups)
        json_s.append(averages)

        with open(filename, "w") as outfile:
            json.dump(json_s, outfile, indent=4)

        try:
            best = open(
                "configurations/"
                + folder_for_configurations
                + "/best_configuration.json"
            )
            jsons_best = json.load(best)

            if jsons_best[1]["averages"]["f1"] < json_s[-1]["averages"]["f1"]:
                with open(
                    "configurations/"
                    + folder_for_configurations
                    + "/best_configuration.json",
                    "w",
                ) as outfile:
                    json_best = json.dumps([dictionary, averages], indent=4)
                    outfile.write(json_best)
        except:
            with open(
                "configurations/"
                + folder_for_configurations
                + "/best_configuration.json",
                "w",
            ) as outfile:
                json_best = json.dumps([dictionary, averages], indent=4)
                outfile.write(json_best)

        ##6. Save the results in the results file##
        save_results(
            results_df,
            row_n,
            json_s,
            n_setups,
            results_file,
        )
