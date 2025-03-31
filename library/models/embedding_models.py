import datetime
import json
import os
import time
import typing

import pandas as pd
import tensorflow as tf

from library.models import encodings, multimodels
from library.utils import data_generator, early_stopping, evaluation, utils
from library.utils.save_results import save_results


def run_embed(
    alphabet: typing.List[str],
    hyper_space_name: str,
    folder_for_configurations: str,
    encoding: str,
    n_setups: int,
    n_setups_parallel: int,
    batch_normalisation: bool,
    model_name: str,
):
    """Function to run the model with different hyperparameters and save the results.
    In this case the encoding is done with the embedding layer.
    Args:
        alphabet (list): List of amino acids.
        hyper_space_name (str): Name of the hyperparameter space file.
        folder_for_configurations (str): Folder to save the configurations.
        encoding (str): Encoding type ('ohe', 'blosum', 'handcrafted').
        n_setups (int): Number of setups.
        n_setups_parallel (int): Number of setups to run in parallel.
        batch_normalisation (bool): Whether to use batch normalization.
        model_name (str): Name of the model ('mlp', 'cnn', 'gru').

        Steps that are taken in the function:
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
    os.makedirs(
        "configurations/" + folder_for_configurations + "/histories", exist_ok=True
    )

    ##2. Encoding specific elements and creating the dictionary for the encoding##
    matrix = list(range(1, len(alphabet) + 1))
    dict_embed = {}
    for i in range(len(alphabet)):
        dict_embed[alphabet[i]] = matrix[:][i]
    dict_encoding = dict_embed
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

    xs_train = []
    ys_train = []
    xs_val = []
    ys_val = []
    for setup_n in range(1, n_setups_parallel + 1):
        df_train = pd.read_csv(f"..//data/setup-{setup_n}/train.csv")

        encoded_sequences_train = utils.dataframe_to_list_of_lists(df_train["sequence"])
        encoded_sequences_train = encodings.embed_prep(
            encoded_sequences_train,
            alphabet=alphabet,
            max_sequence_length=max_sequence_length,
            dict_embed=dict_embed,
        )

        xs_train.append(encoded_sequences_train)
        ys_train.append(df_train["label"])

        df_val = pd.read_csv(f"..//data/setup-{setup_n}/val.csv")

        encoded_sequences_val = utils.dataframe_to_list_of_lists(df_val["sequence"])
        encoded_sequences_val = encodings.embed_prep(
            encoded_sequences_val,
            alphabet=alphabet,
            max_sequence_length=max_sequence_length,
            dict_embed=dict_embed,
        )

        xs_val.append(encoded_sequences_val)
        ys_val.append(df_val["label"])

    ##3. Fit the model with the hyperparameters##
    for row_n in indexes:
        row = hyper_space.iloc[row_n]
        t = time.time()

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
            embedding_layer=True,
            len_encoding=len(alphabet) + 1,
        )

        ##3b. Save the information on the configuration in a json file##
        json_object = json.dumps([dictionary], indent=4)
        filename = (
            "configurations/"
            + folder_for_configurations
            + "/configuration"
            + str(row_n)
            + ".json"
        )
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
        lens_train = [len(x) for x in xs_train]
        max_train_size = int(max(lens_train))

        lens_val = [len(x) for x in xs_val]
        min_val_size = int(min(lens_val))
        max_val_size = int(max(lens_val))

        data_generator_val = data_generator.DataGenerator(
            sequences=xs_val, batch_size=min_val_size, ys=ys_val, n_setups=len(xs_val)
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
