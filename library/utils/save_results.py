import pandas as pd


def save_results(
    results_df: pd.DataFrame, row_n: int, json_s: list, n_setups: int, results_file: str
):
    """
    Updates the results DataFrame with evaluation metrics and saves it to a CSV file.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results.
        row_n (int): Row index in the DataFrame corresponding to the current setup.
        json_s (list): JSON data containing evaluation results.
        n_setups (int): Number of setups used in training.
        results_file (str): Path to the results CSV file.
    """
    averages = json_s[n_setups + 1]["averages"]
    stds = json_s[n_setups + 1]["stds"]

    results_df.loc[row_n, "val_auc"] = averages.get("auc", None)
    results_df.loc[row_n, "std_val_auc"] = stds.get("std_auc", None)
    results_df.loc[row_n, "val_f1"] = averages.get("f1", None)
    results_df.loc[row_n, "std_val_f1"] = stds.get("std_f1", None)
    results_df.loc[row_n, "val_accuracy"] = averages.get("accuracy", None)
    results_df.loc[row_n, "std_val_accuracy"] = stds.get("std_acc", None)
    results_df.loc[row_n, "val_precision"] = averages.get("precision", None)
    results_df.loc[row_n, "std_val_precision"] = stds.get("std_prec", None)
    results_df.loc[row_n, "val_recall"] = averages.get("recall", None)
    results_df.loc[row_n, "std_val_recall"] = stds.get("std_recall", None)
    results_df.loc[row_n, "val_mcc"] = averages.get("mcc", None)
    results_df.loc[row_n, "std_val_mcc"] = stds.get("std_mcc", None)

    # Save the updated results DataFrame to the CSV file
    results_df.to_csv(results_file, index=False)
