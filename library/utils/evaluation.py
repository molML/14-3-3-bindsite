import typing
from statistics import mean, median, stdev

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def performance_calculator_nan_proof(
    predictions, y: np.array, threshold: float = 0.5
) -> typing.Dict[str, float]:
    """
    Function to calculate the performance of the model. It calculates the AUC, F1 score, accuracy, precision, recall,
    Matthews correlation coefficient, confusion matrix, balanced accuracy score, specificity and false positive rate.
    Args:
        predictions (tf.tensor): The predictions of the model.
        y (np.array): The true labels.
        threshold (float): The threshold to use for the predictions. Default is 0.5.
    Returns:
        results_ (typing.Dict[str, float]): A dictionary containing the performance metrics.
    """
    predictions = np.array(predictions)[: len(y)]

    auc = roc_auc_score(y, predictions)
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0

    f1 = f1_score(y, predictions)
    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    mcc = matthews_corrcoef(y, predictions)
    cm = confusion_matrix(y, predictions)
    bal_acc = balanced_accuracy_score(y, predictions)
    sp = cm[0][0] / (cm[0][0] + cm[0][1])
    FP_over_TP_FP = cm[0][1] / (cm[1][1] + cm[0][1])

    results_ = {
        "auc": auc,
        "f1": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "mcc": mcc,
        "cm": str(cm),
        "TN": int(float(cm[0][0])),
        "TP": int(float(cm[1][1])),
        "FP": int(float(cm[0][1])),
        "FN": int(float(cm[1][0])),
        "bal_acc": bal_acc,
        "sp": sp,
        "FP_over_TP_FP": FP_over_TP_FP,
    }
    return results_


def averages_calculator(json_s, n_setups):
    aucs = []
    f1s = []
    accs = []
    precs = []
    recalls = []
    mccs = []
    sps = []
    bal_accs = []

    for i in range(1, n_setups + 1):
        dictionary_now = json_s[i]["validation"]
        aucs.append(dictionary_now["auc"])
        f1s.append(dictionary_now["f1"])
        accs.append(dictionary_now["accuracy"])
        precs.append(dictionary_now["precision"])
        recalls.append(dictionary_now["recall"])
        mccs.append(dictionary_now["mcc"])
        sps.append(dictionary_now["sp"])
        bal_accs.append(dictionary_now["bal_acc"])

    try:
        averages = {
            "averages": {
                "auc": mean(aucs),
                "f1": mean(f1s),
                "accuracy": mean(accs),
                "precision": mean(precs),
                "recall": mean(recalls),
                "mcc": mean(mccs),
                "sp": mean(sps),
                "bal_acc": mean(bal_accs),
            },
            "stds": {
                "std_auc": stdev(aucs),
                "std_f1": stdev(f1s),
                "std_acc": stdev(accs),
                "std_prec": stdev(precs),
                "std_recall": stdev(recalls),
                "std_mcc": stdev(mccs),
                "std_sp": stdev(sps),
                "std_bal_acc": stdev(bal_accs),
            },
            "medians": {
                "auc": median(aucs),
                "f1": median(f1s),
                "accuracy": median(accs),
                "precision": median(precs),
                "recall": median(recalls),
                "mcc": median(mccs),
                "sp": median(sps),
                "bal_acc": median(bal_accs),
            },
            "iqrs": {
                "iqr_auc": float(np.percentile(aucs, [75]) - np.percentile(aucs, [25])),
                "std_f1": float(np.percentile(f1s, [75]) - np.percentile(f1s, [25])),
                "std_acc": float(np.percentile(accs, [75]) - np.percentile(accs, [25])),
                "std_prec": float(
                    np.percentile(precs, [75]) - np.percentile(precs, [25])
                ),
                "std_recall": float(
                    np.percentile(recalls, [75]) - np.percentile(recalls, [25])
                ),
                "std_mcc": float(np.percentile(mccs, [75]) - np.percentile(mccs, [25])),
                "std_sp": float(np.percentile(sps, [75]) - np.percentile(sps, [25])),
                "std_bal_acc": float(
                    np.percentile(bal_accs, [75]) - np.percentile(bal_accs, [25])
                ),
            },
        }
    except:
        averages = {
            "averages": {
                "auc": mean(aucs),
                "f1": mean(f1s),
                "accuracy": mean(accs),
                "precision": mean(precs),
                "recall": mean(recalls),
                "mcc": mean(mccs),
                "sp": mean(sps),
                "bal_acc": mean(bal_accs),
            },
            "stds": {
                "std_auc": 0,
                "std_f1": 0,
                "std_acc": 0,
                "std_prec": 0,
                "std_recall": 0,
                "std_mcc": 0,
                "std_sp": 0,
                "std_bal_acc": 0,
            },
            "medians": {
                "auc": median(aucs),
                "f1": median(f1s),
                "accuracy": median(accs),
                "precision": median(precs),
                "recall": median(recalls),
                "mcc": median(mccs),
                "sp": median(sps),
                "bal_acc": median(bal_accs),
            },
        }
    return averages


def pad_for_evaluation(xs):
    """
    Pads the input sequences to ensure they all have the same length.
    Args:
        xs (list): List of input sequences.
    Returns:
        inputs (list): List of padded input sequences.
    """
    inputs = []
    for x in xs:
        # Filling in units so that each x contains equal amounts of input
        # As otherwise the model cannot process it. It is filled with 0 vectors
        max_size = max([len(x) for x in xs])
        units_missing = max_size - len(x)
        try:  # for one hot encoding, blosum and handcrafted features
            append_tens = tf.zeros(
                (units_missing, x.shape[1], x.shape[2]), dtype=x.dtype
            )
        except:  # for embedding layer
            append_tens = tf.zeros((units_missing, x.shape[1]), dtype=x.dtype)
        input = tf.concat([x, append_tens], axis=0)
        inputs.append(input)
    return inputs
