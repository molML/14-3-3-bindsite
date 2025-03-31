from tensorflow.keras.callbacks import Callback


class CustomEarlyStopping(Callback):
    """
    Custom early stopping based on the mean F1 score of multiple setups.
    This callback stops training when the mean F1 score does not improve for a specified number of epochs.
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        start_epoch (int): Epoch after which the early stopping will start monitoring.
        num_setups (int): Number of setups to monitor.
        row_n (int): Row number for multi-setup training, this is important because the row_n will be appended
        to the keys of the metrics in the logs.
    """

    def __init__(self, patience, start_epoch, num_setups, row_n):
        super(CustomEarlyStopping, self).__init__()
        self.num_setups = num_setups
        self.patience = patience
        self.start_epoch = start_epoch
        self.wait = 0
        self.best_mean_f1 = 0
        self.row_n = row_n

    def on_epoch_end(self, epoch, logs=None):
        row_n = self.row_n
        if epoch >= self.start_epoch:
            setups_keys = ["setup_" + str(i) for i in range(self.num_setups)]
            if row_n == 0:
                recall_keys = [
                    "val_" + setup_key + "_recall" for setup_key in setups_keys
                ]
                precision_keys = [
                    "val_" + setup_key + "_precision" for setup_key in setups_keys
                ]

            else:
                recall_keys = [
                    "val_" + setup_key + "_recall_" + str(row_n)
                    for setup_key in setups_keys
                ]
                precision_keys = [
                    "val_" + setup_key + "_precision_" + str(row_n)
                    for setup_key in setups_keys
                ]

            recalls = [logs.get(recall_key) for recall_key in recall_keys]
            precisions = [logs.get(precision_key) for precision_key in precision_keys]

            try:
                f1_scores = [
                    2 * recall * precision / (precision + recall)
                    for recall, precision in zip(recalls, precisions)
                ]
            except:
                f1_scores = [0 for _, _ in zip(recalls, precisions)]

            current_mean_f1 = sum(f1_scores) / self.num_setups
            if current_mean_f1 is not None:
                if current_mean_f1 > self.best_mean_f1:
                    self.best_mean_f1 = current_mean_f1
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.model.stop_training = True
            else:
                self.wait = 0
