# %% [markdown]
"""
# Example

To evaluate the model performance on the best hyperparameters on the test set and hold out set, run the code below. You will need to specify the folder that contains the configurations and use the corresponsing function to get the test performance or held-out set performance. The performance file will be added to the respective configuration folder.
"""

from runners.evaluate_fixed_encoding_models import (
    held_out_perf_fixed_encoding,
    test_perf_fixed_encoding,
)

# %% [markdown]
"""
The code below will evaluate the model performance on the test set and hold out set.
The performance file will be added to the respective configuration folder. 
The parameters for the function are:
Folder_with_files: 
- Path to the folder that contains the configurations
Model Name:
- mlp: Multi-Layer Perceptron
- cnn: Convolutional Neural Network
- gru: Recurrent Neural Network with Gated Recurrent Unit
"""
model_name = "cnn"
folder_with_files = "../examples/configurations/CNN_ohe_BN_True"
test_perf_fixed_encoding(folder_with_files=folder_with_files, model_name=model_name)
held_out_perf_fixed_encoding(folder_with_files=folder_with_files, model_name=model_name)
# %%
