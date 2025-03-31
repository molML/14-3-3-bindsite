# %% [markdown]
"""
# Example

To evaluate the model performance on the best hyperparameters on the test set and hold out set, run the code below.
You will need to specify the folder that contains the configurations and use the corresponsing function to get the test performance or held-out set performance.
The performance file will be added to the respective configuration folder. To keep the results manageable, the files have been zipped for submission, but unzipping them will give you the same results as in the paper.

"""

from runners.evaluate_fixed_encoding_models import (
    held_out_perf_fixed_encoding,
    test_perf_fixed_encoding,
)

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
"""The code below was run for all models and encodings.
In total you get the following combinations:
* MLP: 
    * ohe
        * batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_ohe_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_ohe_BN_False"
    * blosum
        * batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_blosum_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_blosum_BN_False"
    * handcrafted
        * batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_handcrafted_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_handcrafted_BN_False"
* CNN: 
    * ohe
        * batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_ohe_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_ohe_BN_False"
    * blosum
        * batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_blosum_BN_True" 
        * no batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_blosum_BN_False" 
    * embed
        * batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_embed_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_embed_BN_False"
    * handcrafted
        * batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_handcrafted_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_handcrafted_BN_False"
* GRU:  
    * ohe
        * batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gru_ohe_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gru_ohe_BN_False"
    * blosum
        * batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gru_blosum_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gru_blosum_BN_False"
    * handcrafted
        * batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gru_handcrafted_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gru_handcrafted_BN_False"
"""
model_name = "gru"
folder_with_files = (
    "../results/hyper_parameter_tuning/GRU/configurations/gru_handcrafted_BN_True"
)
test_perf_fixed_encoding(folder_with_files=folder_with_files, model_name=model_name)
held_out_perf_fixed_encoding(folder_with_files=folder_with_files, model_name=model_name)
