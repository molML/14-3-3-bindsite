from runners.evaluate_embedding_models import held_out_perf_embed, test_perf_embed

"""
# Final Performance Evaluation of Embedding Models
This script evaluates the performance of embedding models on the test set and hold-out set.
In this script you will find the specific model name and the folder that contains the configurations used for the final performance evaluation.
"""

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

"""The code below was run for all models and encodings. The path to the folder that contains the configurations is specified.
In total you get the following combinations:
* MLP: 
    * embed
        * batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_embed_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/MLP/configurations/mlp_embed_BN_False"
* CNN: 
    * embed
        * batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_embed_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/CNN/configurations/cnn_embed_BN_False"
* GRU:  
    * embed
        * batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gru_embed_BN_True"
        * no batch_normalisation: "../results/hyper_parameter_tuning/GRU/configurations/gryu_embed_BN_False"
"""
model_name = "gru"
folder_with_files = (
    "../results/hyper_parameter_tuning/GRU/configurations/gru_embed_BN_True"
)
test_perf_embed(folder_with_files=folder_with_files, model_name=model_name)
held_out_perf_embed(folder_with_files=folder_with_files, model_name=model_name)
