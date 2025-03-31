"""
# Explanation of hyperparameter tuning script

This script demonstrates how the hyperparameter tuning process worked for this research specifically. The results can be found under results/hyper_parameter_tuning
"""

from runners.runner import run_model

"""
Model Name:
- mlp: Multi-Layer Perceptron
- cnn: Convolutional Neural Network
- gru: Recurrent Neural Network with Gated Recurrent Unit
Encoding: 
- ohe: One-Hot Encoding
- blosum: BLOSUM Encoding
- embed: Embedding Encoding
- handcrafted: Handcrafted Encoding \n
Batch Normalization:
- True: Use batch normalization
- False: Do not use batch normalization \n
Hyperparameter Space: 
- Path to the hyperparameter space file (CSV format), e.g., "../examples/hyperparameter_spaces/cnn_3_examples.csv"
- Ensure the file is suited for the model and includes the model name in the file
"""

"""The code below was run for all models and encodings.
The results can be found under Hyper_parameter_tuning. 
The hyperparameter files are written after the model name.
In total you get the following combinations:
* MLP: "../results/hyperparameter_tuning/MLP/mlp_hyperspace.csv"
    * ohe
        * batch_normalisation
        * no batch_normalisation
    * blosum
        * batch_normalisation
        * no batch_normalisation
    * embed
        * batch_normalisation
        * no batch_normalisation
    * handcrafted
        * batch_normalisation
        * no batch_normalisation
* CNN: "../results/hyperparameter_tuning/CNN/cnn_hyperspace.csv"
    * ohe
        * batch_normalisation
        * no batch_normalisation
    * blosum
        * batch_normalisation
        * no batch_normalisation
    * embed
        * batch_normalisation
        * no batch_normalisation
    * handcrafted
        * batch_normalisation
        * no batch_normalisation
* GRU:  "../results/hyperparameter_tuning/GRU/gru_hyperspace.csv"
    * ohe
        * batch_normalisation
        * no batch_normalisation
    * blosum
        * batch_normalisation
        * no batch_normalisation
    * embed
        * batch_normalisation
        * no batch_normalisation
    * handcrafted
        * batch_normalisation
        * no batch_normalisation
"""
## Run the model with the specified hyperparameter space and encoding type - do this for all models and encodings if you want to reproduce the results,
## make sure to change the model name and hyperparameter space file accordingly.
model_name = "gru"
hyper_space = "../results/hyperparameter_tuning/GRU/gru_hyperspace.csv"
encoding = "ohe"
batch_normalisation = True
run_model(model_name, hyper_space, encoding, batch_normalisation)
