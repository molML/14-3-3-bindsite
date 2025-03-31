from runners.runner import run_model

"""
# Example Model Runner Hyperparameter Space

This script demonstrates how to run a CNN model using the `run_model` function from the `runners.runner` module.
It will train the model with a specified hyperparameter space and encoding type and save the results in a designated folder.
It specifies the model name, hyperparameter space, encoding type, and whether to use batch normalization.
The results of the model training will be saved under the configuration folder named after the model name, encoding type, and batch normalization setting.
"""


"""Example CNN Model Configuration: 
Specify the model name, hyperparameter space, encoding type, and batch normalization
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
model_name = "cnn"
hyper_space = "../examples/hyperparameter_spaces/cnn_3_examples.csv"
encoding = "ohe"
batch_normalisation = True
run_model(model_name, hyper_space, encoding, batch_normalisation)
