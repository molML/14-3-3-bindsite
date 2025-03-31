from runners.evaluate_embedding_models import held_out_perf_embed, test_perf_embed

"""
# Example

To evaluate the model performance on the best hyperparameters on the test set and hold out set, run the code below. You will need to specify the folder that contains the configurations and use the corresponsing function to get the test performance or held-out set performance. The performance file will be added to the respective configuration folder.
"""

"""Import necessary libraries"""


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
model_name = "mlp"
folder_with_files = "../examples/configurations/MLP_embed_BN_False"
test_perf_embed(folder_with_files=folder_with_files, model_name=model_name)
held_out_perf_embed(folder_with_files=folder_with_files, model_name=model_name)
