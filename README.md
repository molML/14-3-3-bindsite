# 14-3-3 BindSite

Welcome to the official repository of 14-3-3 BindSite. The following steps will help you reproduce and use our study's results. 

> [!TIP]
> You can also use our web application to predict binding sites and conduct a virtual mutation study: [https://14-3-3-bindsite.streamlit.app](https://14-3-3-bindsite.streamlit.app/).

## Installation 

First, download this codebase by the download button (on the top-right corner), or clone the repository with the following command, if you have git installed:

```bash
git clone https://github.com/molML/14-3-3-bindsite.git
```

We'll use `conda` to create an environment for the codebase. If you haven't used conda before, you can follow [the official tutorial](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).


Afterward, start a terminal *in the (root) directory of the codebase* and type the following commands:

> [!IMPORTANT]
> A GPU is required for replicability.

```bash
conda create --name bindsite python==3.9.16
conda activate bindsite 
conda install --file requirements.txt -c conda-forge  
```

## Folder Structure

- `data/`: Contains dataset splits and the external evaluation dataset used in the study.
- `examples/`: Contains example scripts to load and evaluate models, as well as tuning hyper-parameters.
- `library/`: Contains the codebase. Encoding functions, model implementations, and utility functions are available here.
- `models/`: Contains the trained model weights.
- `results/`: Contains the output of the hyperparameter tuning, model scores, and permutation analysis.

## Examples 
3 examples are available in the examples folder. You can test encodings and model architectures used in the study. 

- [examples/hp_tuning.py](examples/hp_tuning.py): Running the hyperparameter tuning pipeline and some example hyperparameter spaces. 
- [examples/evaluation_fixed_encodings.py](examples/evaluation_fixed_encodings.py): Evaluating a pretrained model with fixed encodings (one-hot encoding, BLOSUM62, and handcrafted) on test sets
- [examples/evaluation_embedding.py](examples/evaluation_embedding.py): Evaluating a pretrained model with an embedding layer on test sets

The configurations used in the study are available under `reproducibility/` folder for these scripts.


##  Closing Remarks 

Thanks for visiting our repository! If you have any questions, please don't hesitate to open an issue. We'll be happy to help!

If you use this codebase in your research, please consider citing our paper:

```bibtex
 Preprint in progress.
```

