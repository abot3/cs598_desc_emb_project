#!/bin/bash

# Maybe uncomment this to activate your conda env.
# conda activate <your_conda_env> 

# Make sure the BERT model is available in PyTorch
#conda config --add channels conda-forge
#conda install --yes -c conda-forge pytorch-pretrained-bert
#conda install -c huggingface transformers

pip install transformers
pip install pyhealth
pip install termcolor
pip install pandarallel
pip install colored
