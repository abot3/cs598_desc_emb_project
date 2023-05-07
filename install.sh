#!/bin/bash

# Maybe uncomment this to activate your conda env.
# conda activate <your_conda_env> 

# Make sure the BERT model is available in PyTorch
conda config --add channels conda-forge
conda install -c anaconda jupyter
conda install numpy pandas matplotlib
conda install -c huggingface transformers
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# If you want to use Pip.
# pip install transformers
# pip install pyhealth
# pip install termcolor
# pip install pandarallel
# pip install colored
