# IUQ_Bioacoustics
CSE8803-IUQ term project on performing uncertainty quantification with respect to deep learning approaches to classifying bioacoustics sounds. 

## Installation

Create an environment with basic packages
```console
conda create -n bioacoustics python=3.10 "numpy<2.0" matplotlib seaborn pandas scikit-learn scipy
```

Activate the environment, then follow [these instructions](https://pytorch.org/get-started/locally/) to install `torch>=2.0`. For example
```console
conda activate bioacoustics
pip3 install torch torchvision torchaudio
```
