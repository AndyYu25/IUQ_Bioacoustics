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

## Downloading Processed Data

Download data from [here](https://drive.google.com/file/d/1WUgfFip5xvxQV-RqWRaHCRYuCwqBCGL7/view). Optionally, use `gdown` to download directly using the file ID. Place the data in a `data` directory then unzip.

```console
pip3 install gdown
mkdir data && cd data
gdown 1WUgfFip5xvxQV-RqWRaHCRYuCwqBCGL7
unzip preprocessed.zip
```
