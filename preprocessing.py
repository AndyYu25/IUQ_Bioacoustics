import numpy as np
import torchaudio
import torch
"""
Preprocessing approach:
Liccardi & Carbone 2024: https://arxiv.org/html/2402.17775v2 - Mel Spectrograms only
1. Resample at 47.6kHz
2. Align + Center time series to 8000 timestamps
    a. zero-pad shorter time series
    b. retain only central points
3. Calculate Mel Spectrogram
    a. Frequency fixed at 64
    b. hop-length set to 200
Output: 64x41 images
"""

