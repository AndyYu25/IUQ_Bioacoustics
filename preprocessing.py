import numpy as np
import torchaudio
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram
from torch.nn.functional import pad
import torch
import os
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
def alignAudio(waveform: torch.Tensor, numTimestamps: int = 8000)->torch.Tensor:
    """
    Aligns and centers the given audio to 8000 timestamps
    zero-pads shorter time series
    cuts longer time series to retain only the central points
    """
    currTimestamps = waveform.shape[1]
    #how many timestamps to add (positive) or remove (negative)
    deltaTimestamps = numTimestamps - currTimestamps
    if deltaTimestamps >= 0: #case of shorter time series to zero-pad
        #padding to add to each side
        padLeft = deltaTimestamps // 2
        padRight = deltaTimestamps - padLeft
        #return padded tensor
        print(padLeft, padRight)
        return pad(waveform, (padLeft, padRight))
    else: #case of longer time series to zero-pad
        #central idx of current waveform tensor
        timestampCenter = currTimestamps // 2
        #get values (relative to the centerpoint) to crop
        cropLeft = numTimestamps // 2
        cropRight = numTimestamps - cropLeft
        print(timestampCenter - cropLeft, timestampCenter+cropRight)
        return waveform[:, timestampCenter - cropLeft:timestampCenter+cropRight]

def preprocessFile(loadPath: str, melSpec: MelSpectrogram, resampleRate: int = 47600, 
                   numTimestamps: int = 8000)->torch.Tensor:
    """
    Performs preprocessing into a mel spectrogram on an individual file; returns the melSpectrogram\n
    loadPath: file path to load the audio from
    melSpec: MelSpectrogram class initialized with the requisite parameters
    resampleRate: the rate to resample the audio to, in hertz
    numTimestamps: the number of timestamps to alter the length to
    """
    waveform, sampleRate = torchaudio.load(loadPath)
    #resampling to 47.6 kHz
    waveform = resample(waveform, sampleRate, resampleRate)
    print(waveform.shape)
    #alignment & cropping
    waveform = alignAudio(waveform, numTimestamps)
    #conversion to mel spectrogram
    print(waveform.shape)
    melSpecOut = melSpec.forward(waveform)
    print(melSpecOut.shape)
    return melSpecOut

#frequency/ n_mels = 64; hop_length = 200
#melspecConversion = MelSpectrogram(n_mels = 64, hop_length = 200, normalized = True)
#path = os.path.join(os.getcwd(), "RawSoundData\\Atlantic Spotted Dolphin\\6102500D.wav")
#preprocessFile(path, melspecConversion)