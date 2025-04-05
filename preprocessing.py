import torchaudio
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram
from torch.nn.functional import pad
from torch import save, Tensor
import os
import csv
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
def alignAudio(waveform: Tensor, numTimestamps: int = 8000)->Tensor:
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
        return pad(waveform, (padLeft, padRight))
    else: #case of longer time series to crop
        #central idx of current waveform tensor
        timestampCenter = currTimestamps // 2
        #get values (relative to the centerpoint) to crop
        cropLeft = numTimestamps // 2
        cropRight = numTimestamps - cropLeft
        return waveform[:, timestampCenter - cropLeft:timestampCenter+cropRight]

def preprocessFile(loadPath: str, melSpec: MelSpectrogram, resampleRate: int = 47600, 
                   numTimestamps: int = 8000)->Tensor:
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
    #alignment & cropping
    waveform = alignAudio(waveform, numTimestamps)
    #conversion to mel spectrogram
    melSpecOut = melSpec.forward(waveform)
    #print(melSpecOut.shape)
    return melSpecOut

def preprocessAudio(audioDir: str, outputDir: str, csvpath : str,
                    n_mels:int = 256, hop_length:int = 16, normalized:bool = True,
                    resampleRate:int = 47600, numTimestamps:int = 8191)->None:
    """
    Preprocess all sound files from audioDir and output them into outputDir as pytorch Tensors. 
    Also aggregate each file's corresponding label (represented by the subfolder) into a csv,
    Assumes one layer of subfolders exists (as generated from dataCollection.downloadDataset)
    n_mels, hop_length, and normalized correspond to the mel spectrogram parameters,
    resampleRate and numTimestamps affects the resampled rate and the number of timestampes respectively.

    Base: n_mels = 64, numTimestamps=8000, hop_length = 200 -> 41x64 spectrogram
    Base: n_mels = 41, numTimestamps=8000, hop_length = 126 -> 64x41 spectrogram
    Upscaled: n_mels = 256, numTimestamps=8191, hop_length = 16 -> 512x256 spectrogram
    """
    assert(os.path.exists(audioDir)) #check that that input dir exists
    if not os.path.exists(outputDir): # create output directory if none exists
        os.mkdir(outputDir)
    fileLabelPair = [["fileName", "audioDir"]]
    #instantiate melspectrogram
    melSpec= MelSpectrogram(n_mels = n_mels, hop_length = hop_length, normalized = normalized)

    for subfolder in os.listdir(audioDir): #iterate over each subfolder
        #path to the corresponding subfolder in the output director
        outSubdir = os.path.join(outputDir, subfolder)
        if not os.path.exists(outSubdir): # create output subdir if none exists
            os.mkdir(outSubdir)
        for file in os.listdir(os.path.join(audioDir, subfolder)):
            #associate each file with its label/parent directory name
            fileName = os.path.splitext(file)[0]
            fileLabelPair.append([fileName, subfolder])
            audioPath = os.path.join(audioDir, subfolder, file)
            #initialize the output file path
            outFileName = fileName + ".pt"
            outFilePath = os.path.join(outSubdir, outFileName)
            outSpectrogram = preprocessFile(audioPath, melSpec, resampleRate, numTimestamps)
            save(outSpectrogram, outFilePath)
        print(f"{subfolder} done!")
    #save label csv
    with open(csvpath, "w") as file:
        csvWriter = csv.writer(file)
        csvWriter.writerows(fileLabelPair)


if __name__ == "__main__":
    #frequency/ n_mels = 64; hop_length = 200
    #melspecConversion = MelSpectrogram(n_mels = 64, hop_length = 200, normalized = True)
    #path = os.path.join(os.getcwd(), "RawSoundData\\Atlantic Spotted Dolphin\\6102500D.wav")
    #preprocessFile(path, melspecConversion)
    preprocessAudio("RawSoundData", "preprocessed", "labels.csv")