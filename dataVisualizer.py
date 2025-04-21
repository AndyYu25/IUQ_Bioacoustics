import torch
import matplotlib.pyplot as plt
import os
import glob




def visualizeSpectrogram(mel_spec: torch.Tensor):
    # If it's on GPU, move it back to CPU
    if mel_spec.device.type != 'cpu':
        mel_spec = mel_spec.cpu()


    # 2. Convert to NumPy for plotting
    mel_spec_np = mel_spec.numpy()\
    
    #flatten matrix to a 2d matrix 
    mel_spec_np = mel_spec.squeeze(0)

    # 3. Plot
    plt.figure(figsize=(8, 4))
    # origin='lower' places low mel bins at the bottom
    # aspect='auto' lets matplotlib choose an aspect ratio that fills the figure
    plt.imshow(mel_spec_np, origin='lower', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Amplitude')
    plt.title('Mel-Spectrogram (41 × 64)')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Bin')
    plt.tight_layout()
    plt.show()

def visualizeSpectrogramDir(directory: str, numFiles: int = 36):
    # If it's on GPU, move it back to CPU
    pt_files = sorted(glob.glob(os.path.join(directory, '*.pt')))

    if len(pt_files) != numFiles:
        raise ValueError(f"Expected exactly {numFiles} .pt files, but found {len(pt_files)}")

    batch = []

    for file_path in pt_files:
        tensor = torch.load(file_path)
        
        # Move to CPU if necessary
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        # Expect shape: [1, 41, 64]
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"Expected shape [1, 41, 64], but got {tuple(tensor.shape)} in {file_path}")
        
        batch.append(tensor.squeeze(0))  # shape: [41, 64]

    # Stack into a single tensor: [16, 41, 64]
    batch_tensor = torch.stack(batch)

    # 2. Plotting
    fig, axes = plt.subplots(int(numFiles**0.5), int(numFiles**0.5), figsize=(12, 10))

    for idx, ax in enumerate(axes.flat):
        mel = batch_tensor[idx].numpy()
        im = ax.imshow(mel, origin='lower', aspect='auto', interpolation='nearest')
        ax.axis('off')

    # Add a single colorbar for the whole grid
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Amplitude')

    plt.suptitle(f'Gray Whale Preprocessed Mel-Spectrograms', fontsize=14)
    #plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

if __name__ == "__main__":
    path = torch.load('data\\preprocessed\\Atlantic Spotted Dolphin\\6102500A.pt')
    #visualizeSpectrogram(path)
    rootDir = 'data\\preprocessed\\Gray Whale'
    visualizeSpectrogramDir(rootDir, 36)