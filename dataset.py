import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def prepare_dataset_split(root_dir, test_size=0.2, min_samples=1, random_state=42):
    """
    Prepare stratified train-test split based on the file structure before loading any tensors.
    
    Args:
        root_dir (str): Root directory containing class folders with .pt files
        test_size (float): Fraction of data to use for testing (0.0 to 1.0)
        min_samples (int): Minimum number of samples required for a class to be included
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Information needed to initialize training and test datasets
    """
    # Get all class directories
    all_classes = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
    all_classes.sort()  # Sort for consistency
    
    # Collect file paths and labels
    all_file_paths = []
    all_labels = []
    class_counts = {}
    valid_classes = []
    
    print("Scanning directories...")
    for class_name in all_classes:
        class_dir = os.path.join(root_dir, class_name)
        file_paths = glob.glob(os.path.join(class_dir, "*.pt"))
        count = len(file_paths)
        class_counts[class_name] = count
        
        # Filter based on min_samples
        if count >= min_samples:
            valid_classes.append(class_name)
            all_file_paths.extend(file_paths)
            all_labels.extend([class_name] * count)
    
    # Create class to index mapping
    class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
    
    # Convert string labels to indices
    all_indices = [class_to_idx[label] for label in all_labels]
    
    # Visualize class distribution
    visualize_class_distribution(class_counts, valid_classes, min_samples)
    
    # Perform stratified split
    train_files, test_files, train_indices, test_indices = train_test_split(
        all_file_paths, all_indices, 
        test_size=test_size,
        random_state=random_state,
        stratify=all_indices  # Ensure balanced split
    )
    
    # Verify stratification
    train_counter = Counter(train_indices)
    test_counter = Counter(test_indices)
    
    print("Class distribution in split:")
    for cls_name, idx in class_to_idx.items():
        train_count = train_counter[idx]
        test_count = test_counter[idx]
        total = train_count + test_count
        print(f"  - {cls_name}: Train {train_count}/{total} ({train_count/total:.1%}), Test {test_count}/{total} ({test_count/total:.1%})")
    
    # Return all information needed to initialize datasets
    return {
        'train_files': train_files,
        'test_files': test_files,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'class_to_idx': class_to_idx,
        'valid_classes': valid_classes,
        'total_train_samples': len(train_files),
        'total_test_samples': len(test_files)
    }


def visualize_class_distribution(class_counts, valid_classes, min_samples):
    """
    Visualize the distribution of samples across classes.
    
    Args:
        class_counts (dict): Dict mapping class names to sample counts
        valid_classes (list): List of classes that meet the min_samples criteria
        min_samples (int): Minimum samples threshold used for filtering
    """
    plt.figure(figsize=(12, 8))
    
    # Get counts for all classes
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    
    # Sort by count for better visualization
    sorted_indices = np.argsort(counts)[::-1]  # Descending order
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    # Highlight classes that will be kept vs filtered
    kept_mask = [c in valid_classes for c in sorted_classes]
    kept_colors = ['#2ecc71' if kept else '#e74c3c' for kept in kept_mask]
    
    # Create horizontal bar chart
    bars = plt.barh(sorted_classes, sorted_counts, color=kept_colors)
    
    # Add count annotations on bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + (max(counts) * 0.01), 
            bar.get_y() + bar.get_height()/2, 
            str(sorted_counts[i]),
            va='center'
        )
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Included Classes'),
        Patch(facecolor='#e74c3c', label='Filtered Classes')
    ]
    plt.legend(handles=legend_elements)
        
    # Add summary statistics
    total_samples = sum(counts)
    total_kept = sum(c for c, k in zip(sorted_counts, kept_mask) if k)
    plt.title(f"Class Distribution - {len(classes)} Classes, {total_samples} Total Samples\n"
              f"{len(valid_classes)} Classes Kept ({total_kept} Samples)")
    
    plt.xlabel("Number of Samples")
    plt.ylabel("Class Name")
    plt.tight_layout()
    
    # Adjust layout based on number of classes
    if len(classes) > 20:
        plt.figure(figsize=(12, max(8, len(classes) * 0.3)))
        plt.subplots_adjust(left=0.3)  # Make more room for class names
        
    plt.show()
    
    # Print summary statistics
    print(f"Dataset Summary:")
    print(f"  - Total classes: {len(classes)}")
    print(f"  - Classes meeting min_samples threshold: {len(valid_classes)}")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Kept samples: {total_kept} ({total_kept/total_samples:.1%})")
    print(f"  - Min samples per class: {min(counts)}")
    print(f"  - Max samples per class: {max(counts)}")
    print(f"  - Avg samples per class: {total_samples/len(classes):.1f}")


class TensorClassificationDataset(Dataset):
    def __init__(self, file_paths, labels, class_to_idx, transform=None, 
                 compute_normalization=False, mean=None, std=None):
        """
        PyTorch Dataset for classification using pre-saved .pt tensor files.
        
        Args:
            file_paths (list): List of file paths to .pt files
            labels (list): List of class indices corresponding to file_paths
            class_to_idx (dict): Mapping from class name to index
            transform (callable, optional): Transform to apply to tensors
            compute_normalization (bool): Whether to compute mean and std for normalization
            mean (torch.Tensor, optional): Mean tensor for normalization
            std (torch.Tensor, optional): Std tensor for normalization
        """
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        self.transform = transform
        self.data = []
        
        # Normalization parameters
        self.mean = mean
        self.std = std
        
        # Load all tensors into memory
        print(f"Loading {len(file_paths)} tensors into memory...")
        for path in tqdm(file_paths):
            tensor = torch.load(path)
            self.data.append(tensor)
        
        # Compute normalization parameters if needed
        if compute_normalization:
            self.compute_normalization()
    
    def compute_normalization(self):
        """Calculate the channel-wise mean and standard deviation of the dataset."""
        print("Computing normalization parameters...")
        # Stack tensors along the batch dimension
        all_data = torch.stack(self.data, dim=0)
        
        # Calculate mean and std along batch dimension (dim=0)
        # For shape (batch, channels, height, width)
        self.mean = torch.mean(all_data, dim=(0, 2, 3))
        self.std = torch.std(all_data, dim=(0, 2, 3))
        
        print(f"Normalization parameters:")
        print(f"  - Mean: {self.mean}")
        print(f"  - Std: {self.std}")
        
        return self.mean, self.std
    
    def get_class_weights(self):
        """
        Calculate class weights inversely proportional to class frequencies.
        Useful for handling class imbalance in loss functions.
        
        Returns:
            torch.Tensor: Tensor of class weights
        """
        label_counter = Counter(self.labels)
        
        # Calculate weights as 1/frequency
        weights = {}
        total_samples = len(self.labels)
        for label, count in label_counter.items():
            weights[label] = total_samples / (len(label_counter) * count)
        
        # Convert to tensor in class index order
        weight_tensor = torch.zeros(len(self.class_to_idx))
        for label, weight in weights.items():
            weight_tensor[label] = weight
            
        return weight_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tensor = self.data[idx]
        label = self.labels[idx]
        
        # Apply transforms if any
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label


def create_datasets(root_dir, test_size=0.2, min_samples=1, random_state=42):
    """
    Create train and test datasets with proper normalization.
    
    Args:
        root_dir (str): Root directory with class folders
        test_size (float): Fraction of data for testing
        min_samples (int): Minimum samples per class
        random_state (int): Random seed
        
    Returns:
        tuple: (train_dataset, test_dataset, class_to_idx)
    """
    # First prepare the split
    split_info = prepare_dataset_split(
        root_dir=root_dir,
        test_size=test_size,
        min_samples=min_samples,
        random_state=random_state
    )
    
    # Create training dataset first (with normalization computation)
    train_dataset = TensorClassificationDataset(
        file_paths=split_info['train_files'],
        labels=split_info['train_indices'],
        class_to_idx=split_info['class_to_idx'],
        compute_normalization=True
    )
    
    # Get normalization parameters from training set
    mean = train_dataset.mean
    std = train_dataset.std
    
    # Create transforms for both datasets
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x - mean.view(-1, 1, 1)) / std.view(-1, 1, 1))
    ])
    
    # Update train dataset transform
    train_dataset.transform = transform
    
    # Create test dataset with same normalization
    test_dataset = TensorClassificationDataset(
        file_paths=split_info['test_files'],
        labels=split_info['test_indices'],
        class_to_idx=split_info['class_to_idx'],
        mean=mean,
        std=std,
        transform=transform
    )
    
    return train_dataset, test_dataset, split_info['class_to_idx']
