import os
import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class TensorClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_samples=1):
        """
        PyTorch Dataset for classification using pre-saved .pt tensor files.
        Preloads all tensors into memory during initialization.
        
        Args:
            root_dir (str): Directory with all the class folders
            transform (callable, optional): Optional transform to be applied on a sample
            min_samples (int): Minimum number of samples required for a class to be included
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.class_counts = {}
        
        # Get all class names (directories in root)
        all_dirs = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
        all_dirs.sort()  # Sort to ensure consistent class indices
        
        # Count samples per class and filter based on min_samples
        print(f"Scanning directories for class distribution...")
        for class_name in all_dirs:
            class_dir = os.path.join(root_dir, class_name)
            tensor_paths = glob.glob(os.path.join(class_dir, "*.pt"))
            sample_count = len(tensor_paths)
            self.class_counts[class_name] = sample_count
            
            if sample_count >= min_samples:
                self.class_names.append(class_name)
        
        # Create class to index mapping for valid classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Calculate total files to load
        total_files = sum(self.class_counts[cls] for cls in self.class_names)
        
        # Visualize class distribution before loading
        self.visualize_class_distribution()
        
        if min_samples > 1:
            filtered_classes = set(all_dirs) - set(self.class_names)
            print(f"Filtered out {len(filtered_classes)} classes with fewer than {min_samples} samples")
            
        # Collect and load all tensors and their corresponding labels with progress bar
        print(f"Preloading {total_files} tensor files from {len(self.class_names)} classes...")
        
        pbar = tqdm(total=total_files, desc="Loading tensors")
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            tensor_paths = glob.glob(os.path.join(class_dir, "*.pt"))
            for tensor_path in tensor_paths:
                # Load tensor directly into memory
                tensor = torch.load(tensor_path)
                self.data.append(tensor)
                self.labels.append(class_idx)
                pbar.update(1)
        
        pbar.close()
        print(f"Successfully loaded {len(self.data)} tensors")
    
    def visualize_class_distribution(self):
        """
        Visualize the distribution of samples across classes
        with a smart layout and color-coded bars.
        """
        plt.figure(figsize=(12, 8))
        
        # Get counts for all classes
        classes = list(self.class_counts.keys())
        counts = [self.class_counts[cls] for cls in classes]
        
        # Sort by count for better visualization
        sorted_indices = np.argsort(counts)[::-1]  # Descending order
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        # Highlight classes that will be kept vs filtered
        kept_mask = [c in self.class_names for c in sorted_classes]
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
                  f"{len(self.class_names)} Classes Kept ({total_kept} Samples)")
        
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
        print(f"  - Classes meeting min_samples threshold: {len(self.class_names)}")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Kept samples: {total_kept} ({total_kept/total_samples:.1%})")
        print(f"  - Min samples per class: {min(counts)}")
        print(f"  - Max samples per class: {max(counts)}")
        print(f"  - Avg samples per class: {total_samples/len(classes):.1f}")
    
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
        weight_tensor = torch.zeros(len(self.class_names))
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