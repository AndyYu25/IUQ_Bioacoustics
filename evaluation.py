import torch
import time
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt

def compute_ece(softmax_probs, true_labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        softmax_probs: predicted probabilities (softmax outputs) - [n_samples, n_classes]
        true_labels: ground truth labels - [n_samples]
        n_bins: number of bins for confidence histogram
    
    Returns:
        ece_score: the expected calibration error
    """
    confidences = np.max(softmax_probs, axis=1)
    predictions = np.argmax(softmax_probs, axis=1)
    accuracies = (predictions == true_labels)
    
    # Bin the confidences
    bin_indices = np.linspace(0, 1, n_bins+1)
    ece = 0
    
    # Create arrays for plotting calibration
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_start = bin_indices[i]
        bin_end = bin_indices[i+1]
        
        # Find samples in this bin
        in_bin = np.logical_and(confidences > bin_start, confidences <= bin_end)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            # Add to ECE (weighted by bin size)
            ece += bin_count * np.abs(bin_accuracy - bin_confidence)
            
            # Store for plotting
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
    
    # Normalize by total number of samples
    ece /= len(softmax_probs)
    
    return ece, bin_accuracies, bin_confidences, bin_counts

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data loader
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        device: device to run inference on
    
    Returns:
        metrics: dictionary of evaluation metrics
    """
    model.eval()
    
    # Lists to store predictions and targets
    all_preds = []
    all_targets = []
    all_probs = []
    
    # For timing
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store predictions and targets
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Concatenate all probability arrays
    all_probs = np.vstack(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Calculate ECE
    ece, bin_accs, bin_confs, bin_counts = compute_ece(all_probs, all_targets)
    
    # Calculate entropy as uncertainty measure
    class_probs = all_probs
    entropies = -np.sum(class_probs * np.log(np.clip(class_probs, 1e-10, 1.0)), axis=1)
    mean_entropy = np.mean(entropies)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ece': ece,
        'mean_entropy': mean_entropy,
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / len(all_targets),
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts
    }
    
    return metrics

def plot_calibration_curve(bin_accuracies, bin_confidences, bin_counts, title='Calibration Curve'):
    """
    Plot calibration curve
    """
    plt.figure(figsize=(10, 8))
    
    # Plot the calibration curve
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot the calibration points
    plt.scatter(bin_confidences, bin_accuracies, 
                s=[count/10 for count in bin_counts],  # Size proportional to bin count
                alpha=0.8, 
                c='red', 
                label='Model Calibration')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Plot the histogram of confidences
    plt.figure(figsize=(10, 4))
    plt.bar(bin_confidences, bin_counts, width=0.1, alpha=0.5, color='blue')
    plt.xlabel('Confidence')
    plt.ylabel('Sample Count')
    plt.title('Confidence Histogram')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

