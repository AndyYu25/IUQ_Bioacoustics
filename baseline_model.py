import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        
        # 1x1 -> 3x3 convolution branch
        self.branch2_1 = nn.Conv2d(in_channels, ch3x3red, kernel_size=1)
        self.branch2_2 = nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        
        # 1x1 -> 5x5 convolution branch (implemented as two 3x3 convs)
        self.branch3_1 = nn.Conv2d(in_channels, ch5x5red, kernel_size=1)
        self.branch3_2 = nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        self.branch3_3 = nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1)
        
        # Max pooling -> 1x1 convolution branch
        self.branch4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        
        self.bn = nn.BatchNorm2d(ch1x1 + ch3x3 + ch5x5 + pool_proj)
        
    def forward(self, x):
        branch1 = F.relu(self.branch1(x))
        
        branch2 = F.relu(self.branch2_1(x))
        branch2 = F.relu(self.branch2_2(branch2))
        
        branch3 = F.relu(self.branch3_1(x))
        branch3 = F.relu(self.branch3_2(branch3))
        branch3 = F.relu(self.branch3_3(branch3))
        
        branch4 = self.branch4_1(x)
        branch4 = F.relu(self.branch4_2(branch4))
        
        # Concatenate branches along the channel dimension
        outputs = [branch1, branch2, branch3, branch4]
        output = torch.cat(outputs, 1)
        return F.relu(self.bn(output))


class MarineMammalInceptionNet(nn.Module):
    def __init__(self, num_classes, dropoutProbs=0.2):
        super(MarineMammalInceptionNet, self).__init__()
        
        # Initial convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception modules
        # in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj
        self.inception3a = InceptionBlock(64, 64, 48, 64, 16, 32, 32)
        self.inception3b = InceptionBlock(192, 128, 64, 96, 32, 64, 64)
        
        # Additional pooling after inception blocks
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate size after forward pass through convolutions and pooling
        # Input: (1, 64, 41) -> after conv, bn, pool layers and inception blocks:
        # Pool1 reduces to ~(32, 32, 21)
        # Pool2 reduces to ~(64, 16, 11)
        # Inception modules keep same dimensions but increase channels
        # Pool3 reduces to ~(352, 8, 6)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(352, 256)
        self.dropout = nn.Dropout(dropoutProbs)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers, w/ dropout
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# === BAYESIAN VERSION (Add-on only, baseline untouched) ===

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class MarineMammalBNN(nn.Module):
    def __init__(self, num_classes, dropoutProbs=0.2):
        super(MarineMammalBNN, self).__init__()

        # Same structure as baseline
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(64, 64, 48, 64, 16, 32, 32)
        self.inception3b = InceptionBlock(192, 128, 64, 96, 32, 64, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(352, 256)
        self.dropout = nn.Dropout(dropoutProbs)
        self.bayes_fc2 = BayesianLinear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.bayes_fc2(x)
