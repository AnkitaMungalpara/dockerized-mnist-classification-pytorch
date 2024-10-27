# Dockerized Deep Learning for MNIST Digit Classification with PyTorch


This project implements a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) on the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) using PyTorch. The project is containerized using Docker to ensure easy setup and consistent environments across different machines. The script allows training the model from scratch, resuming training from a checkpoint, and evaluating the model's performance.


## Table of Contents
- [Overview](#overview)
- [What is Docker?](#what-is-docker)
- [Why use Docker?](#why-use-docker)
- [Requirements](#requirements)
- [Docker Setup](#docker-setup)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running the Docker Container](#running-the-docker-container)
  - [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Training Script Arguments](#training-script-arguments)
- [Model Architecture](#model-architecture)
- [Data Loading and Transformations](#data-loading-and-transformations)
- [Model Initialization](#model-initialization)
- [Checkpoint Loading and Saving](#checkpoint-loading-and-saving)
- [Training and Evaluation Loop](#training-and-evaluation-loop)
- [Results](#results)

## Overview
The goal of this project is to classify handwritten digits (0-9) from the MNIST dataset using a Convolutional Neural Network (CNN). The project uses [PyTorch](https://pytorch.org/) for the model implementation, and [Docker](https://www.docker.com/) is used to containerize the application for ease of use and portability.

### MNIST Dataset
The MNIST (Modified National Institute of Standards and Technology) dataset is a database of handwritten digits that is usually used for training multiple image processing systems. Here are some key details about the dataset:

- **Content**: 28x28 grayscale images of handwritten digits (0-9)
- **Size**:
  - 60,000 training images
  - 10,000 test images
- **Format**: Each image is represented as a 2D PyTorch tensor
- **Labels**: Each image is associated with a label (0-9)
- **Source**: The dataset is built into PyTorch and can be easily downloaded using `torchvision.datasets.MNIST`

In this project, we use PyTorch's `torchvision.datasets.MNIST` to download and load the MNIST dataset. The data is normalized and transformed into PyTorch tensors for training and testing.

## What is Docker?

Docker is an open-source platform that automates the deployment of applications in lightweight, portable containers. These containers package an application and all of its dependencies, ensuring it runs the same regardless of the environment. Docker provides a way to isolate applications from the underlying system, preventing dependency conflicts and making it easier to manage and deploy applications across different systems.

## Why use Docker?

Setting up environments for machine learning and deep learning projects can be challenging because of dependencies on hardware (such as CUDA for GPUs) and incompatibilities across Python versions and libraries. Docker offers a self-contained environment that resolves such issues.

For this project, Docker is especially useful because of the following:

- **Environment Consistency**: Every user runs the project in exactly the same environment. This solves the "`it works on my machine`" conundrum.
- **Easy Setup**: PyTorch, torchvision, and other dependencies don't need to be manually installed when using Docker.
- **Reproducibility**: By specifying dependencies in a `Dockerfile`, you can duplicate the environment required to perform the training pipeline.

## Requirements
To run this project, you need to have Docker installed on your system. The installation process varies depending on your operating system. Once installed, verify by running:

```bash
docker --version
```

## Docker Setup

### Dockerfile
Here's the Dockerfile for containerizing the MNIST training:

```dockerfile
FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 --no-cache-dir install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 --no-cache-dir install numpy==1.23.4

COPY train.py /workspace/ 

CMD ["python", "train.py"]
```

### Building the Docker Image
To build the Docker image for this project, navigate to the root directory of your project and run:

```bash
docker build --tag mnist-classifier .
```

### Running the Docker Container

To run the container for training, use the following command:

```bash
docker run --name mnist-container --rm -v $(pwd):/workspace mnist-classifier python /workspace/train.py
```

### Resuming from Checkpoint
To resume training from a saved checkpoint, mount the directory where the checkpoint is stored and pass the `--resume` argument:

```bash
docker run --name mnist-container --rm -v $(pwd):/workspace mnist-classifier python /workspace/train.py --resume
```


## Training Script Arguments

You can specify the following command-line arguments while running the training script:

<table>
  <tr>
    <th>Argument</th>
    <th>Default</th>
    <th>Type</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>--batch-size</code></td>
    <td>64</td>
    <td><code>int</code></td>
    <td>Input batch size for training.</td>
  </tr>
  <tr>
    <td><code>--test-batch-size</code></td>
    <td>1000</td>
    <td><code>int</code></td>
    <td>Input batch size for testing.</td>
  </tr>
  <tr>
    <td><code>--epochs</code></td>
    <td>15</td>
    <td><code>int</code></td>
    <td>Number of epochs to train.</td>
  </tr>
  <tr>
    <td><code>--lr</code></td>
    <td>0.001</td>
    <td><code>float</code></td>
    <td>Learning rate for the optimizer.</td>
  </tr>
  <tr>
    <td><code>--gamma</code></td>
    <td>0.7</td>
    <td><code>float</code></td>
    <td>Learning rate step gamma for the learning rate scheduler.</td>
  </tr>
  <tr>
    <td><code>--no-cuda</code></td>
    <td><code>False</code></td>
    <td><code>bool</code></td>
    <td>Disables CUDA (GPU) training.</td>
  </tr>
  <tr>
    <td><code>--no-mps</code></td>
    <td><code>False</code></td>
    <td><code>bool</code></td>
    <td>Disables macOS GPU training (MPS backend).</td>
  </tr>
  <tr>
    <td><code>--dry-run</code></td>
    <td><code>False</code></td>
    <td><code>bool</code></td>
    <td>Quickly check a single pass for debugging purposes.</td>
  </tr>
  <tr>
    <td><code>--seed</code></td>
    <td>1</td>
    <td><code>int</code></td>
    <td>Random seed for reproducibility.</td>
  </tr>
  <tr>
    <td><code>--log-interval</code></td>
    <td>10</td>
    <td><code>int</code></td>
    <td>Number of batches to wait before logging training status.</td>
  </tr>
  <tr>
    <td><code>--save-model</code></td>
    <td><code>True</code></td>
    <td><code>bool</code></td>
    <td>Save the model after each epoch.</td>
  </tr>
  <tr>
    <td><code>--resume</code></td>
    <td><code>True</code></td>
    <td><code>bool</code></td>
    <td>Resume training from the last checkpoint if available.</td>
  </tr>
</table>


## Model Architecture

Here’s the CNN architecture for MNIST digit classification:

```python

import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Defining the model architecture
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

### Data Loading and Transformations

The dataset undergoes normalization and is converted into tensors for PyTorch training and testing. Here’s the data-loading and transformation setup:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

training_data = datasets.MNIST(
    "../data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST("../data", train=False, transform=transform)

train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
```

### Model Initialization

Initializing the model and optimizer with parameters for training:

```python
import torch.optim as optim
from model import Net  # Assuming Net is your CNN architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
```

### Checkpoint Loading and Saving

To resume training from a checkpoint, the `--resume` argument is provided:

```python
import torch

model_checkpoint_path = "model_checkpoint.pth"
start_epoch = 1

# Loading the model checkpoint if 'resume' argument is True
if args.resume:
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

# Saving the model after each epoch
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
```

### Training and Evaluation Loop

The main training and evaluation loop, including checkpoint saving:

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

for epoch in range(start_epoch, args.epochs + 1):
    train_epoch(epoch, args, model, device, train_loader, optimizer)  # Training for one epoch
    test_epoch(model, device, test_loader)  # Testing on validation data
    scheduler.step()

    if args.save_model:
        save_checkpoint(model, optimizer, epoch, model_checkpoint_path)
```

## Results

After training the model for 15 epochs, the results are:

```
Train Epoch: 15 [59520/60000 (99%)]     
Loss: 0.000001

Test set: 
Average loss: 0.0306, 
Accuracy: 9927/10000 (99%)
```
