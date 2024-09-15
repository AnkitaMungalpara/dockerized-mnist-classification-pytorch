# Dockerized MNIST Digit Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch. The project is fully containerized using Docker to ensure easy setup and consistent environments across different machines. The script allows training the model from scratch, resuming training from a checkpoint, and evaluating the model's performance.

## Table of Contents
- [Overview](#overview)
- [What is Docker?](#What-is-Docker?)
- [Why use Docker?](#Why-use-Docker?)
- [Requirements](#Requirements)
- [Docker Setup](#docker-setup)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running the Docker Container](#running-the-docker-container)
  - [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Training Script Arguments](#Training-Script-Arguments)
- [Model Architecture](#model-architecture)
- [Testing](#Testing)
- [Results](#results)

## Overview
The goal of this project is to classify handwritten digits (0-9) from the MNIST dataset using a Convolutional Neural Network (CNN). The project uses PyTorch for the model implementation, and Docker is used to containerize the application for ease of use and portability.

### MNIST Dataset
The MNIST (Modified National Institute of Standards and Technology) dataset is a database of handwritten digits that is usually used for training multiple image processing systems. Here are some key details about the dataset:

- Content: 28x28 grayscale images of handwritten digits (0-9)
- Size:
  - 60,000 training images
  - 10,000 test images
- Format: Each image is represented as a 2D PyTorch tensor
- Labels: Each image is associated with a label (0-9)
- Source: The dataset is built into PyTorch and can be easily downloaded using `torchvision.datasets.MNIST`

You can also download the MNIST dataset directly from [here](https://pytorch.org/vision/stable/datasets.html#mnist).

In this project, we use PyTorch's `torchvision.datasets.MNIST` to download and load the MNIST dataset. The data is normalized and transformed into PyTorch tensors for training and testing.


## What is Docker?

Docker is an open-source platform that automates the deployment of applications in lightweight, portable containers. These containers package an application and all of its dependencies, ensuring it runs the same regardless of the environment. Docker provides a way to isolate applications from the underlying system, preventing dependency conflicts and making it easier to manage and deploy applications across different systems.

## Why use Docker?

Setting up environments for machine learning and deep learning projects can be challenging because of dependencies on hardware (such as CUDA for GPUs) and incompatibilities across Python versions and libraries. Docker offers a self-contained environment that resolves such issues.

For this project, Docker is especially useful because of the following:

- **Environment Consistency**: Regardless of the underlying operating system, every user runs the project in exactly the same environment. This solves the "`it works on my machine`" conundrum.
- **Easy Setup**: PyTorch, torchvision, and other dependencies don't need to be manually installed when using Docker.
- **Reproducibility**: By specifying dependencies in a `Dockerfile`, you can duplicate the environment required to perform the training pipeline.


## Requirements
To run this project, you need to have Docker installed on your system. The installation process varies depending on your operating system:

- For macOS:
  - Install Docker Desktop for Mac.
  - Download from: [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop).
- For Windows:
  - Install Docker Desktop with WSL 2 backend.
  - Download [Docker Desktop for Windows](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe?_gl=1*1c1toot*_gcl_au*MzY4MDUzNDQ5LjE3MjU5MDUxNTk.*_ga*MTQwOTc1NzA4My4xNzI1NDEyNzYx*_ga_XJWPQMJYHQ*MTcyNTkwNTE1OS4zLjEuMTcyNTkwNTMxMS4yMi4wLjA.). During installation, ensure that the "**Use WSL 2 based engine**" option is selected.
- For Linux:
  - Install Docker Engine.
  - Follow the instructions for your specific Linux distribution and install it from [Docker Desktop for Linux](https://docs.docker.com/desktop/install/linux/).

After installation, verify that Docker is correctly installed and running by opening a terminal or command prompt and running:

```bash
docker --version
```

## Docker Setup

### Building the Docker Image
To build the Docker image for this project, navigate to the root directory of your project and run the following command:

```bash
docker build --tag mnist-classifier .
```

This will create a Docker image named `mnist-classifier`.

### Running the Docker Container

To run the container for training, use the following command:

```bash
docker run --name DOCKER_CONTAINER_NAME --rm -v /path/to/root:/workspace mnist-classifier python /workspace/train.py
```

### Resuming from Checkpoint
To resume training from a saved checkpoint, mount the directory where the checkpoint is stored using the `-v` flag and pass the `--resume` argument:

```bash
docker run --name DOCKER_CONTAINER_NAME --rm -v /path/to/root:/workspace mnist-classifier python /workspace/train.py --resume
```
This will load the saved model from the checkpoint and continue training.

## Training Script Arguments

You can specify the following command-line arguments while running the training script.

The following table lists the command-line arguments with their default values, types, and descriptions.

| Argument             | Default | Type   | Description                                                       |
|----------------------|---------|--------|-------------------------------------------------------------------|
| `--batch-size`        | 64      | `int`  | Input batch size for training.                                     |
| `--test-batch-size`   | 1000    | `int`  | Input batch size for testing.                                      |
| `--epochs`            | 15      | `int`  | Number of epochs to train.                                         |
| `--lr`                | 0.001   | `float`| Learning rate for the optimizer.                                   |
| `--gamma`             | 0.7     | `float`| Learning rate step gamma for the learning rate scheduler.          |
| `--no-cuda`           | `False` | `bool` | Disables CUDA (GPU) training.                                      |
| `--no-mps`            | `False` | `bool` | Disables macOS GPU training (MPS backend).                         |
| `--dry-run`           | `False` | `bool` | Quickly check a single pass for debugging purposes.                |
| `--seed`              | 1       | `int`  | Random seed for reproducibility.                                   |
| `--log-interval`      | 10      | `int`  | Number of batches to wait before logging training status.          |
| `--save-model`        | `True`  | `bool` | Save the model after each epoch.                                   |
| `--resume`            | `True`  | `bool` | Resume training from the last checkpoint if available.             |



## Model Architecture

The model is a simple CNN that consists of two convolutional layers followed by two fully connected layers. After that, the output is passed through a `log-softmax` activation function for classification into one of the 10-digit classes.


## Results

After training the model for 10 epochs, we have got the following results: 

```
Train Epoch: 15 [59520/60000 (99%)]     
Loss: 0.000001

Test set: 
Average loss: 0.0306, 
Accuracy: 9927/10000 (99%)
```
